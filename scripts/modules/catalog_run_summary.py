"""Summarize decoded run catalogs under analysis/results/."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modules.catalog_merge import default_output_root
from modules.catalog_plot_common import is_spark_row, rows_to_fft_samples
from modules.event_catalog_io import CATALOG_CSV_NAME, parse_timestamp_cell, read_catalog_rows

_RUN_DIR_RE = re.compile(r"^run(\d+)$", re.IGNORECASE)
DASH = "—"


@dataclass(frozen=True)
class RunSummary:
    run: str
    n_signal: int
    n_noise: int
    n_spark: int
    duration_h: float | None
    spark_per_h: float | None
    sn_ratio: float | None
    mean_integ: float | None
    mean_maxv: float | None


def _run_sort_key(path: Path) -> tuple[int, str]:
    name = path.parent.name
    m = _RUN_DIR_RE.match(name)
    if m:
        return int(m.group(1)), name
    return 10**9, name


def discover_run_catalogs(results_root: Path | None = None) -> list[Path]:
    """``results/run*/event_catalog.csv`` を run 番号順に返す。"""
    root = default_output_root() if results_root is None else results_root
    if not root.is_dir():
        return []
    paths = [p for p in root.glob("run*/event_catalog.csv") if p.is_file()]
    return sorted(paths, key=_run_sort_key)


def summarize_catalog_rows(rows: list[dict[str, Any]], *, run: str) -> RunSummary:
    """1 カタログ分の集計。"""
    int_sig, max_sig, int_noise, max_noise = rows_to_fft_samples(rows)
    n_spark = sum(1 for row in rows if is_spark_row(row))

    times = []
    for row in rows:
        ts = parse_timestamp_cell(row.get("timestamp", ""))
        if ts is not None:
            times.append(ts)

    duration_h: float | None = None
    if len(times) >= 2:
        duration_h = (max(times) - min(times)).total_seconds() / 3600.0
    elif len(times) == 1:
        duration_h = 0.0

    spark_per_h: float | None = None
    if duration_h is not None and duration_h > 0.0:
        spark_per_h = n_spark / duration_h

    sn_ratio: float | None = None
    if int_noise.size > 0:
        sn_ratio = float(int_sig.size) / float(int_noise.size)
    elif int_sig.size > 0:
        sn_ratio = math.inf

    mean_integ = float(np.mean(int_sig)) if int_sig.size > 0 else None
    mean_maxv = float(np.mean(max_sig)) if max_sig.size > 0 else None

    return RunSummary(
        run=run,
        n_signal=int(int_sig.size),
        n_noise=int(int_noise.size),
        n_spark=n_spark,
        duration_h=duration_h,
        spark_per_h=spark_per_h,
        sn_ratio=sn_ratio,
        mean_integ=mean_integ,
        mean_maxv=mean_maxv,
    )


def summarize_catalog_path(csv_path: Path) -> RunSummary:
    """CSV パスから run ラベル付きで集計。"""
    run = csv_path.parent.name if csv_path.name == CATALOG_CSV_NAME else csv_path.stem
    return summarize_catalog_rows(read_catalog_rows(csv_path), run=run)


def _fmt_duration(duration_h: float | None) -> str:
    if duration_h is None:
        return DASH
    return f"{duration_h:.2f} h"


def _fmt_float(value: float | None, *, precision: int = 2) -> str:
    if value is None:
        return DASH
    if math.isinf(value):
        return "inf"
    return f"{value:.{precision}g}"


def format_summary_table(summaries: list[RunSummary]) -> str:
    """固定幅のサマリー表（TOTAL 行付き）。"""
    header = (
        f"{'run':<10} {'signal':>7} {'noise':>7} {'spark':>6} "
        f"{'duration':>9} {'spark/h':>8} {'S/N':>8} {'⟨integ⟩':>8} {'⟨maxv⟩':>8}"
    )
    lines = [header]

    tot_sig = tot_noise = tot_spark = 0
    for s in summaries:
        tot_sig += s.n_signal
        tot_noise += s.n_noise
        tot_spark += s.n_spark
        lines.append(
            f"{s.run:<10} {s.n_signal:>7} {s.n_noise:>7} {s.n_spark:>6} "
            f"{_fmt_duration(s.duration_h):>9} {_fmt_float(s.spark_per_h):>8} "
            f"{_fmt_float(s.sn_ratio, precision=3):>8} "
            f"{_fmt_float(s.mean_integ):>8} {_fmt_float(s.mean_maxv):>8}"
        )

    if summaries:
        lines.append(
            f"{'TOTAL':<10} {tot_sig:>7} {tot_noise:>7} {tot_spark:>6} "
            f"{DASH:>9} {DASH:>8} {DASH:>8} {DASH:>8} {DASH:>8}"
        )

    return "\n".join(lines)


def collect_run_summaries(results_root: Path | None = None) -> list[RunSummary]:
    """探索して各 run のサマリーを返す。"""
    summaries: list[RunSummary] = []
    for csv_path in discover_run_catalogs(results_root):
        try:
            summaries.append(summarize_catalog_path(csv_path))
        except Exception as exc:
            print(f"[skip] failed to read {csv_path}: {exc}", file=sys.stderr, flush=True)
    return summaries


def main() -> None:
    """CLI: 引数なしで ``results/run*`` カタログのサマリー表を stdout に出力。"""
    summaries = collect_run_summaries()
    if not summaries:
        print("No run catalogs found under results/run*/event_catalog.csv", file=sys.stderr)
        sys.exit(1)
    print(format_summary_table(summaries), flush=True)


if __name__ == "__main__":
    main()
