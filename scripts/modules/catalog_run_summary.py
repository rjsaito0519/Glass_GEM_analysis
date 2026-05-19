"""Summarize decoded run catalogs under analysis/results/."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.catalog_merge import default_output_root
from modules.catalog_plot_common import is_spark_row, rows_to_fft_samples
from modules.event_catalog_io import (
    CATALOG_CSV_NAME,
    CATALOG_META_NAME,
    parse_timestamp_cell,
    read_catalog_rows,
)

_RUN_DIR_RE = re.compile(r"^run(\d+)$", re.IGNORECASE)
_RUN_COMMENT_LINE_RE = re.compile(r"\[(run\d+)\]\s*(.*)$", re.IGNORECASE)
DASH = "—"
SPARK_RATE_REF_MIN = 10.0
COMMENT_COL_WIDTH = 40
MISC_COMMENTS_NAME = "comments.txt"


@dataclass(frozen=True)
class RunSummary:
    run: str
    n_signal: int
    n_noise: int
    n_spark: int
    duration_h: float | None
    spark_per_10min: float | None
    sn_ratio: float | None
    comment: str


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


def misc_comments_path_from_run_dir(run_dir: Path) -> Path:
    """データキャンペーン ``misc/comments.txt`` のパス。"""
    return run_dir.resolve().parent / "misc" / MISC_COMMENTS_NAME


def load_misc_comments(comments_path: Path) -> dict[str, str]:
    """``misc/comments.txt`` を ``{run0093: comment, ...}`` に読む。"""
    if not comments_path.is_file():
        return {}
    out: dict[str, str] = {}
    text = comments_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = _RUN_COMMENT_LINE_RE.search(line)
        if not m:
            continue
        run_key = m.group(1).lower()
        comment = m.group(2).strip()
        if comment:
            out[run_key] = comment
    return out


def run_dir_from_catalog_meta(csv_path: Path) -> Path | None:
    """``event_catalog.meta.json`` の ``run_dir`` を返す。"""
    meta_path = csv_path.parent / CATALOG_META_NAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_dir = meta.get("run_dir")
    if not run_dir:
        return None
    return Path(str(run_dir))


def lookup_run_comment(
    csv_path: Path,
    *,
    run: str,
    comments_cache: dict[Path, dict[str, str]],
) -> str:
    """run ディレクトリ上の ``misc/comments.txt`` からコメントを引く。"""
    run_dir = run_dir_from_catalog_meta(csv_path)
    if run_dir is None:
        return DASH
    comments_path = misc_comments_path_from_run_dir(run_dir)
    if comments_path not in comments_cache:
        comments_cache[comments_path] = load_misc_comments(comments_path)
    comment = comments_cache[comments_path].get(run.lower(), "")
    return comment if comment else DASH


def summarize_catalog_rows(
    rows: list[dict[str, Any]],
    *,
    run: str,
    comment: str = DASH,
) -> RunSummary:
    """1 カタログ分の集計。"""
    int_sig, _, int_noise, _ = rows_to_fft_samples(rows)
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

    spark_per_10min: float | None = None
    if duration_h is not None and duration_h > 0.0:
        duration_min = duration_h * 60.0
        spark_per_10min = n_spark / duration_min * SPARK_RATE_REF_MIN

    sn_ratio: float | None = None
    if int_noise.size > 0:
        sn_ratio = float(int_sig.size) / float(int_noise.size)
    elif int_sig.size > 0:
        sn_ratio = math.inf

    return RunSummary(
        run=run,
        n_signal=int(int_sig.size),
        n_noise=int(int_noise.size),
        n_spark=n_spark,
        duration_h=duration_h,
        spark_per_10min=spark_per_10min,
        sn_ratio=sn_ratio,
        comment=comment,
    )


def summarize_catalog_path(
    csv_path: Path,
    *,
    comments_cache: dict[Path, dict[str, str]] | None = None,
) -> RunSummary:
    """CSV パスから run ラベル付きで集計。"""
    run = csv_path.parent.name if csv_path.name == CATALOG_CSV_NAME else csv_path.stem
    cache: dict[Path, dict[str, str]] = {} if comments_cache is None else comments_cache
    comment = lookup_run_comment(csv_path, run=run, comments_cache=cache)
    return summarize_catalog_rows(read_catalog_rows(csv_path), run=run, comment=comment)


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


def _fmt_comment(comment: str) -> str:
    if not comment or comment == DASH:
        return DASH
    if len(comment) <= COMMENT_COL_WIDTH:
        return comment
    return comment[: COMMENT_COL_WIDTH - 1] + "…"


def format_summary_table(summaries: list[RunSummary]) -> str:
    """固定幅のサマリー表。"""
    header = (
        f"{'run':<10} {'signal':>7} {'noise':>7} {'spark':>6} "
        f"{'duration':>9} {'spk/10m':>8} {'S/N':>8}  comment"
    )
    lines = [header]

    for s in summaries:
        lines.append(
            f"{s.run:<10} {s.n_signal:>7} {s.n_noise:>7} {s.n_spark:>6} "
            f"{_fmt_duration(s.duration_h):>9} {_fmt_float(s.spark_per_10min):>8} "
            f"{_fmt_float(s.sn_ratio, precision=3):>8}  {_fmt_comment(s.comment)}"
        )

    return "\n".join(lines)


def collect_run_summaries(results_root: Path | None = None) -> list[RunSummary]:
    """探索して各 run のサマリーを返す。"""
    summaries: list[RunSummary] = []
    comments_cache: dict[Path, dict[str, str]] = {}
    for csv_path in discover_run_catalogs(results_root):
        try:
            summaries.append(summarize_catalog_path(csv_path, comments_cache=comments_cache))
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
