"""Plot integral / vmax histograms from decoded event catalog CSV files."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from modules.event_catalog_io import (
    parse_float_cell,
    parse_fft_noise_cell,
    read_catalog_rows,
    resolve_catalog_paths,
)

INTEGRAL_BIN_EDGES = np.linspace(0.0, 200.0, 101)
MAX_BIN_EDGES = np.linspace(0.0, 400.0, 51)

LABEL_SIGNAL = "signal"
LABEL_NOISE = "noise"


def _is_spark_row(row: dict[str, Any]) -> bool:
    s = (row.get("spark") or "").strip()
    return s == "1"


def rows_to_fft_samples(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """非スパークかつ FFT 分類済み行から (int_sig, max_sig, int_noise, max_noise)。"""
    int_sig: list[float] = []
    max_sig: list[float] = []
    int_noise: list[float] = []
    max_noise: list[float] = []
    for row in rows:
        if _is_spark_row(row):
            continue
        fft_lab = parse_fft_noise_cell(row.get("fft_is_noise", ""))
        if fft_lab is None:
            continue
        integral = parse_float_cell(row.get("integral_mv_us", ""))
        vmax = parse_float_cell(row.get("vmax_mv", ""))
        if not np.isfinite(integral) or not np.isfinite(vmax):
            continue
        if fft_lab:
            int_noise.append(integral)
            max_noise.append(vmax)
        else:
            int_sig.append(integral)
            max_sig.append(vmax)
    return (
        np.asarray(int_sig, dtype=float),
        np.asarray(max_sig, dtype=float),
        np.asarray(int_noise, dtype=float),
        np.asarray(max_noise, dtype=float),
    )


def rows_signal_by_run(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """signal のみを ``run_num`` ごとに分ける。"""
    per_run: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row in rows:
        if _is_spark_row(row):
            continue
        if parse_fft_noise_cell(row.get("fft_is_noise", "")) is not False:
            continue
        integral = parse_float_cell(row.get("integral_mv_us", ""))
        vmax = parse_float_cell(row.get("vmax_mv", ""))
        if not np.isfinite(integral) or not np.isfinite(vmax):
            continue
        run_num = (row.get("run_num") or "").strip() or "?"
        ints, maxs = per_run[run_num]
        ints.append(integral)
        maxs.append(vmax)
    return {
        k: (np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float))
        for k, v in per_run.items()
    }


def _draw_step_series(
    ax,
    series: list[tuple[np.ndarray, str, str]],
    *,
    bins: np.ndarray,
    density: bool,
    ylabel: str,
    xlabel: str,
) -> None:
    any_data = any(arr.size > 0 for arr, _, _ in series)
    if not any_data:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
    else:
        for arr, color, lab in series:
            if arr.size > 0:
                ax.hist(
                    arr,
                    bins=bins,
                    histtype="step",
                    linewidth=1.2,
                    color=color,
                    label=lab,
                    density=density,
                )
        ax.legend(loc="upper right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def plot_combined(
    int_sig: np.ndarray,
    max_sig: np.ndarray,
    int_noise: np.ndarray,
    max_noise: np.ndarray,
) -> None:
    """全 run 結合: signal + noise、カウント表示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[plot_event_catalog] matplotlib required: pip install 'glassgem-analysis[viz]'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    from modules.project_matplotlib_rc import MPL_RC

    series_int = [
        (int_sig, "C0", LABEL_SIGNAL),
        (int_noise, "C3", LABEL_NOISE),
    ]
    series_max = [
        (max_sig, "C0", LABEL_SIGNAL),
        (max_noise, "C3", LABEL_NOISE),
    ]

    with plt.rc_context(MPL_RC):
        _fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.35})
        _draw_step_series(
            axes[0],
            series_int,
            bins=INTEGRAL_BIN_EDGES,
            density=False,
            ylabel="Counts",
            xlabel="Integral [mV·µs]",
        )
        _draw_step_series(
            axes[1],
            series_max,
            bins=MAX_BIN_EDGES,
            density=False,
            ylabel="Counts",
            xlabel="Max voltage [mV]",
        )
        plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.08, hspace=0.35)
        plt.show()


def plot_same_overlay(per_run: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """全入力を同一 Figure に: signal のみ、run ごとに density 正規化で重ねる。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[plot_event_catalog] matplotlib required: pip install 'glassgem-analysis[viz]'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    from modules.project_matplotlib_rc import MPL_RC

    runs = sorted(per_run.keys())
    series_int = [
        (per_run[run][0], f"C{i % 10}", run) for i, run in enumerate(runs) if per_run[run][0].size > 0
    ]
    series_max = [
        (per_run[run][1], f"C{i % 10}", run) for i, run in enumerate(runs) if per_run[run][1].size > 0
    ]

    with plt.rc_context(MPL_RC):
        _fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.35})
        _draw_step_series(
            axes[0],
            series_int,
            bins=INTEGRAL_BIN_EDGES,
            density=True,
            ylabel="Density",
            xlabel="Integral [mV·µs]",
        )
        _draw_step_series(
            axes[1],
            series_max,
            bins=MAX_BIN_EDGES,
            density=True,
            ylabel="Density",
            xlabel="Max voltage [mV]",
        )
        plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.08, hspace=0.35)
        plt.show()


def merge_signal_by_run(
    catalogs: list[dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """複数カタログ分の per-run 配列を結合する。"""
    merged: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for per_run in catalogs:
        for run_num, (ints, maxs) in per_run.items():
            mi, mm = merged[run_num]
            mi.extend(ints.tolist())
            mm.extend(maxs.tolist())
    return {
        k: (np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float))
        for k, v in merged.items()
    }


def main() -> None:
    """CLI: カタログ CSV / 結果ディレクトリからヒストグラムを表示。"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event_catalog.csv or directory containing it",
    )
    parser.add_argument(
        "--same",
        action="store_true",
        help="Overlay all runs on one figure; signal only, density-normalized, legend=run_num",
    )
    args = parser.parse_args()

    catalog_paths = resolve_catalog_paths(args.paths)
    if not catalog_paths:
        print("No catalog CSV files found.", file=sys.stderr)
        sys.exit(1)

    if args.same:
        per_catalog = []
        for csv_path in catalog_paths:
            per_run = rows_signal_by_run(read_catalog_rows(csv_path))
            if per_run:
                per_catalog.append(per_run)
            else:
                print(f"[skip] no FFT-signal rows in {csv_path}", file=sys.stderr, flush=True)
        per_run = merge_signal_by_run(per_catalog)
        if not per_run:
            print("No FFT-signal rows to plot.", file=sys.stderr)
            sys.exit(1)
        print(
            f"Overlay: {len(per_run)} run(s) from {len(catalog_paths)} catalog file(s)",
            flush=True,
        )
        plot_same_overlay(per_run)
        return

    all_rows: list[dict[str, Any]] = []
    for csv_path in catalog_paths:
        all_rows.extend(read_catalog_rows(csv_path))

    int_sig, max_sig, int_noise, max_noise = rows_to_fft_samples(all_rows)
    n_used = int(int_sig.size + int_noise.size)
    if n_used == 0:
        print("No classified non-spark events to plot.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Combined: signal={int_sig.size}  noise={int_noise.size}  "
        f"(from {len(catalog_paths)} catalog file(s))",
        flush=True,
    )
    plot_combined(int_sig, max_sig, int_noise, max_noise)


if __name__ == "__main__":
    main()
