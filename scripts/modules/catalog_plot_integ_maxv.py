"""Plot integral / vmax histograms from decoded event catalog CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from modules.catalog_plot_common import (
    INTEGRAL_BIN_EDGES,
    LABEL_NOISE,
    LABEL_SIGNAL,
    MAX_BIN_EDGES,
    draw_step_series,
    load_catalog_rows,
    require_matplotlib,
    rows_to_fft_samples,
    signal_samples_by_catalog,
)
from modules.event_catalog_io import resolve_catalog_paths

SCRIPT_NAME = "plot_integ_maxv"


def plot_combined(
    int_sig: np.ndarray,
    max_sig: np.ndarray,
    int_noise: np.ndarray,
    max_noise: np.ndarray,
) -> None:
    """全 run 結合: signal + noise、カウント表示。"""
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.pyplot as plt

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
        draw_step_series(
            axes[0],
            series_int,
            bins=INTEGRAL_BIN_EDGES,
            density=False,
            ylabel="Counts",
            xlabel="Integral [mV·µs]",
        )
        draw_step_series(
            axes[1],
            series_max,
            bins=MAX_BIN_EDGES,
            density=False,
            ylabel="Counts",
            xlabel="Max voltage [mV]",
        )
        plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.08, hspace=0.35)
        plt.show()


def plot_same_overlay(per_catalog: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """全入力 PATH を同一 Figure に: signal のみ、PATH ごとに density 正規化で重ねる。"""
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    runs = sorted(per_catalog.keys())
    series_int = [
        (per_catalog[label][0], f"C{i % 10}", label)
        for i, label in enumerate(runs)
        if per_catalog[label][0].size > 0
    ]
    series_max = [
        (per_catalog[label][1], f"C{i % 10}", label)
        for i, label in enumerate(runs)
        if per_catalog[label][1].size > 0
    ]

    with plt.rc_context(MPL_RC):
        _fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.35})
        draw_step_series(
            axes[0],
            series_int,
            bins=INTEGRAL_BIN_EDGES,
            density=True,
            ylabel="Scaled counts",
            xlabel="Integral [mV·µs]",
        )
        draw_step_series(
            axes[1],
            series_max,
            bins=MAX_BIN_EDGES,
            density=True,
            ylabel="Scaled counts",
            xlabel="Max voltage [mV]",
        )
        plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.08, hspace=0.35)
        plt.show()


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
        help="Overlay catalogs on one figure; signal only, density-normalized, legend=results subdir name",
    )
    args = parser.parse_args()

    catalog_paths = resolve_catalog_paths(args.paths)
    if not catalog_paths:
        print("No catalog CSV files found.", file=sys.stderr)
        sys.exit(1)

    if args.same:
        per_catalog = signal_samples_by_catalog(catalog_paths)
        if not per_catalog:
            print("No FFT-signal rows to plot.", file=sys.stderr)
            sys.exit(1)
        print(
            f"Overlay: {len(per_catalog)} catalog(s) from {len(catalog_paths)} path(s)",
            flush=True,
        )
        plot_same_overlay(per_catalog)
        return

    all_rows, _paths = load_catalog_rows(args.paths)
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
