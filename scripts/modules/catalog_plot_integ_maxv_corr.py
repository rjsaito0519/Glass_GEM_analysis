"""Plot integral vs vmax 2D histogram from decoded event catalog CSV files."""

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
    load_catalog_rows,
    require_matplotlib,
    rows_to_fft_samples,
)

SCRIPT_NAME = "plot_integ_maxv_corr"

X_BIN_EDGES = INTEGRAL_BIN_EDGES
Y_BIN_EDGES = MAX_BIN_EDGES
X_MIN = float(X_BIN_EDGES[0])
X_MAX = float(X_BIN_EDGES[-1])
Y_MIN = float(Y_BIN_EDGES[0])
Y_MAX = float(Y_BIN_EDGES[-1])


def _draw_hist2d_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
) -> None:
    """1 パネル分の 2D ヒスト（矩形メッシュ、1D hist と同じ bin 境界）。"""
    ax.set_title(title)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    if x.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Integral [mV·µs]")
        ax.set_ylabel("Max voltage [mV]")
        return

    from matplotlib.colors import LogNorm

    _, _, _, mesh = ax.hist2d(
        x,
        y,
        bins=[X_BIN_EDGES, Y_BIN_EDGES],
        norm=LogNorm(vmin=1),
        cmap="viridis",
    )
    ax.set_xlabel("Integral [mV·µs]")
    ax.set_ylabel("Max voltage [mV]")
    ax.figure.colorbar(mesh, ax=ax)


def plot_integ_maxv_corr(
    int_sig: np.ndarray,
    max_sig: np.ndarray,
    int_noise: np.ndarray,
    max_noise: np.ndarray,
) -> None:
    """signal / noise を別パネルで integral–vmax 2D ヒスト表示。"""
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
        _draw_hist2d_panel(axes[0], int_sig, max_sig, title=LABEL_SIGNAL)
        _draw_hist2d_panel(axes[1], int_noise, max_noise, title=LABEL_NOISE)
        if int_sig.size == 0 and int_noise.size == 0:
            for ax in axes:
                ax.set_xlabel("Integral [mV·µs]")
                ax.set_ylabel("Max voltage [mV]")
        elif int_sig.size == 0:
            axes[1].set_ylabel("")
        elif int_noise.size == 0:
            axes[0].set_ylabel("Max voltage [mV]")
        fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12, wspace=0.28)
        plt.show()


def main() -> None:
    """CLI: カタログ CSV / 結果ディレクトリから 2D ヒストを表示。"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event_catalog.csv or directory containing it",
    )
    args = parser.parse_args()

    all_rows, catalog_paths = load_catalog_rows(args.paths)
    if not catalog_paths:
        print("No catalog CSV files found.", file=sys.stderr)
        sys.exit(1)

    int_sig, max_sig, int_noise, max_noise = rows_to_fft_samples(all_rows)
    n_used = int(int_sig.size + int_noise.size)
    if n_used == 0:
        print("No classified non-spark events to plot.", file=sys.stderr)
        sys.exit(1)

    print(
        f"2D hist: signal={int_sig.size}  noise={int_noise.size}  "
        f"(from {len(catalog_paths)} catalog file(s))",
        flush=True,
    )
    plot_integ_maxv_corr(int_sig, max_sig, int_noise, max_noise)


if __name__ == "__main__":
    main()
