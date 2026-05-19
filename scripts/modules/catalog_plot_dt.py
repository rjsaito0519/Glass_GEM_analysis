"""Plot dt_ggem_nim_ns histogram with interactive integral/max follow-up."""

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
    dt_bin_edges_auto,
    dt_signal_samples_by_catalog,
    load_catalog_rows,
    require_matplotlib,
    rows_to_dt_signal_lists,
)
from modules.event_catalog_io import resolve_catalog_paths

SCRIPT_NAME = "plot_dt_ggem_nim"

NS_PER_US = 1000.0
DT_XLABEL = r"$\Delta t$ = $t_{\max,\mathrm{GGEM}} - t_{\mathrm{fall,NIM}}$ [µs]"


def dt_ns_to_us(values: list[float]) -> list[float]:
    """カタログ ``dt_ggem_nim_ns`` [ns] → プロット用 [µs]。"""
    return [float(v) / NS_PER_US for v in values]


def dt_catalog_ns_to_us(
    per_catalog: dict[str, tuple[list[float], list[float], list[float]]],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """PATH ごとの dt リストを ns → µs。"""
    return {
        label: (dt_ns_to_us(dt), int_s, max_s)
        for label, (dt, int_s, max_s) in per_catalog.items()
    }


def _spawn_integral_max_figure(
    lo: float,
    hi: float,
    seq: int,
    dt_signal: list[float],
    int_signal: list[float],
    max_signal: list[float],
) -> None:
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    int_s = [i for d, i in zip(dt_signal, int_signal) if lo <= d <= hi]
    max_s = [m for d, m in zip(dt_signal, max_signal) if lo <= d <= hi]

    print(
        f"[follow-up] window #{seq}: dt in [{lo:.6g}, {hi:.6g}] µs -> "
        f"signal {len(int_s)} events (int/max)",
        file=sys.stderr,
        flush=True,
    )

    with plt.rc_context(MPL_RC):
        fig2, axes2 = plt.subplots(2, 1, figsize=(9, 7.5))
        ax_i, ax_m = axes2[0], axes2[1]
        arr_is, arr_ms = np.asarray(int_s, dtype=float), np.asarray(max_s, dtype=float)
        if arr_is.size == 0:
            ax_i.text(0.5, 0.5, "no samples in range", ha="center", va="center", transform=ax_i.transAxes)
        else:
            ax_i.hist(arr_is, bins=INTEGRAL_BIN_EDGES, histtype="step", linewidth=1.8, color="C0")
        ax_i.set_xlabel("Integral [mV·µs]")
        ax_i.set_ylabel("Counts")

        if arr_ms.size == 0:
            ax_m.text(0.5, 0.5, "no samples in range", ha="center", va="center", transform=ax_m.transAxes)
        else:
            ax_m.hist(arr_ms, bins=MAX_BIN_EDGES, histtype="step", linewidth=1.8, color="C0")
        ax_m.set_xlabel("Max voltage [mV]")
        ax_m.set_ylabel("Counts")

        fig2.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.08, hspace=0.28)
        try:
            fig2.canvas.manager.set_window_title(f"int/max (range #{seq})")
        except Exception:
            pass
        plt.show(block=False)
        plt.pause(0.05)


def plot_dt_hist_then_integral_max_in_range(
    dt_signal: list[float],
    dt_noise: list[float],
    int_signal: list[float],
    max_signal: list[float],
    *,
    bin_edges: np.ndarray,
) -> None:
    """Δt ヒスト表示。右クリックで範囲、``w`` で int/max 追従 Figure。"""
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    clicks: list[float] = []
    follow_counter = [0]

    def _redraw_dt_hist(ax) -> None:
        ax.clear()
        s = np.asarray(dt_signal, dtype=float)
        n_arr = np.asarray(dt_noise, dtype=float)
        series = []
        if s.size > 0:
            series.append((s, "C0", LABEL_SIGNAL))
        if n_arr.size > 0:
            series.append((n_arr, "C3", LABEL_NOISE))
        draw_step_series(
            ax,
            series,
            bins=bin_edges,
            density=False,
            ylabel="Counts",
            xlabel=DT_XLABEL,
        )
        if len(clicks) >= 2:
            lo, hi = sorted((clicks[-2], clicks[-1]))
            ax.axvline(lo, color="red", ls=":", lw=2.0)
            ax.axvline(hi, color="red", ls=":", lw=2.0)

    def _spawn_from_current_clicks() -> None:
        if len(clicks) < 2:
            print(
                "[follow-up] need at least 2 right-clicks for a dt range (w ignored).",
                file=sys.stderr,
                flush=True,
            )
            return
        lo, hi = sorted((clicks[-2], clicks[-1]))
        follow_counter[0] += 1
        _spawn_integral_max_figure(
            lo, hi, follow_counter[0], dt_signal, int_signal, max_signal
        )

    def on_click(event) -> None:
        if getattr(event, "button", None) != 3:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or ax != ax_dt:
            return
        xd = getattr(event, "xdata", None)
        if xd is None:
            return
        clicks.append(float(xd))
        _redraw_dt_hist(ax_dt)
        fig_dt.canvas.draw_idle()

    def on_key(event) -> None:
        key = getattr(event, "key", None)
        if key is None or str(key).lower() != "w":
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or ax != ax_dt:
            return
        _spawn_from_current_clicks()

    with plt.rc_context(MPL_RC):
        fig_dt, ax_dt = plt.subplots(1, 1, figsize=(9, 5.5))
        fig_dt.subplots_adjust(left=0.1, right=0.97, top=0.96, bottom=0.14)
        try:
            fig_dt.canvas.manager.set_window_title("dt histogram — R-click range, w follow-up")
        except Exception:
            pass
        _redraw_dt_hist(ax_dt)
        fig_dt.canvas.mpl_connect("button_press_event", on_click)
        fig_dt.canvas.mpl_connect("key_press_event", on_key)
        plt.ion()
        plt.show(block=False)
        try:
            while plt.fignum_exists(fig_dt.number):
                plt.pause(0.15)
        finally:
            plt.ioff()


def plot_dt_same_overlay(
    per_catalog: dict[str, tuple[list[float], list[float], list[float]]],
    *,
    bin_edges: np.ndarray,
) -> None:
    """Δt ヒストを PATH ごと（signal のみ）に重ね表示。右クリック範囲 + ``w`` で int/max 追従。"""
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    dt_all: list[float] = []
    int_all: list[float] = []
    max_all: list[float] = []
    for dt_sig, int_sig, max_sig in per_catalog.values():
        dt_all.extend(dt_sig)
        int_all.extend(int_sig)
        max_all.extend(max_sig)

    clicks: list[float] = []
    follow_counter = [0]
    labels = sorted(per_catalog.keys())

    def _redraw_dt_hist(ax) -> None:
        ax.clear()
        series = [
            (np.asarray(per_catalog[label][0], dtype=float), f"C{i % 10}", label)
            for i, label in enumerate(labels)
            if per_catalog[label][0]
        ]
        draw_step_series(
            ax,
            series,
            bins=bin_edges,
            density=True,
            ylabel="Scaled counts",
            xlabel=DT_XLABEL,
        )
        if len(clicks) >= 2:
            lo, hi = sorted((clicks[-2], clicks[-1]))
            ax.axvline(lo, color="red", ls=":", lw=2.0)
            ax.axvline(hi, color="red", ls=":", lw=2.0)

    def _spawn_from_current_clicks() -> None:
        if len(clicks) < 2:
            print(
                "[follow-up] need at least 2 right-clicks for a dt range (w ignored).",
                file=sys.stderr,
                flush=True,
            )
            return
        lo, hi = sorted((clicks[-2], clicks[-1]))
        follow_counter[0] += 1
        _spawn_integral_max_figure(lo, hi, follow_counter[0], dt_all, int_all, max_all)

    def on_click(event) -> None:
        if getattr(event, "button", None) != 3:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or ax != ax_dt:
            return
        xd = getattr(event, "xdata", None)
        if xd is None:
            return
        clicks.append(float(xd))
        _redraw_dt_hist(ax_dt)
        fig_dt.canvas.draw_idle()

    def on_key(event) -> None:
        key = getattr(event, "key", None)
        if key is None or str(key).lower() != "w":
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or ax != ax_dt:
            return
        _spawn_from_current_clicks()

    with plt.rc_context(MPL_RC):
        fig_dt, ax_dt = plt.subplots(1, 1, figsize=(9, 5.5))
        fig_dt.subplots_adjust(left=0.1, right=0.97, top=0.96, bottom=0.14)
        try:
            fig_dt.canvas.manager.set_window_title("dt overlay — R-click range, w follow-up")
        except Exception:
            pass
        _redraw_dt_hist(ax_dt)
        fig_dt.canvas.mpl_connect("button_press_event", on_click)
        fig_dt.canvas.mpl_connect("key_press_event", on_key)
        plt.ion()
        plt.show(block=False)
        try:
            while plt.fignum_exists(fig_dt.number):
                plt.pause(0.15)
        finally:
            plt.ioff()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event_catalog.csv or directory containing it",
    )
    parser.add_argument("--dt-min", type=float, default=-8.0, help="dt histogram lower edge [µs]")
    parser.add_argument("--dt-max", type=float, default=8.0, help="dt histogram upper edge [µs]")
    parser.add_argument("--dt-bins", type=int, default=160, help="Number of dt histogram bins")
    parser.add_argument(
        "--dt-auto",
        action="store_true",
        help="Auto bin edges from data min/max with 5%% padding",
    )
    parser.add_argument(
        "--same",
        action="store_true",
        help="Overlay catalogs on one figure; signal dt only, scaled counts, legend=results subdir name",
    )
    args = parser.parse_args()

    if args.dt_bins < 1:
        parser.error("--dt-bins must be >= 1")

    catalog_paths = resolve_catalog_paths(args.paths)
    if not catalog_paths:
        print("No catalog CSV files found.", file=sys.stderr)
        sys.exit(1)

    if args.same:
        per_catalog = dt_catalog_ns_to_us(dt_signal_samples_by_catalog(catalog_paths))
        if not per_catalog:
            print("No dt signal rows to plot.", file=sys.stderr)
            sys.exit(1)
        all_dt = [d for dt, _, _ in per_catalog.values() for d in dt]
        if args.dt_auto:
            bin_edges = dt_bin_edges_auto(all_dt, [], n_bins=args.dt_bins)
        else:
            bin_edges = np.linspace(args.dt_min, args.dt_max, args.dt_bins + 1)
        print(
            f"dt overlay: {len(per_catalog)} catalog(s)  signal_dt={len(all_dt)}  "
            f"(from {len(catalog_paths)} path(s))",
            flush=True,
        )
        plot_dt_same_overlay(per_catalog, bin_edges=bin_edges)
        return

    all_rows, _paths = load_catalog_rows(args.paths)
    dt_signal, dt_noise, int_signal, max_signal = rows_to_dt_signal_lists(all_rows)
    dt_signal = dt_ns_to_us(dt_signal)
    dt_noise = dt_ns_to_us(dt_noise)
    n_used = len(dt_signal) + len(dt_noise)
    if n_used == 0:
        print("No dt samples to plot.", file=sys.stderr)
        sys.exit(1)

    if args.dt_auto:
        bin_edges = dt_bin_edges_auto(dt_signal, dt_noise, n_bins=args.dt_bins)
    else:
        bin_edges = np.linspace(args.dt_min, args.dt_max, args.dt_bins + 1)

    print(
        f"dt hist: signal={len(dt_signal)}  noise={len(dt_noise)}  "
        f"(from {len(catalog_paths)} catalog file(s))",
        flush=True,
    )
    plot_dt_hist_then_integral_max_in_range(
        dt_signal, dt_noise, int_signal, max_signal, bin_edges=bin_edges
    )


if __name__ == "__main__":
    main()
