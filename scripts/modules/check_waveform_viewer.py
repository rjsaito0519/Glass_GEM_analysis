"""Interactive stacked waveform viewer for a run directory (all CH*)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from modules.common import aligned_csv_paths, discover_channel_roots, read_waveform_csv

PerChannelCsv = list[tuple[int, Path, list[Path]]]
TraceSeries = list[tuple[int, list[tuple[Path, np.ndarray, np.ndarray]]]]

_MATPLOTLIB_WARNED = False


def batch_starts(n_events: int, overlay_n: int) -> list[int]:
    """バッチ先頭 index の一覧（``overlay_n <= 1`` のときは全 index）。"""
    if n_events <= 0:
        return []
    if overlay_n <= 1:
        return list(range(n_events))
    return list(range(0, n_events, overlay_n))


def align_start_index(event: int, overlay_n: int, n_events: int) -> int:
    """先頭 index を有効なバッチ先頭に丸める。"""
    if n_events <= 0:
        return 0
    if overlay_n <= 1:
        return max(0, min(event, n_events - 1))
    k = (event // overlay_n) * overlay_n
    starts = batch_starts(n_events, overlay_n)
    if k > starts[-1]:
        return starts[-1]
    return k


def step_start_index(idx: int, delta: int, n_events: int, overlay_n: int) -> int:
    """``n`` / ``p`` 用に先頭 index を進める（wrap）。"""
    starts = batch_starts(n_events, overlay_n)
    if not starts:
        return 0
    if idx not in starts:
        idx = align_start_index(idx, overlay_n, n_events)
    i = starts.index(idx)
    return starts[(i + delta) % len(starts)]


def load_overlay_traces(
    per_ch_csv: PerChannelCsv,
    start_index: int,
    overlay_n: int,
) -> tuple[TraceSeries, list[str]]:
    """各 CH で ``overlay_n`` 本まで連続読み込み（時刻 µs, 電圧 mV）。"""
    out: TraceSeries = []
    warnings: list[str] = []
    for ch_id, _csv_dir, paths in per_ch_csv:
        series: list[tuple[Path, np.ndarray, np.ndarray]] = []
        for j in range(overlay_n):
            ev = start_index + j
            if ev >= len(paths):
                break
            fp = paths[ev]
            try:
                t, v = read_waveform_csv(fp)
                if t.size == 0:
                    raise ValueError("empty waveform")
                series.append((fp, t, v))
            except Exception as exc:
                warnings.append(f"CH{ch_id} {fp.name}: {exc}")
        if series:
            out.append((ch_id, series))
    return out, warnings


def _ensure_interactive_backend() -> bool:
    global _MATPLOTLIB_WARNED
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not _MATPLOTLIB_WARNED:
            print(
                "[check_waveform] matplotlib not installed; "
                "install viz extras: pip install 'glassgem-analysis[viz]'",
                file=sys.stderr,
                flush=True,
            )
            _MATPLOTLIB_WARNED = True
        return False
    if "agg" in plt.get_backend().lower():
        for name in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                plt.switch_backend(name)
                return True
            except Exception:
                continue
        if not _MATPLOTLIB_WARNED:
            print(
                "[check_waveform] interactive display needs TkAgg or Qt; set MPLBACKEND.",
                file=sys.stderr,
                flush=True,
            )
            _MATPLOTLIB_WARNED = True
        return False
    return True


def _stacked_axes(fig, n: int) -> np.ndarray:
    gs = fig.add_gridspec(n, 1, hspace=0.08)
    axes_list = []
    for i in range(n):
        ax = fig.add_subplot(gs[i, 0], sharex=axes_list[0] if i > 0 else None)
        axes_list.append(ax)
    return np.atleast_1d(np.array(axes_list, dtype=object))


def _paint_axes(axes_arr: np.ndarray, traces: TraceSeries) -> None:
    for ax, (ch_id, series) in zip(axes_arr, traces):
        n_draw = len(series)
        for j, (fp, t, v) in enumerate(series):
            color = f"C{j % 10}"
            alpha = 1.0 if n_draw == 1 else 0.35 + 0.65 * (1.0 - j / max(n_draw - 1, 1))
            label = fp.stem if n_draw <= 15 else None
            ax.plot(t, v, lw=0.9, color=color, alpha=alpha, label=label)
        ax.text(0.02, 0.98, f"CH{ch_id}", transform=ax.transAxes, va="top", ha="left")
        if n_draw > 1 and n_draw <= 15:
            ax.legend(loc="upper right", fontsize=9)
        elif n_draw > 1:
            ax.text(
                0.02,
                0.86,
                f"{n_draw} traces",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="0.35",
            )
    axes_arr[-1].set_xlabel("Time [µs]")


def _page_title(idx: int, overlay_n: int, n_events: int) -> str:
    if overlay_n <= 1:
        return f"{idx}"
    end = min(idx + overlay_n - 1, n_events - 1)
    return f"{idx}–{end}"


def run_interactive(
    per_ch_csv: PerChannelCsv,
    n_events: int,
    *,
    overlay_n: int,
) -> None:
    """全 CH 積み上げ波形を ``n`` / ``p`` で切り替える。"""
    if n_events == 0:
        print("No CSV events found (empty csv dirs?).", file=sys.stderr)
        sys.exit(1)
    if not _ensure_interactive_backend():
        sys.exit(1)

    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    idx = 0
    fig = None

    def redraw() -> None:
        nonlocal fig
        traces, warns = load_overlay_traces(per_ch_csv, idx, overlay_n)
        for w in warns:
            print(w, file=sys.stderr, flush=True)
        if not traces:
            print("No traces loaded for this page.", file=sys.stderr)
            return

        n_ch = len(traces)
        fig_h = max(3.5, 2.4 * n_ch)
        with plt.rc_context(MPL_RC):
            if fig is None:
                fig = plt.figure(figsize=(10.0, fig_h))
                fig.canvas.mpl_connect("key_press_event", on_key)
            else:
                fig.clear()
                fig.set_size_inches(10.0, fig_h)

            axes_arr = _stacked_axes(fig, n_ch)
            _paint_axes(axes_arr, traces)
            if overlay_n > 1:
                axes_arr[0].tick_params(labelbottom=False)
            fig.suptitle(_page_title(idx, overlay_n, n_events), fontsize=10)
            fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.12, hspace=0.08)
            fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal idx
        if event.key in ("n", "N"):
            idx = step_start_index(idx, 1, n_events, overlay_n)
            redraw()
        elif event.key in ("p", "P"):
            idx = step_start_index(idx, -1, n_events, overlay_n)
            redraw()
        elif event.key in ("q", "Q", "escape"):
            plt.close("all")

    with plt.rc_context(MPL_RC):
        redraw()
        plt.show()


_CSV_SUBDIR = "csv"


def resolve_run_paths(run_dir: Path) -> tuple[PerChannelCsv, int]:
    """run を開き ``(per_ch_csv, n_events)`` を返す。失敗時は終了。"""
    channels = discover_channel_roots(run_dir, csv_subdir=_CSV_SUBDIR)
    if not channels:
        print(f"No CH*/{_CSV_SUBDIR} found under {run_dir}", file=sys.stderr)
        sys.exit(1)
    per_ch_csv, n_events = aligned_csv_paths(channels, csv_subdir=_CSV_SUBDIR)
    return per_ch_csv, n_events


def main() -> None:
    """CLI: ``RUN_DIR`` [``--n N``]。"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=1, metavar="N")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n requires N >= 1")

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    per_ch_csv, n_events = resolve_run_paths(run_dir)
    run_interactive(per_ch_csv, n_events, overlay_n=args.n)


if __name__ == "__main__":
    main()
