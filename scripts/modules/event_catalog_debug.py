"""Interactive ``--debug`` waveform browser for event catalog builds."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modules.common import (
    _first_falling_crossing_us,
    _first_rising_crossing_us,
    align_pair,
    dt_us,
    dominant_peak_freq_hz_excluding_dc,
    inferred_fall_time_us,
    max_v_time_us,
    one_sided_psd_m2_per_hz,
    read_waveform_csv,
    window_arrays,
)

_DEBUG_MATPLOTLIB_WARNED = False


@dataclass
class CatalogDebugEvent:
    """1 イベント分のデバッグ表示用データ。"""

    index: int
    ggem_csv: str
    timestamp: str
    t_ggem_us: np.ndarray
    v_ggem_mv: np.ndarray
    t_nim_us: np.ndarray | None
    v_nim_mv: np.ndarray | None
    tmin_us: float
    tmax_us: float
    center_run_mv: float
    edge_mv: float
    nim_width_ns: float
    spark: bool
    fft_is_noise: str
    dt_ggem_nim_ns: str
    subtract_mean_fft: bool
    rel_tol_dt: float
    noise_dominant_peak_min_hz: float
    read_error: str | None = None


def make_debug_snapshot(
    *,
    k: int,
    ggem_name: str,
    ts_str: str,
    t_ggem_us: np.ndarray,
    v_ggem_mv: np.ndarray,
    nim_paths: list[Path],
    tmin_us: float,
    tmax_us: float,
    center_run_mv: float,
    edge_mv: float,
    nim_width_ns: float,
    spark: bool,
    fft_is_noise: str,
    dt_ggem_nim_ns: str,
    subtract_mean_fft: bool,
    rel_tol_dt: float,
    noise_dominant_peak_min_hz: float,
    read_error: str | None = None,
) -> CatalogDebugEvent:
    """GGEM / NIM 波形を揃えたデバッグ用スナップショットを作る。"""
    t_nim: np.ndarray | None = None
    v_nim: np.ndarray | None = None
    if read_error is None and k < len(nim_paths):
        try:
            t_n, v_n = read_waveform_csv(nim_paths[k])
            aligned = align_pair(t_ggem_us, v_ggem_mv, t_n, v_n)
            if aligned is not None:
                _tg, _vg, t_nim, v_nim = aligned
        except Exception:
            t_nim = None
            v_nim = None
    return CatalogDebugEvent(
        index=k,
        ggem_csv=ggem_name,
        timestamp=ts_str,
        t_ggem_us=np.asarray(t_ggem_us, dtype=float),
        v_ggem_mv=np.asarray(v_ggem_mv, dtype=float),
        t_nim_us=None if t_nim is None else np.asarray(t_nim, dtype=float),
        v_nim_mv=None if v_nim is None else np.asarray(v_nim, dtype=float),
        tmin_us=tmin_us,
        tmax_us=tmax_us,
        center_run_mv=center_run_mv,
        edge_mv=edge_mv,
        nim_width_ns=nim_width_ns,
        spark=spark,
        fft_is_noise=fft_is_noise,
        dt_ggem_nim_ns=dt_ggem_nim_ns,
        subtract_mean_fft=subtract_mean_fft,
        rel_tol_dt=rel_tol_dt,
        noise_dominant_peak_min_hz=noise_dominant_peak_min_hz,
        read_error=read_error,
    )


def _ensure_interactive_backend() -> bool:
    global _DEBUG_MATPLOTLIB_WARNED
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not _DEBUG_MATPLOTLIB_WARNED:
            print(
                "[debug] matplotlib not installed; install viz extras: pip install 'glassgem-analysis[viz]'",
                file=sys.stderr,
                flush=True,
            )
            _DEBUG_MATPLOTLIB_WARNED = True
        return False
    if "agg" in plt.get_backend().lower():
        for name in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                plt.switch_backend(name)
                return True
            except Exception:
                continue
        if not _DEBUG_MATPLOTLIB_WARNED:
            print(
                "[debug] interactive display needs TkAgg or Qt (could not switch from Agg); "
                "set MPLBACKEND before run.",
                file=sys.stderr,
                flush=True,
            )
            _DEBUG_MATPLOTLIB_WARNED = True
        return False
    return True


def _span_analysis_window(ax, tmin_us: float, tmax_us: float) -> None:
    ax.axvspan(tmin_us, tmax_us, alpha=0.18, color="C2", zorder=0, label="analysis window")


def _draw_waveform_axis(
    ax,
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    tmin_us: float,
    tmax_us: float,
    center_run_mv: float,
    ylabel: str,
    markers: list[tuple[float, str, str]],
    fill_window_trace: bool = False,
) -> None:
    _span_analysis_window(ax, tmin_us, tmax_us)
    if fill_window_trace:
        tw, vw = window_arrays(t_us, v_mv, tmin_us=tmin_us, tmax_us=tmax_us)
        if tw.size >= 2:
            ax.fill_between(
                tw,
                center_run_mv,
                vw,
                alpha=0.22,
                color="C2",
                zorder=1,
                label="window (filled)",
            )
    ax.plot(t_us, v_mv, lw=0.8, color="C0", zorder=2)
    ax.axhline(center_run_mv, color="C3", ls="--", lw=0.9, alpha=0.85, label="run baseline")
    for t_mark, label, color in markers:
        if np.isfinite(t_mark):
            ax.axvline(t_mark, color=color, ls="-", lw=1.1, alpha=0.9, label=label)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)


def _ggem_markers(ev: CatalogDebugEvent) -> list[tuple[float, str, str]]:
    tw, vw = window_arrays(ev.t_ggem_us, ev.v_ggem_mv, tmin_us=ev.tmin_us, tmax_us=ev.tmax_us)
    if tw.size < 2:
        return []
    vw_c = np.asarray(vw, dtype=float) - ev.center_run_mv
    t_peak = max_v_time_us(tw, vw_c)
    if t_peak is None:
        return []
    return [(t_peak, "GGEM peak (window)", "C1")]


def _nim_markers(ev: CatalogDebugEvent) -> list[tuple[float, str, str]]:
    if ev.t_nim_us is None or ev.v_nim_mv is None:
        return []
    t = ev.t_nim_us
    v = ev.v_nim_mv
    out: list[tuple[float, str, str]] = []
    t_fall = _first_falling_crossing_us(t, v, ev.edge_mv)
    if t_fall is not None:
        out.append((t_fall, "NIM fall", "C4"))
    t_rise = _first_rising_crossing_us(t, v, ev.edge_mv)
    if t_rise is not None:
        out.append((t_rise, "NIM rise", "C5"))
    t_inf = inferred_fall_time_us(t, v, v_th=ev.edge_mv, nim_width_ns=ev.nim_width_ns)
    if t_inf is not None and (t_fall is None or abs(t_inf - t_fall) > 1e-9):
        out.append((t_inf, "NIM fall (inferred)", "C4"))
    return out


def _draw_psd_axis(ax, ev: CatalogDebugEvent) -> None:
    tw, vw = window_arrays(ev.t_ggem_us, ev.v_ggem_mv, tmin_us=ev.tmin_us, tmax_us=ev.tmax_us)
    if tw.size < 4:
        ax.text(0.5, 0.5, "PSD: window too short", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Frequency [MHz]")
        return
    vw_c = np.asarray(vw, dtype=float) - ev.center_run_mv
    sample_dt_us = dt_us(tw, rel_tol=ev.rel_tol_dt)
    if sample_dt_us is None:
        ax.text(0.5, 0.5, "PSD: irregular sampling", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Frequency [MHz]")
        return
    fs_hz = 1.0 / (sample_dt_us * 1e-6)
    psd_result = one_sided_psd_m2_per_hz(
        vw_c, fs_hz=fs_hz, subtract_mean=ev.subtract_mean_fft
    )
    if psd_result is None:
        ax.text(0.5, 0.5, "PSD: unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Frequency [MHz]")
        return
    freq_hz, psd = psd_result
    freq_mhz = freq_hz * 1e-6
    ax.semilogy(freq_mhz, np.maximum(psd, 1e-30), lw=0.9, color="C0")
    f_dom = dominant_peak_freq_hz_excluding_dc(freq_hz, psd)
    if f_dom is not None:
        ax.axvline(f_dom * 1e-6, color="C1", ls="--", lw=1.0, label=f"dom peak {f_dom * 1e-6:.3g} MHz")
    thr_mhz = ev.noise_dominant_peak_min_hz * 1e-6
    ax.axvline(
        thr_mhz,
        color="C3",
        ls=":",
        lw=1.0,
        label=f"noise threshold {thr_mhz:.3g} MHz",
    )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(r"PSD [mV$^2$/Hz]")
    ax.legend(loc="upper right", fontsize=8)


def _sync_waveform_xlim(ax: Any, ev: CatalogDebugEvent) -> None:
    """GGEM / NIM で共通の時間軸範囲 [µs] を設定する（``sharex`` 用）。"""
    lo: float | None = None
    hi: float | None = None
    for t in (ev.t_ggem_us, ev.t_nim_us):
        if t is None or t.size == 0:
            continue
        t0, t1 = float(np.min(t)), float(np.max(t))
        lo = t0 if lo is None else min(lo, t0)
        hi = t1 if hi is None else max(hi, t1)
    if lo is None or hi is None:
        return
    span = hi - lo
    pad = 0.02 * span if span > 0 else 1.0
    ax.set_xlim(lo - pad, hi + pad)


def run_catalog_debug_browser(events: list[CatalogDebugEvent], *, run_num: str) -> None:
    """イベント列を n / p キーで切り替えながら表示する。"""
    if not events:
        print("[debug] no events to display", file=sys.stderr, flush=True)
        return
    if not _ensure_interactive_backend():
        return

    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    class _Browser:
        def __init__(self) -> None:
            self.idx = 0
            with plt.rc_context(MPL_RC):
                self.fig = plt.figure(figsize=(13.0, 7.0))
                gs = self.fig.add_gridspec(
                    2,
                    2,
                    width_ratios=(3.2, 0.85),
                    height_ratios=(1.0, 1.0),
                    hspace=0.10,
                    wspace=0.24,
                )
                self.ax_ggem = self.fig.add_subplot(gs[0, 0])
                self.ax_nim = self.fig.add_subplot(gs[1, 0], sharex=self.ax_ggem)
                self.ax_psd = self.fig.add_subplot(gs[:, 1])
                self.fig.canvas.mpl_connect("key_press_event", self._on_key)
                self._draw()
                print(
                    "[debug] interactive viewer: n=next, p=prev, q=quit",
                    file=sys.stderr,
                    flush=True,
                )
                plt.show()

        def _on_key(self, event) -> None:
            if event.key in ("n", "N"):
                if self.idx < len(events) - 1:
                    self.idx += 1
                    self._draw()
            elif event.key in ("p", "P"):
                if self.idx > 0:
                    self.idx -= 1
                    self._draw()
            elif event.key in ("q", "Q", "escape"):
                plt.close(self.fig)

        def _draw(self) -> None:
            ev = events[self.idx]
            self.ax_ggem.clear()
            self.ax_nim.clear()
            self.ax_psd.clear()
            if ev.read_error:
                msg = f"read error: {ev.read_error}"
                self.ax_ggem.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax_ggem.transAxes)
                self.ax_nim.set_visible(False)
                self.ax_psd.set_visible(False)
            else:
                self.ax_nim.set_visible(True)
                self.ax_psd.set_visible(True)
                _draw_waveform_axis(
                    self.ax_ggem,
                    ev.t_ggem_us,
                    ev.v_ggem_mv,
                    tmin_us=ev.tmin_us,
                    tmax_us=ev.tmax_us,
                    center_run_mv=ev.center_run_mv,
                    ylabel="GGEM [mV]",
                    markers=_ggem_markers(ev),
                    fill_window_trace=True,
                )
                if ev.t_nim_us is not None and ev.v_nim_mv is not None:
                    _draw_waveform_axis(
                        self.ax_nim,
                        ev.t_nim_us,
                        ev.v_nim_mv,
                        tmin_us=ev.tmin_us,
                        tmax_us=ev.tmax_us,
                        center_run_mv=ev.center_run_mv,
                        ylabel="NIM [mV]",
                        markers=_nim_markers(ev),
                    )
                else:
                    self.ax_nim.text(
                        0.5,
                        0.5,
                        "NIM waveform unavailable",
                        ha="center",
                        va="center",
                        transform=self.ax_nim.transAxes,
                    )
                _draw_psd_axis(self.ax_psd, ev)
                _sync_waveform_xlim(self.ax_nim, ev)
                self.ax_ggem.tick_params(labelbottom=False)
                self.ax_nim.set_xlabel("Time [µs]")
            spark_s = "spark" if ev.spark else "non-spark"
            fft_s = ev.fft_is_noise if ev.fft_is_noise else "—"
            dt_s = ev.dt_ggem_nim_ns if ev.dt_ggem_nim_ns else "—"
            ts_part = f"  {ev.timestamp}" if ev.timestamp else ""
            self.fig.suptitle(
                f"{run_num}  k={ev.index}  {ev.ggem_csv}{ts_part}  "
                f"[{self.idx + 1}/{len(events)}]  {spark_s}  fft={fft_s}  dt_ns={dt_s}",
                fontsize=10,
            )
            self.fig.canvas.draw_idle()

    with plt.rc_context(MPL_RC):
        _Browser()
