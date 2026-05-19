"""Trend plot: signal count, S/N, and spark rate per minute from event catalog."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from modules.catalog_plot_common import (
    TrendSegmentSpan,
    catalog_segment_label,
    compress_path_gaps_xnum,
    gap_compress_margin_days,
    prepare_trend_binned,
    require_matplotlib,
)
from modules.event_catalog_io import parse_float_cell, read_catalog_rows, resolve_catalog_paths

SCRIPT_NAME = "plot_trend_sn_spark"

# スパーク率の表示単位（分あたりではなく「10 分あたり」に正規化）
SPARK_RATE_REF_MIN = 10.0


def _draw_segment_boundaries(
    axes,
    xnum: np.ndarray,
    spans: list[TrendSegmentSpan],
) -> None:
    """複数カタログの区切りを破線とラベルで示す。"""
    if len(spans) <= 1:
        return

    for i in range(len(spans) - 1):
        left = spans[i]
        right = spans[i + 1]
        if left.end <= 0 or right.start >= xnum.size:
            continue
        x_lo = float(xnum[left.end - 1])
        x_hi = float(xnum[right.start])
        x_sep = 0.5 * (x_lo + x_hi)
        for ax in axes:
            ax.axvline(x_sep, color="0.45", ls="--", lw=1.2, zorder=2)

    ax0 = axes[0]
    y_top = 1.02
    for span in spans:
        if span.end <= span.start:
            continue
        x_mid = 0.5 * (float(xnum[span.start]) + float(xnum[span.end - 1]))
        ax0.text(
            x_mid,
            y_top,
            span.label,
            transform=ax0.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
            color="0.35",
            clip_on=False,
        )


def _format_tick_time(dt: datetime, *, show_date: bool) -> str:
    if show_date:
        return dt.strftime("%b-%d %H:%M")
    return dt.strftime("%H:%M")


def _pick_segment_tick_indices(
    span: TrendSegmentSpan,
    *,
    max_ticks: int,
    skip_first: bool,
    skip_last: bool,
) -> list[int]:
    n = span.end - span.start
    if n <= 0:
        return []
    lo = span.start + (1 if skip_first and n > 1 else 0)
    hi = span.end - 1 - (1 if skip_last and n > 1 else 0)
    if hi < lo:
        return []
    if lo == hi:
        return [lo]
    k = min(max_ticks, hi - lo + 1)
    return np.linspace(lo, hi, k, dtype=int).tolist()


def _build_multisegment_ticks(
    xnum: np.ndarray,
    centers_dt: list[datetime],
    spans: list[TrendSegmentSpan],
    *,
    max_ticks_per_segment: int,
    min_tick_sep_days: float,
) -> tuple[list[float], list[str]]:
    """目盛り候補を選び、x 方向の最小間隔で間引く。"""
    candidates: list[tuple[float, str, int]] = []
    last_date: datetime.date | None = None
    multi = len(spans) > 1

    for si, span in enumerate(spans):
        skip_last = multi and si < len(spans) - 1
        skip_first = multi and si > 0
        for i in _pick_segment_tick_indices(
            span,
            max_ticks=max_ticks_per_segment,
            skip_first=skip_first,
            skip_last=skip_last,
        ):
            dt = centers_dt[i]
            show_date = last_date is None or dt.date() != last_date
            if show_date:
                last_date = dt.date()
            candidates.append((float(xnum[i]), _format_tick_time(dt, show_date=show_date), i))

    if not candidates:
        return [], []

    candidates.sort(key=lambda t: t[0])
    tick_x: list[float] = []
    tick_lbl: list[str] = []
    for xv, lbl, _idx in candidates:
        if tick_x and (xv - tick_x[-1]) < float(min_tick_sep_days):
            continue
        tick_x.append(xv)
        tick_lbl.append(lbl)

    return tick_x, tick_lbl


def _set_multisegment_time_ticks(
    ax,
    xnum: np.ndarray,
    centers_dt: list[datetime],
    spans: list[TrendSegmentSpan],
    *,
    max_ticks_per_segment: int = 2,
    min_tick_sep_days: float,
) -> None:
    """圧縮後 x でも実時刻ラベルをセグメントごとに付ける（境界付近は間引く）。"""
    tick_x, tick_lbl = _build_multisegment_ticks(
        xnum,
        centers_dt,
        spans,
        max_ticks_per_segment=max_ticks_per_segment,
        min_tick_sep_days=min_tick_sep_days,
    )
    if tick_x:
        ax.set_xticks(tick_x)
        ax.set_xticklabels(tick_lbl, ha="center")


def _run_baseline_mv(rows: list[dict[str, Any]]) -> float:
    """カタログ内の run ベースライン [mV]（最初の有効値）。"""
    for row in rows:
        b = parse_float_cell(row.get("baseline_mv", ""))
        if np.isfinite(b):
            return float(b)
    return float("nan")


def plot_trend(
    centers_s: np.ndarray,
    signal_counts: np.ndarray,
    noise_counts: np.ndarray,
    spark_counts: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_sem: np.ndarray,
    *,
    bin_minutes: float,
    fit_exp: bool,
    segment_spans: list[TrendSegmentSpan] | None = None,
    segment_run_baselines: list[float] | None = None,
    compress_gaps: bool = False,
) -> None:
    require_matplotlib(SCRIPT_NAME)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from modules.project_matplotlib_rc import MPL_RC

    centers_dt = np.array([datetime.fromtimestamp(float(s)) for s in centers_s], dtype=object)
    half_bin_days = (0.5 * bin_minutes * 60.0) / 86400.0
    xnum = np.asarray(mdates.date2num(list(centers_dt)), dtype=float)
    xerr = np.full(xnum.shape, half_bin_days, dtype=float)

    sn = np.full(signal_counts.shape, np.nan, dtype=float)
    valid = noise_counts > 0
    sn[valid] = signal_counts[valid] / noise_counts[valid]
    sn[~valid & (signal_counts > 0)] = np.inf
    sig_yerr = np.sqrt(signal_counts.astype(float))

    sn_yerr = np.full(sn.shape, np.nan, dtype=float)
    for i in range(sn.size):
        s = float(signal_counts[i])
        n = float(noise_counts[i])
        if n <= 0.0 or s <= 0.0:
            continue
        r = s / n
        sn_yerr[i] = r * math.sqrt((1.0 / s) + (1.0 / n))

    scale_per_10min = SPARK_RATE_REF_MIN / float(bin_minutes)
    spark_rate = spark_counts.astype(float) * scale_per_10min
    spark_yerr = np.sqrt(spark_counts.astype(float)) * scale_per_10min

    spans = segment_spans or []
    if compress_gaps and len(spans) > 1:
        xnum = compress_path_gaps_xnum(xnum, spans)

    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, gridspec_kw={"hspace": 0.10})
        ax0, ax1, ax2, ax3 = axes

        if len(spans) > 1:
            _draw_segment_boundaries(axes, xnum, spans)

        ax0.errorbar(
            xnum,
            signal_counts,
            xerr=xerr,
            yerr=sig_yerr,
            fmt="o",
            markersize=4.5,
            color="C0",
            ecolor="C0",
            elinewidth=1.0,
            capsize=2.5,
            label="signal count",
            zorder=3,
        )
        ax0.set_ylabel(f"Signal / {bin_minutes:g} min")
        ax0.legend(loc="upper right", fontsize=10)

        finite = np.isfinite(sn) & np.isfinite(sn_yerr)
        if np.any(finite):
            ax1.errorbar(
                xnum[finite],
                sn[finite],
                xerr=xerr[finite],
                yerr=sn_yerr[finite],
                fmt="o",
                markersize=4.5,
                color="C2",
                ecolor="C2",
                elinewidth=1.0,
                capsize=2.5,
                label="S/N",
                zorder=3,
            )
        inf_mask = np.isinf(sn)
        if np.any(inf_mask):
            y_inf = np.full(
                np.count_nonzero(inf_mask),
                np.nanmax(sn[finite]) * 1.1 if np.any(finite) else 1.0,
            )
            ax1.scatter(
                xnum[inf_mask], y_inf, marker="^", color="C3", s=36, label="S/N = inf (noise=0)", zorder=3
            )
        ax1.set_ylabel("S/N")
        ax1.legend(loc="upper right", fontsize=10)

        spark_mask = spark_counts > 0
        if np.any(spark_mask):
            ax2.errorbar(
                xnum[spark_mask],
                spark_rate[spark_mask],
                xerr=xerr[spark_mask],
                yerr=spark_yerr[spark_mask],
                fmt="o",
                markersize=4.5,
                color="C4",
                ecolor="C4",
                elinewidth=1.0,
                capsize=2.5,
                label="spark rate",
                zorder=3,
            )
        elif spark_counts.size > 0:
            ax2.plot([], [], linestyle="none", label="no sparks in bins")
        ax2.set_ylabel(f"Sparks / {SPARK_RATE_REF_MIN:g} min")
        ax2.legend(loc="upper right", fontsize=10)

        bl_mask = np.isfinite(baseline_mean)
        if np.any(bl_mask):
            yerr_bl = baseline_sem[bl_mask]
            yerr_bl = np.where(np.isfinite(yerr_bl), yerr_bl, 0.0)
            ax3.errorbar(
                xnum[bl_mask],
                baseline_mean[bl_mask],
                xerr=xerr[bl_mask],
                yerr=yerr_bl,
                fmt="o",
                markersize=4.5,
                color="C5",
                ecolor="C5",
                elinewidth=1.0,
                capsize=2.5,
                label="⟨baseline_indiv⟩",
                zorder=3,
            )
        else:
            ax3.text(0.5, 0.5, "no baseline_indiv_mv", ha="center", va="center", transform=ax3.transAxes)

        run_bl_label_added = False
        if segment_spans and segment_run_baselines:
            for span, run_bl in zip(segment_spans, segment_run_baselines):
                if not np.isfinite(run_bl) or span.end <= span.start:
                    continue
                x0 = float(xnum[span.start])
                x1 = float(xnum[span.end - 1])
                lab = "run baseline" if not run_bl_label_added else None
                ax3.hlines(run_bl, x0, x1, colors="0.55", linestyles="--", linewidth=1.0, label=lab)
                run_bl_label_added = True

        ax3.set_ylabel("Baseline [mV]")
        ax3.set_xlabel("Time")
        ax3.legend(loc="upper right", fontsize=10)

        if fit_exp:
            try:
                from lmfit import Model
            except Exception:
                print("[warn] lmfit not available; skip exponential fit.", file=sys.stderr)
            else:

                def exp_decay(x: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
                    return a * np.exp(-x / tau) + c

                def add_fit(
                    ax,
                    y: np.ndarray,
                    yerr: np.ndarray,
                    mask: np.ndarray,
                    *,
                    color: str,
                    label: str,
                ) -> tuple[str, str] | None:
                    if np.count_nonzero(mask) < 4:
                        return None
                    x_min = float(np.min(centers_s[mask]))
                    x = (centers_s[mask] - x_min) / 60.0
                    yy = y[mask].astype(float)
                    ee = np.where((yerr[mask] > 0) & np.isfinite(yerr[mask]), yerr[mask], 1.0)
                    weights = 1.0 / ee
                    model = Model(exp_decay)
                    params = model.make_params(
                        a=max(float(np.max(yy) - np.min(yy)), 1e-6),
                        tau=max(float((x[-1] - x[0]) / 2.0), 1e-6),
                        c=float(np.min(yy)),
                    )
                    params["tau"].set(min=1e-9)
                    try:
                        out = model.fit(yy, params, x=x, weights=weights)
                    except Exception:
                        return None
                    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 300)
                    y_dense = out.eval(x=x_dense)
                    t_dense = [datetime.fromtimestamp(float(x_min + xv * 60.0)) for xv in x_dense]
                    x_dense_num = np.asarray(mdates.date2num(t_dense), dtype=float)
                    ax.plot(x_dense_num, y_dense, color=color, linewidth=1.4, linestyle="--", label=label)
                    tau = float(out.params["tau"].value)
                    c = float(out.params["c"].value)
                    tau_err = out.params["tau"].stderr
                    c_err = out.params["c"].stderr

                    def _fmt_val_err(val: float, err: float | None, unit: str = "") -> str:
                        if err is None or (not np.isfinite(err)):
                            return f"{val:.4g}{unit}"
                        return f"{val:.4g} ± {float(err):.2g}{unit}"

                    return (
                        f"tau={_fmt_val_err(tau, tau_err, ' min')}",
                        f"const={_fmt_val_err(c, c_err)}",
                    )

                sig_mask = signal_counts > 0
                note_sig = add_fit(
                    ax0,
                    signal_counts.astype(float),
                    sig_yerr,
                    sig_mask,
                    color="C1",
                    label="exp fit",
                )
                if note_sig is not None:
                    tau_lbl, const_lbl = note_sig
                    ax0.plot([], [], linestyle="none", label=tau_lbl)
                    ax0.plot([], [], linestyle="none", label=const_lbl)
                sn_mask = np.isfinite(sn) & (sn > 0) & np.isfinite(sn_yerr)
                note_sn = add_fit(ax1, sn, sn_yerr, sn_mask, color="C4", label="exp fit")
                if note_sn is not None:
                    tau_lbl, const_lbl = note_sn
                    ax1.plot([], [], linestyle="none", label=tau_lbl)
                    ax1.plot([], [], linestyle="none", label=const_lbl)
                ax0.legend(loc="upper right", fontsize=10)
                ax1.legend(loc="upper right", fontsize=10)

        if compress_gaps and len(spans) > 1:
            sep = gap_compress_margin_days()
            _set_multisegment_time_ticks(
                ax3,
                xnum,
                [dt for dt in centers_dt],
                spans,
                min_tick_sep_days=sep * 0.9,
            )
        else:
            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax3.xaxis.set_major_locator(locator)
            ax3.xaxis.set_major_formatter(formatter)
        top = 0.92 if len(spans) > 1 else 0.97
        fig.subplots_adjust(left=0.12, right=0.98, top=top, bottom=0.09, wspace=0.05)
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event_catalog.csv or directory containing it",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=10.0,
        metavar="MIN",
        help="Time-bin width in minutes (default: 10)",
    )
    parser.add_argument(
        "--no-fit-exp",
        action="store_true",
        help="Disable exp fit even with a single catalog (default: fit only for one PATH)",
    )
    parser.add_argument(
        "--no-compress-gaps",
        action="store_true",
        help="Segmented mode only: keep calendar x-axis (no gap compress; default compress when PATH gap >= 1 h)",
    )
    args = parser.parse_args()

    if args.span <= 0.0:
        parser.error("--span must be > 0")

    catalog_paths = resolve_catalog_paths(args.paths)
    if not catalog_paths:
        print("No catalog CSV files found.", file=sys.stderr)
        sys.exit(1)

    rows_per_path = [read_catalog_rows(p) for p in catalog_paths]
    (
        centers_s,
        signal_counts,
        noise_counts,
        spark_counts,
        baseline_mean,
        baseline_sem,
        spans,
        mode,
        n_skip_ts,
    ) = prepare_trend_binned(catalog_paths, rows_per_path, bin_minutes=args.span)

    if centers_s.size == 0:
        print("No events with timestamps to plot.", file=sys.stderr)
        sys.exit(1)

    n_sig = int(np.sum(signal_counts))
    n_noi = int(np.sum(noise_counts))
    n_spk = int(np.sum(spark_counts))
    if n_sig + n_noi + n_spk == 0:
        print("No binned events to plot.", file=sys.stderr)
        sys.exit(1)

    if n_skip_ts > 0:
        print(f"[warn] skipped {n_skip_ts} row(s) without parseable timestamp", file=sys.stderr, flush=True)

    multi = len(catalog_paths) > 1
    fit_exp = len(catalog_paths) == 1 and not args.no_fit_exp
    compress_gaps = mode == "segmented" and not args.no_compress_gaps
    if multi:
        print("exp fit: off (multiple catalogs)", flush=True)
    elif args.no_fit_exp:
        print("exp fit: off (--no-fit-exp)", flush=True)

    if mode == "continuous":
        mode_msg = "continuous (merged binning)"
    elif compress_gaps:
        mode_msg = "segmented (gap compress)"
    elif args.no_compress_gaps:
        mode_msg = "segmented (calendar x, --no-compress-gaps)"
    else:
        mode_msg = "segmented"

    n_bl_bins = int(np.sum(np.isfinite(baseline_mean)))
    seg_labels = ", ".join(s.label for s in spans) if spans else "—"
    print(
        f"Trend: mode={mode_msg}  span={args.span:g} min  bins={signal_counts.size}  "
        f"signal={n_sig}  noise={n_noi}  spark={n_spk}  baseline_bins={n_bl_bins}  "
        f"segments=[{seg_labels}]  "
        f"(from {len(catalog_paths)} catalog file(s))",
        flush=True,
    )
    run_baselines: list[float] | None = None
    if spans:
        run_baselines = []
        for span in spans:
            bl = float("nan")
            for path, rows in zip(catalog_paths, rows_per_path):
                if catalog_segment_label(path) == span.label:
                    bl = _run_baseline_mv(rows)
                    break
            run_baselines.append(bl)
    plot_trend(
        centers_s,
        signal_counts,
        noise_counts,
        spark_counts,
        baseline_mean,
        baseline_sem,
        bin_minutes=args.span,
        fit_exp=fit_exp,
        segment_spans=spans if spans else None,
        segment_run_baselines=run_baselines,
        compress_gaps=compress_gaps,
    )


if __name__ == "__main__":
    main()
