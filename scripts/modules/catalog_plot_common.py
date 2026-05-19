"""Shared helpers for catalog-based matplotlib plots."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modules.event_catalog_io import (
    CATALOG_CSV_NAME,
    parse_float_cell,
    parse_fft_noise_cell,
    parse_timestamp_cell,
    read_catalog_rows,
    resolve_catalog_paths,
)

INTEGRAL_BIN_EDGES = np.linspace(0.0, 100.0, 101)
MAX_BIN_EDGES = np.linspace(0.0, 400.0, 51)

LABEL_SIGNAL = "signal"
LABEL_NOISE = "noise"


def require_matplotlib(script_name: str) -> None:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        print(
            f"[{script_name}] matplotlib required: pip install 'glassgem-analysis[viz]'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


def load_catalog_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    """Resolve paths and concatenate all catalog rows."""
    catalog_paths = resolve_catalog_paths(paths)
    if not catalog_paths:
        return [], []
    all_rows: list[dict[str, Any]] = []
    for csv_path in catalog_paths:
        all_rows.extend(read_catalog_rows(csv_path))
    return all_rows, catalog_paths


def is_spark_row(row: dict[str, Any]) -> bool:
    return (row.get("spark") or "").strip() == "1"


def rows_to_fft_samples(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """非スパークかつ FFT 分類済み行から (int_sig, max_sig, int_noise, max_noise)。"""
    int_sig: list[float] = []
    max_sig: list[float] = []
    int_noise: list[float] = []
    max_noise: list[float] = []
    for row in rows:
        if is_spark_row(row):
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
    per_run: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in rows:
        if is_spark_row(row):
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


def signal_samples_by_catalog(
    catalog_paths: Sequence[Path],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """PATH ごとに signal の (integral, vmax) を返す。ラベルは ``catalog_segment_label``。"""
    out: dict[str, tuple[list[float], list[float]]] = {}
    for csv_path in catalog_paths:
        int_sig, max_sig, _, _ = rows_to_fft_samples(read_catalog_rows(csv_path))
        if int_sig.size == 0:
            print(f"[skip] no FFT-signal rows in {csv_path}", file=sys.stderr, flush=True)
            continue
        label = catalog_segment_label(csv_path)
        if label in out:
            ints, maxs = out[label]
            ints.extend(int_sig.tolist())
            maxs.extend(max_sig.tolist())
        else:
            out[label] = (int_sig.tolist(), max_sig.tolist())
    return {
        k: (np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float))
        for k, v in out.items()
    }


def dt_signal_samples_by_catalog(
    catalog_paths: Sequence[Path],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """PATH ごとに signal の (dt, integral, vmax) リストを返す。"""
    out: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for csv_path in catalog_paths:
        dt_sig, _, int_sig, max_sig = rows_to_dt_signal_lists(read_catalog_rows(csv_path))
        if not dt_sig:
            print(f"[skip] no dt signal rows in {csv_path}", file=sys.stderr, flush=True)
            continue
        label = catalog_segment_label(csv_path)
        if label in out:
            dt_p, int_p, max_p = out[label]
            dt_p.extend(dt_sig)
            int_p.extend(int_sig)
            max_p.extend(max_sig)
        else:
            out[label] = (list(dt_sig), list(int_sig), list(max_sig))
    return out


def rows_to_dt_signal_lists(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """(dt_sig, dt_noise, int_sig, max_sig) — signal 行のみ int/max は dt と同順。"""
    dt_sig: list[float] = []
    dt_noise: list[float] = []
    int_sig: list[float] = []
    max_sig: list[float] = []
    for row in rows:
        if is_spark_row(row):
            continue
        fft_lab = parse_fft_noise_cell(row.get("fft_is_noise", ""))
        if fft_lab is None:
            continue
        dt = parse_float_cell(row.get("dt_ggem_nim_ns", ""))
        if not np.isfinite(dt):
            continue
        if fft_lab:
            dt_noise.append(dt)
            continue
        integral = parse_float_cell(row.get("integral_mv_us", ""))
        vmax = parse_float_cell(row.get("vmax_mv", ""))
        if not np.isfinite(integral) or not np.isfinite(vmax):
            continue
        dt_sig.append(dt)
        int_sig.append(integral)
        max_sig.append(vmax)
    return dt_sig, dt_noise, int_sig, max_sig


def _baseline_mean_sem_from_sums(
    baseline_sum: np.ndarray,
    baseline_sq_sum: np.ndarray,
    baseline_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ビンごとの ``baseline_indiv_mv`` 平均と標準誤差 [mV]。"""
    n_bins = baseline_n.size
    mean = np.full(n_bins, np.nan, dtype=float)
    sem = np.full(n_bins, np.nan, dtype=float)
    has = baseline_n > 0
    if not np.any(has):
        return mean, sem
    mean[has] = baseline_sum[has] / baseline_n[has]
    multi = baseline_n >= 2
    if np.any(multi):
        var = (baseline_sq_sum[multi] - (baseline_sum[multi] ** 2) / baseline_n[multi]) / (
            baseline_n[multi] - 1
        )
        sem[multi] = np.sqrt(np.maximum(var, 0.0) / baseline_n[multi])
    single = (baseline_n == 1) & has
    sem[single] = 0.0
    return mean, sem


def bin_events_by_time(
    rows: list[dict[str, Any]],
    *,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """``(edges_s, signal, noise, spark, baseline_mean, baseline_sem, n_skip_ts)``。"""
    if bin_minutes <= 0.0:
        raise ValueError("bin_minutes must be > 0")

    bin_s = float(bin_minutes) * 60.0
    rows_with_ts: list[dict[str, Any]] = []
    times_s: list[float] = []
    is_spark: list[bool] = []
    fft_noise: list[bool | None] = []
    n_skip_ts = 0

    for row in rows:
        ts = parse_timestamp_cell(row.get("timestamp", ""))
        if ts is None:
            n_skip_ts += 1
            continue
        rows_with_ts.append(row)
        times_s.append(ts.timestamp())
        is_spark.append(is_spark_row(row))
        fft_noise.append(parse_fft_noise_cell(row.get("fft_is_noise", "")))

    if not times_s:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, n_skip_ts

    t_arr = np.asarray(times_s, dtype=float)
    t0 = float(np.min(t_arr))
    t1 = float(np.max(t_arr))
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if edges.size < 2:
        edges = np.array([t0, t0 + bin_s], dtype=float)

    n_bins = edges.size - 1
    signal_counts = np.zeros(n_bins, dtype=int)
    noise_counts = np.zeros(n_bins, dtype=int)
    spark_counts = np.zeros(n_bins, dtype=int)
    baseline_sum = np.zeros(n_bins, dtype=float)
    baseline_sq_sum = np.zeros(n_bins, dtype=float)
    baseline_n = np.zeros(n_bins, dtype=int)

    for row, t, spark, fft_lab in zip(rows_with_ts, times_s, is_spark, fft_noise):
        idx = int(np.searchsorted(edges, t, side="right") - 1)
        idx = max(0, min(idx, n_bins - 1))
        if spark:
            spark_counts[idx] += 1
        elif fft_lab is False:
            signal_counts[idx] += 1
        elif fft_lab is True:
            noise_counts[idx] += 1
        bl = parse_float_cell(row.get("baseline_indiv_mv", ""))
        if not spark and np.isfinite(bl):
            baseline_sum[idx] += bl
            baseline_sq_sum[idx] += bl * bl
            baseline_n[idx] += 1

    baseline_mean, baseline_sem = _baseline_mean_sem_from_sums(
        baseline_sum, baseline_sq_sum, baseline_n
    )
    return edges, signal_counts, noise_counts, spark_counts, baseline_mean, baseline_sem, n_skip_ts


@dataclass(frozen=True)
class TrendSegment:
    """1 入力カタログ分の時間ビン集計。"""

    label: str
    edges_s: np.ndarray
    signal_counts: np.ndarray
    noise_counts: np.ndarray
    spark_counts: np.ndarray
    baseline_mean: np.ndarray
    baseline_sem: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.signal_counts.size)


@dataclass(frozen=True)
class TrendSegmentSpan:
    """結合後配列上の区間 ``[start, end)`` とラベル。"""

    label: str
    start: int
    end: int


def catalog_segment_label(path: Path) -> str:
    """プロット用の短いラベル（``results/run0089`` など）。"""
    p = path.resolve()
    if p.name == CATALOG_CSV_NAME:
        return p.parent.name
    return p.stem


def segment_bin_centers_s(edges_s: np.ndarray) -> np.ndarray:
    """ビン境界からビン中心 [s] を返す。"""
    return 0.5 * (edges_s[:-1] + edges_s[1:])


def merge_trend_segments(
    segments: Sequence[TrendSegment],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[TrendSegmentSpan],
]:
    """PATH 順にビン系列を連結し、区間メタデータを返す。

    返る第1要素は **ビン中心** ``centers_s``（``edges`` の連結ではない）。
  """
    if not segments:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, []

    center_parts: list[np.ndarray] = []
    sig_parts: list[np.ndarray] = []
    noise_parts: list[np.ndarray] = []
    spark_parts: list[np.ndarray] = []
    bl_mean_parts: list[np.ndarray] = []
    bl_sem_parts: list[np.ndarray] = []
    spans: list[TrendSegmentSpan] = []
    offset = 0

    for seg in segments:
        n = seg.n_bins
        if n == 0:
            continue
        center_parts.append(segment_bin_centers_s(seg.edges_s))
        sig_parts.append(seg.signal_counts)
        noise_parts.append(seg.noise_counts)
        spark_parts.append(seg.spark_counts)
        bl_mean_parts.append(seg.baseline_mean)
        bl_sem_parts.append(seg.baseline_sem)
        spans.append(TrendSegmentSpan(label=seg.label, start=offset, end=offset + n))
        offset += n

    if not spans:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, []

    centers_s = np.concatenate(center_parts)
    return (
        centers_s,
        np.concatenate(sig_parts),
        np.concatenate(noise_parts),
        np.concatenate(spark_parts),
        np.concatenate(bl_mean_parts),
        np.concatenate(bl_sem_parts),
        spans,
    )


def gap_compress_min_minutes() -> float:
    """この分数以上のカレンダー空きだけギャップ圧縮する。"""
    return 60.0


def gap_compress_margin_days() -> float:
    """ギャップ圧縮後に意図的に残す区切り余白 [days]（60 分）。"""
    return 60.0 / (24.0 * 60.0)


def gap_compress_margin_days_from_bin_minutes(bin_minutes: float) -> float:
    """後方互換: 圧縮時の区切り余白（``bin_minutes`` 非依存）。"""
    _ = bin_minutes
    return gap_compress_margin_days()


def catalog_time_extent_s(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """カタログ行の timestamp 範囲 [s]（最早, 最遅）。パース不能のみなら None。"""
    times: list[float] = []
    for row in rows:
        ts = parse_timestamp_cell(row.get("timestamp", ""))
        if ts is not None:
            times.append(ts.timestamp())
    if not times:
        return None
    return float(min(times)), float(max(times))


def path_pair_gap_minutes(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> float | None:
    """PATH 順の隣接カタログ間: B の最早 − A の最遅 [min]。"""
    ext_a = catalog_time_extent_s(rows_a)
    ext_b = catalog_time_extent_s(rows_b)
    if ext_a is None or ext_b is None:
        return None
    return (ext_b[0] - ext_a[1]) / 60.0


def max_path_gap_minutes(row_groups: Sequence[list[dict[str, Any]]]) -> float:
    """隣接 PATH 間の最大カレンダーギャップ [min]（1 本なら 0）。"""
    if len(row_groups) < 2:
        return 0.0
    gaps: list[float] = []
    for i in range(len(row_groups) - 1):
        g = path_pair_gap_minutes(row_groups[i], row_groups[i + 1])
        if g is not None:
            gaps.append(float(g))
    return max(gaps) if gaps else 0.0


def gap_margin_days_from_bin_minutes(bin_minutes: float) -> float:
    """後方互換: 圧縮時の区切り余白。"""
    return gap_compress_margin_days_from_bin_minutes(bin_minutes)


def gap_slot_days_from_bin_minutes(bin_minutes: float) -> float:
    """後方互換エイリアス。"""
    return gap_compress_margin_days_from_bin_minutes(bin_minutes)


def compress_path_gaps_xnum(
    xnum: np.ndarray,
    spans: Sequence[TrendSegmentSpan],
    *,
    min_gap_minutes: float | None = None,
    compress_margin_days: float | None = None,
) -> np.ndarray:
    """セグメント別ビンを描画 x で連結する。

    - カレンダー差 >= ``min_gap_minutes``（既定 60 分）: 潰して ``compress_margin_days`` だけ空ける
    - それ未満: 実カレンダー差（分）をそのままプロット上の隙間にする
    """
    x_arr = np.asarray(xnum, dtype=float)
    if x_arr.size == 0 or len(spans) <= 1:
        return x_arr.copy()

    gap_min = gap_compress_min_minutes() if min_gap_minutes is None else float(min_gap_minutes)
    min_gap_days = gap_min / (24.0 * 60.0)
    compress_margin = (
        float(compress_margin_days)
        if compress_margin_days is not None
        else gap_compress_margin_days()
    )

    x_plot = x_arr.copy()
    for i in range(1, len(spans)):
        prev = spans[i - 1]
        curr = spans[i]
        if prev.end <= 0 or curr.start >= x_arr.size:
            continue
        raw_hi = float(x_arr[curr.start])
        gap_cal = raw_hi - float(x_arr[prev.end - 1])
        prev_end_plot = float(x_plot[prev.end - 1])
        margin = compress_margin if gap_cal >= min_gap_days else gap_cal
        offset = prev_end_plot + margin - raw_hi
        x_plot[curr.start : curr.end] = x_arr[curr.start : curr.end] + offset

    return x_plot


def prepare_trend_binned(
    catalog_paths: Sequence[Path],
    rows_per_path: Sequence[list[dict[str, Any]]],
    *,
    bin_minutes: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[TrendSegmentSpan],
    str,
    int,
]:
    """トレンド用ビン集計。``mode`` は ``continuous`` または ``segmented``。"""
    if len(catalog_paths) != len(rows_per_path):
        raise ValueError("catalog_paths and rows_per_path length mismatch")

    n_skip = 0
    min_gap = gap_compress_min_minutes()
    use_segmented = len(rows_per_path) > 1 and max_path_gap_minutes(rows_per_path) >= min_gap

    if not use_segmented:
        all_rows: list[dict[str, Any]] = []
        for rows in rows_per_path:
            all_rows.extend(rows)
        edges, sig, noise, spark, bl_mean, bl_sem, skip = bin_events_by_time(
            all_rows, bin_minutes=bin_minutes
        )
        n_skip = skip
        if edges.size < 2:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty, empty, [], "continuous", n_skip
        centers = segment_bin_centers_s(edges)
        return centers, sig, noise, spark, bl_mean, bl_sem, [], "continuous", n_skip

    segments: list[TrendSegment] = []
    for path, rows in zip(catalog_paths, rows_per_path):
        edges, sig, noise, spark, bl_mean, bl_sem, skip = bin_events_by_time(
            rows, bin_minutes=bin_minutes
        )
        n_skip += skip
        if edges.size < 2:
            continue
        segments.append(
            TrendSegment(
                label=catalog_segment_label(path),
                edges_s=edges,
                signal_counts=sig,
                noise_counts=noise,
                spark_counts=spark,
                baseline_mean=bl_mean,
                baseline_sem=bl_sem,
            )
        )
    if not segments:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, [], "segmented", n_skip
    centers, sig, noise, spark, bl_mean, bl_sem, spans = merge_trend_segments(segments)
    return centers, sig, noise, spark, bl_mean, bl_sem, spans, "segmented", n_skip


def draw_step_series(
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


def dt_bin_edges_auto(
    dt_signal: list[float],
    dt_noise: list[float],
    *,
    n_bins: int,
) -> np.ndarray:
    s = np.asarray(dt_signal, dtype=float)
    n_arr = np.asarray(dt_noise, dtype=float)
    if s.size == 0 and n_arr.size == 0:
        return np.linspace(-8.0, 8.0, n_bins + 1)
    allv = np.concatenate([s, n_arr]) if s.size and n_arr.size else (s if s.size else n_arr)
    lo_d, hi_d = float(np.min(allv)), float(np.max(allv))
    span = hi_d - lo_d
    pad = 0.05 * span if span > 0 else max(1e-12, 0.01 * max(abs(lo_d), abs(hi_d), 1.0))
    return np.linspace(lo_d - pad, hi_d + pad, n_bins + 1)
