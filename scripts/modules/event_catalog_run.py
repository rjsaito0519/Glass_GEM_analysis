"""Build catalog rows for one run."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

from modules.common import (
    align_pair,
    discover_channel_roots,
    fft_is_noise_label,
    inferred_fall_time_us,
    is_spark_event,
    max_v_time_us,
    read_waveform_csv,
    sorted_waveform_csvs,
    trapz_y_x,
    window_arrays,
)
from modules.event_catalog_baseline import (
    _finite_float_or_none,
    baseline_gaussian_from_pool,
    save_baseline_pool_fit_png,
)
from modules.event_catalog_debug import (
    CatalogDebugEvent,
    make_debug_snapshot,
    run_catalog_debug_browser,
)


def mean_voltage_outside_window_mv(
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    tmin_us: float,
    tmax_us: float,
) -> float | None:
    """解析窓外サンプルの平均電圧 [mV]（窓外が無ければ None）。"""
    t = np.asarray(t_us, dtype=float)
    v = np.asarray(v_mv, dtype=float)
    m = (t < tmin_us) | (t > tmax_us)
    if not np.any(m):
        return None
    return float(np.mean(v[m]))


def extract_timestamp_from_filename(path: Path, *, pattern: re.Pattern[str], fmt: str) -> datetime | None:
    """ファイル名から正規表現で時刻を取り出してパースする。"""
    m = pattern.search(path.name)
    if m is None:
        return None
    token = m.group(1) if m.groups() else m.group(0)
    try:
        return datetime.strptime(token, fmt)
    except ValueError:
        return None


def load_ggem_nim_paths(
    run_dir: Path,
    csv_subdir: str,
    *,
    ggem_channel_id: int,
    nim_channel_id: int,
) -> tuple[list[Path], list[Path]] | None:
    """run 内の GGEM / NIM 波形 CSV 列（ソート済み）を返す。"""
    channels = discover_channel_roots(run_dir, csv_subdir=csv_subdir)
    if not channels:
        return None
    ggem_root = next((root for cid, root in channels if cid == ggem_channel_id), None)
    if ggem_root is None:
        return None
    p_ggem = sorted_waveform_csvs(ggem_root / csv_subdir)
    if not p_ggem:
        return None
    nim_root = next((root for cid, root in channels if cid == nim_channel_id), None)
    p_nim: list[Path] = sorted_waveform_csvs(nim_root / csv_subdir) if nim_root is not None else []
    return p_ggem, p_nim


def unique_result_subdir(output_root: Path, run_dir: Path, used: set[str]) -> Path:
    """結果出力用の衝突しないサブディレクトリを確保する。"""
    base = re.sub(r"[^\w\-.]+", "_", run_dir.name)[:80] or "run"
    cand = base
    i = 0
    h = abs(hash(str(run_dir.resolve()))) % 1_000_000
    while cand in used:
        i += 1
        cand = f"{base}_{h:x}_{i}"
    used.add(cand)
    return output_root / cand


def compute_dt_ggem_nim_ns(
    t_ggem_us: np.ndarray,
    v_ggem_mv: np.ndarray,
    tw: np.ndarray,
    vw_corr: np.ndarray,
    nim_paths: list[Path],
    k: int,
    *,
    edge_mv: float,
    nim_width_ns: float,
) -> float | None:
    """GGEM 窓内最大と NIM 落下の時刻差 [ns]。"""
    t_max_ggem = max_v_time_us(tw, vw_corr)
    if t_max_ggem is None:
        return None
    if k >= len(nim_paths):
        return None
    fp_nim = nim_paths[k]
    try:
        t_nim, v_nim = read_waveform_csv(fp_nim)
    except Exception:
        return None
    aligned = align_pair(t_ggem_us, v_ggem_mv, t_nim, v_nim)
    if aligned is None:
        return None
    _tga, _vga, t_na, v_na = aligned
    t_fall_nim = inferred_fall_time_us(t_na, v_na, v_th=edge_mv, nim_width_ns=nim_width_ns)
    if t_fall_nim is None:
        return None
    return float((float(t_max_ggem) - float(t_fall_nim)) * 1e3)


def _catalog_cell_mv(x: float) -> str:
    """mV 列の CSV セル（nan は空）。"""
    if x != x:
        return ""
    return f"{x:.3f}"


def _catalog_cell_dt_ns(dt_ns: float | None) -> str:
    """``dt_ggem_nim_ns`` 列の CSV セル。"""
    if dt_ns is None or not np.isfinite(dt_ns):
        return ""
    return f"{float(dt_ns):.12g}"


@dataclass
class EventCatalogRow:
    """カタログ CSV の 1 行（CSV 列はすべて文字列、``file_index`` / ``spark`` のみ int）。"""

    run_num: str
    ggem_csv: str
    timestamp: str
    file_index: int
    spark: int
    fft_is_noise: str
    baseline_mv: str
    baseline_indiv_mv: str
    integral_mv_us: str
    vmax_mv: str
    dt_ggem_nim_ns: str

    def as_csv_dict(self) -> dict[str, str]:
        """CSV 行用の文字列 dict。"""
        return {
            "run_num": self.run_num,
            "ggem_csv": self.ggem_csv,
            "timestamp": self.timestamp,
            "file_index": str(self.file_index),
            "spark": str(self.spark),
            "fft_is_noise": self.fft_is_noise,
            "baseline_mv": self.baseline_mv,
            "baseline_indiv_mv": self.baseline_indiv_mv,
            "integral_mv_us": self.integral_mv_us,
            "vmax_mv": self.vmax_mv,
            "dt_ggem_nim_ns": self.dt_ggem_nim_ns,
        }


def process_run_to_rows(
    run_dir: Path,
    *,
    run_num: str,
    csv_subdir: str,
    ggem_channel_id: int,
    nim_channel_id: int,
    tmin_us: float,
    tmax_us: float,
    spark_threshold_mv: float,
    spark_min_duration_us: float,
    noise_dominant_peak_min_hz: float,
    rel_tol_dt: float,
    subtract_mean_for_fft: bool,
    timestamp_pattern: re.Pattern[str],
    timestamp_format: str,
    edge_mv: float,
    nim_width_ns: float,
    baseline_png_path: Path | None = None,
    debug: bool = False,
) -> tuple[list[EventCatalogRow], dict[str, Any]]:
    """1 run 分を走査し、行リストと meta を返す。"""
    loaded = load_ggem_nim_paths(
        run_dir,
        csv_subdir,
        ggem_channel_id=ggem_channel_id,
        nim_channel_id=nim_channel_id,
    )
    if loaded is None:
        return [], {"error": "no_ggem_paths"}
    ggem_paths, nim_paths = loaded

    per_event_mean_outside: list[float] = []
    skipped_read_pool = 0
    for fp in tqdm(
        ggem_paths,
        desc=f"{run_num} baseline",
        unit="evt",
        file=sys.stderr,
        mininterval=0.2,
    ):
        try:
            t_us, v_mv = read_waveform_csv(fp)
        except Exception as exc:
            print(f"GGEM {fp.name}: {exc}", file=sys.stderr, flush=True)
            skipped_read_pool += 1
            continue
        m_out = mean_voltage_outside_window_mv(t_us, v_mv, tmin_us=tmin_us, tmax_us=tmax_us)
        if m_out is not None:
            per_event_mean_outside.append(m_out)

    pool = np.asarray(per_event_mean_outside, dtype=np.float64)
    center_run_mv, fit_info = baseline_gaussian_from_pool(pool)
    sigma_fit = fit_info.get("gaussian_fit_sigma_mv")
    hist_edges_png = fit_info.get("_hist_edges_for_png")

    if baseline_png_path is not None:
        save_baseline_pool_fit_png(
            pool,
            center_run_mv,
            _finite_float_or_none(sigma_fit),
            hist_edges_png,
            baseline_png_path,
            run_num=run_num,
            method=str(fit_info.get("method", "")),
            tmin_us=tmin_us,
            tmax_us=tmax_us,
        )

    rows: list[EventCatalogRow] = []
    debug_events: list[CatalogDebugEvent] = []
    skipped_read_catalog = 0
    skipped_window = 0
    n_spark = 0
    n_fft_signal = 0
    n_fft_noise = 0
    n_fft_unknown = 0

    for k, fp in enumerate(
        tqdm(
            ggem_paths,
            desc=f"{run_num} catalog",
            unit="evt",
            file=sys.stderr,
            mininterval=0.2,
        )
    ):
        ggem_name = fp.name
        ts = extract_timestamp_from_filename(fp, pattern=timestamp_pattern, fmt=timestamp_format)
        ts_str = ts.isoformat(sep=" ", timespec="seconds") if ts is not None else ""

        try:
            t_us, v_mv = read_waveform_csv(fp)
        except Exception as exc:
            print(f"GGEM {fp.name}: {exc}", file=sys.stderr, flush=True)
            skipped_read_catalog += 1
            rows.append(
                EventCatalogRow(
                    run_num=run_num,
                    ggem_csv=ggem_name,
                    timestamp=ts_str,
                    file_index=k,
                    spark=0,
                    fft_is_noise="",
                    baseline_mv=_catalog_cell_mv(center_run_mv),
                    baseline_indiv_mv="",
                    integral_mv_us="",
                    vmax_mv="",
                    dt_ggem_nim_ns="",
                )
            )
            if debug:
                debug_events.append(
                    make_debug_snapshot(
                        k=k,
                        ggem_name=ggem_name,
                        ts_str=ts_str,
                        t_ggem_us=np.array([]),
                        v_ggem_mv=np.array([]),
                        nim_paths=nim_paths,
                        tmin_us=tmin_us,
                        tmax_us=tmax_us,
                        center_run_mv=center_run_mv,
                        edge_mv=edge_mv,
                        nim_width_ns=nim_width_ns,
                        spark=False,
                        fft_is_noise="",
                        dt_ggem_nim_ns="",
                        subtract_mean_fft=subtract_mean_for_fft,
                        rel_tol_dt=rel_tol_dt,
                        noise_dominant_peak_min_hz=noise_dominant_peak_min_hz,
                        read_error=str(exc),
                    )
                )
            continue

        m_out_csv = mean_voltage_outside_window_mv(t_us, v_mv, tmin_us=tmin_us, tmax_us=tmax_us)
        baseline_indiv_mv = (
            float(m_out_csv) if m_out_csv is not None else float("nan")
        )

        spark = is_spark_event(
            t_us,
            v_mv,
            threshold_mv=spark_threshold_mv,
            min_duration_us=spark_min_duration_us,
            rel_tol_dt=rel_tol_dt,
        )
        if spark:
            n_spark += 1
        tw, vw = window_arrays(t_us, v_mv, tmin_us=tmin_us, tmax_us=tmax_us)
        if tw.size < 2:
            skipped_window += 1
            rows.append(
                EventCatalogRow(
                    run_num=run_num,
                    ggem_csv=ggem_name,
                    timestamp=ts_str,
                    file_index=k,
                    spark=int(spark),
                    fft_is_noise="",
                    baseline_mv=_catalog_cell_mv(center_run_mv),
                    baseline_indiv_mv=_catalog_cell_mv(baseline_indiv_mv),
                    integral_mv_us="",
                    vmax_mv="",
                    dt_ggem_nim_ns="",
                )
            )
            if debug:
                debug_events.append(
                    make_debug_snapshot(
                        k=k,
                        ggem_name=ggem_name,
                        ts_str=ts_str,
                        t_ggem_us=t_us,
                        v_ggem_mv=v_mv,
                        nim_paths=nim_paths,
                        tmin_us=tmin_us,
                        tmax_us=tmax_us,
                        center_run_mv=center_run_mv,
                        edge_mv=edge_mv,
                        nim_width_ns=nim_width_ns,
                        spark=bool(spark),
                        fft_is_noise="",
                        dt_ggem_nim_ns="",
                        subtract_mean_fft=subtract_mean_for_fft,
                        rel_tol_dt=rel_tol_dt,
                        noise_dominant_peak_min_hz=noise_dominant_peak_min_hz,
                    )
                )
            continue

        vw_c = np.asarray(vw, dtype=float) - center_run_mv
        integral_bc = trapz_y_x(vw_c, tw)
        vmax_bc = float(np.max(vw_c))

        if spark:
            fft_lab = ""
        else:
            lab = fft_is_noise_label(
                tw,
                vw_c,
                noise_dominant_peak_min_hz=noise_dominant_peak_min_hz,
                rel_tol_dt=rel_tol_dt,
                subtract_mean_for_fft=subtract_mean_for_fft,
            )
            if lab is None:
                fft_lab = ""
                n_fft_unknown += 1
            elif lab:
                fft_lab = "1"
                n_fft_noise += 1
            else:
                fft_lab = "0"
                n_fft_signal += 1

        dt_ns: float | None
        if spark:
            dt_ns = None
        else:
            dt_ns = compute_dt_ggem_nim_ns(
                t_us,
                v_mv,
                tw,
                vw_c,
                nim_paths,
                k,
                edge_mv=edge_mv,
                nim_width_ns=nim_width_ns,
            )
        dt_str = _catalog_cell_dt_ns(dt_ns)
        rows.append(
            EventCatalogRow(
                run_num=run_num,
                ggem_csv=ggem_name,
                timestamp=ts_str,
                file_index=k,
                spark=int(spark),
                fft_is_noise=fft_lab,
                baseline_mv=_catalog_cell_mv(center_run_mv),
                baseline_indiv_mv=_catalog_cell_mv(baseline_indiv_mv),
                integral_mv_us=_catalog_cell_mv(float(integral_bc)),
                vmax_mv=_catalog_cell_mv(vmax_bc),
                dt_ggem_nim_ns=dt_str,
            )
        )
        if debug:
            debug_events.append(
                make_debug_snapshot(
                    k=k,
                    ggem_name=ggem_name,
                    ts_str=ts_str,
                    t_ggem_us=t_us,
                    v_ggem_mv=v_mv,
                    nim_paths=nim_paths,
                    tmin_us=tmin_us,
                    tmax_us=tmax_us,
                    center_run_mv=center_run_mv,
                    edge_mv=edge_mv,
                    nim_width_ns=nim_width_ns,
                    spark=bool(spark),
                    fft_is_noise=fft_lab,
                    dt_ggem_nim_ns=dt_str,
                    subtract_mean_fft=subtract_mean_for_fft,
                    rel_tol_dt=rel_tol_dt,
                    noise_dominant_peak_min_hz=noise_dominant_peak_min_hz,
                )
            )

    if debug:
        run_catalog_debug_browser(debug_events, run_num=run_num)

    bfit: dict[str, Any] = {
        "model": fit_info.get("gaussian_fit_model"),
        "method": fit_info.get("method"),
        "center_mV": center_run_mv,
        "sigma_mV": fit_info.get("gaussian_fit_sigma_mv"),
        "sigma_source": fit_info.get("gaussian_fit_sigma_source"),
        "sigma_pooled_rmse_mV": fit_info.get("gaussian_fit_sigma_pooled_rmse_mv"),
        "sigma_sample_stdev_mV": fit_info.get("gaussian_fit_sigma_sample_stdev_mv"),
        "n_pool": fit_info.get("n_per_event_means", 0),
        "n_bins": fit_info.get("n_bins"),
        "hist_range_mv": fit_info.get("hist_range_mv"),
        "fit_success": fit_info.get("fit_success"),
        "fallback_reason": fit_info.get("fallback_reason"),
    }
    if "gaussian_fit_sample_mean_mv" in fit_info:
        bfit["gaussian_fit_sample_mean_mv"] = fit_info["gaussian_fit_sample_mean_mv"]
    if fit_info.get("gaussian_fit_lmfit_chisqr") is not None:
        bfit["lmfit_chisqr"] = fit_info["gaussian_fit_lmfit_chisqr"]
    if fit_info.get("gaussian_fit_lmfit_redchi") is not None:
        bfit["lmfit_redchi"] = fit_info["gaussian_fit_lmfit_redchi"]
    if fit_info.get("gaussian_fit_lmfit_nfev") is not None:
        bfit["lmfit_nfev"] = fit_info["gaussian_fit_lmfit_nfev"]

    meta: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "run_num": run_num,
        "csv_subdir": csv_subdir,
        "tmin_us": tmin_us,
        "tmax_us": tmax_us,
        "spark_threshold_mv": spark_threshold_mv,
        "spark_min_duration_us": spark_min_duration_us,
        "noise_dominant_peak_min_hz": noise_dominant_peak_min_hz,
        "rel_tol_dt": rel_tol_dt,
        "subtract_mean_fft": subtract_mean_for_fft,
        "timestamp_regex": timestamp_pattern.pattern,
        "timestamp_format": timestamp_format,
        "baseline_gaussian_fit": bfit,
        "center_run_mv": center_run_mv,
        "sigma_mle_mv": fit_info.get("gaussian_fit_sigma_mv"),
        "baseline_fit_method": fit_info.get("method"),
        "baseline_fit_png": baseline_png_path.name if baseline_png_path is not None else None,
        "n_baseline_per_event_means": fit_info.get("n_per_event_means", 0),
        "ggem_channel_id": ggem_channel_id,
        "nim_channel_id": nim_channel_id,
        "nim_paths_count": len(nim_paths),
        "edge_mv": edge_mv,
        "nim_width_ns": nim_width_ns,
        "n_ggem_files": len(ggem_paths),
        "skipped_read_pool_scan": skipped_read_pool,
        "skipped_read_catalog_scan": skipped_read_catalog,
        "skipped_window": skipped_window,
        "n_spark": n_spark,
        "n_fft_signal": n_fft_signal,
        "n_fft_noise": n_fft_noise,
        "n_fft_unknown_non_spark": n_fft_unknown,
        "cli_debug": debug,
    }
    n_ev_baseline = int(fit_info.get("n_per_event_means", 0))
    sig_s = fit_info.get("gaussian_fit_sigma_mv")
    src = str(fit_info.get("gaussian_fit_sigma_source", ""))
    if fit_info.get("fit_success"):
        sig_part = (
            ""
            if sig_s is None or not isinstance(sig_s, (int, float)) or not np.isfinite(sig_s)
            else f"  sigma_fit={float(sig_s):.6g} mV"
        )
    else:
        sig_part = (
            ""
            if sig_s is None or not isinstance(sig_s, (int, float)) or not np.isfinite(sig_s)
            else f"  sigma_diag={float(sig_s):.6g} mV"
        )
    src_part = f"  [{src}]" if src else ""
    print(
        f"[{run_num}] baseline: center={center_run_mv:.6g} mV{sig_part}{src_part}  "
        f"(n_pool={n_ev_baseline})",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[{run_num}] counts: GGEM_files={len(ggem_paths)}  "
        f"spark={n_spark}  fft_signal={n_fft_signal}  fft_noise={n_fft_noise}  "
        f"fft_unknown={n_fft_unknown}  skip_window={skipped_window}  "
        f"read_fail_pool={skipped_read_pool}  read_fail_catalog={skipped_read_catalog}",
        file=sys.stderr,
        flush=True,
    )
    return rows, meta
