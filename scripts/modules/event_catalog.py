"""Waveform event catalog (one row per GGEM CSV): run-wide baseline, spark, features, GGEM–NIM timing.

Run-wide **baseline** is the Gaussian fit **μ** (``lmfit``) to a **Freedman–Diaconis** histogram of
per-event mean voltages [mV] **outside** the analysis window (one value per GGEM CSV in the run).
The model is a bin-integrated normal with total count fixed to ``N``; on failure or fewer than two
histogram bins, the code falls back to the sample mean. Extras: ``pip install -e ".[catalog,viz]"``
for ``lmfit`` and baseline PNG plots.

**CLI（リポジトリ内の解析スクリプト共通）**: 末尾に可変長 ``RUN_DIR``（指定時は設定の ``run_dirs`` を上書き）、
``--conf``、``--debug`` のみ。デバッグ時の波形 PNG 枚数上限は ``debug_max_waveforms`` を JSON で指定。

Each row adds spark flag, optional **FFT-based** noise/signal label, integral / vmax, and Δt vs NIM.

GGEM = main GEM waveform channel; NIM = NIM logic channel for edge timing. No extra channel gate
beyond spark for the FFT label path.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Sequence
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

from modules.cli_common import add_standard_cli_arguments
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

CATALOG_CSV_COLUMNS = (
    "run_num",         # この行が属する run の識別子（通常は run ディレクトリ名など）
    "ggem_csv",        # 主波形（GGEM）の CSV ファイル名（stem ではなくファイル名）
    "timestamp",       # 設定の正規表現・日時書式からファイル名に埋め込まれた時刻（失敗時は空文字）
    "file_index",      # 当 run 内でソートされた GGEM CSV 列の 0 始まりインデックス
    "spark",           # 全レンジ上でスパークと判定されれば 1、そうでなければ 0
    "fft_is_noise",    # スパーク時は空。非スパークのみ: PSD ノイズ判定なら "1"、信号側 "0"、判定不能 ""
    "baseline_mv",     # run 全体: 窓外平均プールのヒストに lmfit ビン積分ガウスを当てた μ [mV]（失敗時は算術平均）。窓内から減算
    "integral_mv_us",  # 解析窓内、(v − baseline) の台形積分（単位は mV·µs）
    "vmax_mv",         # 解析窓内、(v − baseline) の最大値 [mV]
    "dt_ggem_nim_ns",  # スパーク時は空。非スパーク: GGEM 窓内最大時刻と NIM 落下推定時刻の差 [ns]（符号は設定）
)


def _finite_float_or_none(x: Any) -> float | None:
    """任意値を有限の float に正規化し、不可なら ``None``。"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def _require_lmfit() -> None:
    """lmfit が無い場合は分かりやすい :exc:`ImportError` を出す。"""
    try:
        import lmfit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Baseline histogram Gaussian fit requires lmfit. "
            "Install: pip install 'glassgem-analysis[catalog]'"
        ) from e


def _baseline_histogram_edges_counts(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Freedman–Diaconis ビン境界でヒストグラムを返す ``(counts, edges)``。"""
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    lo, hi = float(np.min(p)), float(np.max(p))
    if lo == hi:
        edges = np.linspace(lo - 0.5, hi + 0.5, 11)
    else:
        edges = np.histogram_bin_edges(p, bins="fd")
        if edges.size < 2:
            edges = np.linspace(lo, hi, 11)
    counts, edges = np.histogram(p, bins=edges)
    if counts.size == 0:
        edges = np.linspace(lo - 0.5, hi + 0.5, 11)
        counts, edges = np.histogram(p, bins=edges)
    return counts.astype(np.float64), edges.astype(np.float64)


def _fit_gaussian_bin_integral_lmfit(
    p: np.ndarray,
    counts: np.ndarray,
    edges: np.ndarray,
    n: int,
) -> tuple[float, float | None, bool, str | None, Any]:
    """ビン積分ガウス（総度数 ``n`` 固定）を lmfit で当てはめ、``(μ, σ, 成功, 失敗理由, MinimizerResult|None)`` を返す。

    ``minimize`` が完了するたびに :func:`lmfit.fit_report` 相当の全文を標準エラーへ出す。
    """
    from lmfit import Parameters, fit_report, minimize
    from scipy.stats import norm

    d_lo = float(np.min(p))
    d_hi = float(np.max(p))
    d_span = max(d_hi - d_lo, 1e-12)
    mu0 = float(np.median(p))
    sig0 = float(np.std(p, ddof=1)) if n > 1 else 1e-3
    sig0 = max(sig0, 1e-6)

    def residual(params: Any, c: np.ndarray, e: np.ndarray, n_tot: int) -> np.ndarray:
        mu = float(params["mu"].value)
        sigma = float(params["sigma"].value)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
            return np.full(c.shape, 1e6, dtype=np.float64)
        z1 = (e[1:] - mu) / sigma
        z0 = (e[:-1] - mu) / sigma
        exp_counts = n_tot * (norm.cdf(z1) - norm.cdf(z0))
        eps = 1e-9
        return ((c - exp_counts) / np.sqrt(np.maximum(exp_counts, eps))).astype(np.float64)

    pars = Parameters()
    pars.add("mu", value=mu0, min=d_lo - 3.0 * d_span, max=d_hi + 3.0 * d_span)
    pars.add("sigma", value=sig0, min=1e-9, max=max(50.0 * sig0, 5.0 * d_span, 1e-3))

    try:
        out = minimize(residual, pars, args=(counts, edges, n), method="leastsq")
    except Exception as exc:
        print(f"[baseline lmfit] minimize raised: {exc}", file=sys.stderr, flush=True)
        return float(np.mean(p)), None, False, "minimize_exception", None

    print(
        "[baseline lmfit] bin-integral Gaussian (weighted residual), method=leastsq\n"
        + fit_report(out),
        file=sys.stderr,
        flush=True,
    )

    mu = float(out.params["mu"].value)
    sigma = float(out.params["sigma"].value)
    reason: str | None = None
    ok = bool(out.success) and np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-8
    if ok and sigma > 5.0 * d_span + 1e-6:
        ok = False
        reason = "sigma_too_large"
    elif not ok:
        reason = str(getattr(out, "message", "fit_failed"))[:200]

    if not ok:
        print(
            f"[baseline lmfit] post-fit validation failed ({reason!r}); using sample mean fallback.",
            file=sys.stderr,
            flush=True,
        )
        return float(np.mean(p)), None, False, reason or "fit_failed", out

    return mu, sigma, True, None, out


def baseline_gaussian_from_pool(pool_mv: np.ndarray) -> tuple[float, dict[str, Any]]:
    """窓外平均 [mV] のプールについて FD ヒスト + lmfit ビン積分ガウスで μ を推定する（失敗時は算術平均）。"""
    p = np.asarray(pool_mv, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = int(p.size)
    mu_mean = float(np.mean(p)) if n > 0 else 0.0
    rmse_about_mean = float(np.sqrt(np.mean((p - mu_mean) ** 2))) if n > 0 else 0.0
    stdev = float(np.std(p, ddof=1)) if n > 1 else None

    if n == 0:
        print(
            "[baseline lmfit] skipped: empty pool (no per-event outside-window means).",
            file=sys.stderr,
            flush=True,
        )
        info: dict[str, Any] = {
            "method": "empty_pool",
            "n_per_event_means": 0,
            "n_bins": 0,
            "hist_range_mv": None,
            "fit_success": False,
            "fallback_reason": None,
            "gaussian_fit_model": "gaussian_bin_integral_lmfit",
            "gaussian_fit_mu_mv": 0.0,
            "gaussian_fit_sigma_mv": None,
            "gaussian_fit_sample_mean_mv": 0.0,
            "gaussian_fit_sigma_pooled_rmse_mv": None,
            "gaussian_fit_sigma_sample_stdev_mv": None,
            "gaussian_fit_sigma_source": "none",
            "_hist_edges_for_png": None,
        }
        return 0.0, info

    if n == 1:
        print(
            "[baseline lmfit] skipped: N=1 (no histogram / leastsq fit).",
            file=sys.stderr,
            flush=True,
        )
        v0 = float(p[0])
        info = {
            "method": "single_event",
            "n_per_event_means": 1,
            "n_bins": 0,
            "hist_range_mv": [v0, v0],
            "fit_success": True,
            "fallback_reason": None,
            "gaussian_fit_model": "gaussian_bin_integral_lmfit",
            "gaussian_fit_mu_mv": v0,
            "gaussian_fit_sigma_mv": None,
            "gaussian_fit_sample_mean_mv": v0,
            "gaussian_fit_sigma_pooled_rmse_mv": 0.0,
            "gaussian_fit_sigma_sample_stdev_mv": None,
            "gaussian_fit_sigma_source": "single_point_no_sigma",
            "_hist_edges_for_png": None,
        }
        return v0, info

    _require_lmfit()
    counts, edges = _baseline_histogram_edges_counts(p)
    n_bins = int(counts.size)
    hist_lo, hist_hi = float(edges[0]), float(edges[-1])

    if n_bins < 2:
        print(
            "[baseline lmfit] skipped: histogram has fewer than 2 bins (no leastsq fit).",
            file=sys.stderr,
            flush=True,
        )
        info = {
            "method": "histogram_lt2_bins_fallback_mean",
            "n_per_event_means": n,
            "n_bins": n_bins,
            "hist_range_mv": [hist_lo, hist_hi],
            "fit_success": False,
            "fallback_reason": "histogram_lt2_bins",
            "gaussian_fit_model": "gaussian_bin_integral_lmfit",
            "gaussian_fit_mu_mv": mu_mean,
            "gaussian_fit_sigma_mv": rmse_about_mean,
            "gaussian_fit_sample_mean_mv": mu_mean,
            "gaussian_fit_sigma_pooled_rmse_mv": rmse_about_mean,
            "gaussian_fit_sigma_sample_stdev_mv": stdev,
            "gaussian_fit_sigma_source": "pooled_rmse_about_mean",
            "_hist_edges_for_png": edges,
        }
        return mu_mean, info

    mu_hat, sigma_hat, ok, fail_reason, lm_out = _fit_gaussian_bin_integral_lmfit(p, counts, edges, n)
    rmse_about_mu_hat = float(np.sqrt(np.mean((p - mu_hat) ** 2)))

    if ok:
        method = "gaussian_hist_fit"
        sigma_src = "lmfit_bin_integral"
        mu_out = float(mu_hat)
        sigma_out = float(sigma_hat) if sigma_hat is not None else None
    else:
        method = "gaussian_hist_fit_fallback_mean"
        sigma_src = "pooled_rmse_about_mean"
        mu_out = float(mu_hat)
        sigma_out = rmse_about_mu_hat

    info = {
        "method": method,
        "n_per_event_means": n,
        "n_bins": n_bins,
        "hist_range_mv": [hist_lo, hist_hi],
        "fit_success": ok,
        "fallback_reason": fail_reason if not ok else None,
        "gaussian_fit_model": "gaussian_bin_integral_lmfit",
        "gaussian_fit_mu_mv": mu_out,
        "gaussian_fit_sigma_mv": sigma_out,
        "gaussian_fit_sample_mean_mv": mu_mean,
        "gaussian_fit_sigma_pooled_rmse_mv": rmse_about_mu_hat,
        "gaussian_fit_sigma_sample_stdev_mv": stdev,
        "gaussian_fit_sigma_source": sigma_src,
        "_hist_edges_for_png": edges,
    }
    if lm_out is not None:
        info["gaussian_fit_lmfit_chisqr"] = getattr(lm_out, "chisqr", None)
        info["gaussian_fit_lmfit_redchi"] = getattr(lm_out, "redchi", None)
        info["gaussian_fit_lmfit_nfev"] = getattr(lm_out, "nfev", None)
    return mu_out, info


def save_baseline_pool_fit_png(
    pool_mv: np.ndarray,
    mu_fit_mv: float,
    sigma_fit_mv: float | None,
    hist_edges: np.ndarray | None,
    out_path: Path,
    *,
    run_num: str,
    method: str,
    tmin_us: float,
    tmax_us: float,
) -> None:
    """フィットと同一の ``hist_edges`` でヒストを描き、積分ガウスに相当する曲線を重ねる。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    p = np.asarray(pool_mv, dtype=np.float64)
    p = p[np.isfinite(p)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    if p.size == 0:
        ax.text(
            0.5,
            0.5,
            "No per-event outside-window means\n(baseline = 0 mV; empty pool)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"{run_num}  |  Baseline pool (empty)  |  window [{tmin_us:g}, {tmax_us:g}] µs")
    elif p.size == 1:
        v0 = float(p[0])
        ax.axvline(v0, color="C3", lw=2.0, label=rf"$\mu$={v0:.5g} mV (N=1)")
        ax.set_xlim(v0 - 1.0, v0 + 1.0)
        ax.set_xlabel("Mean voltage [mV] outside window (one value per GGEM CSV)")
        ax.set_ylabel("Counts per bin")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(
            f"{run_num}  |  Baseline pool (N=1)  |  window [{tmin_us:g}, {tmax_us:g}] µs"
        )
    else:
        n_ev = int(p.size)
        edges = hist_edges
        if edges is None or edges.size < 2 or not np.all(np.isfinite(edges)):
            _, edges = _baseline_histogram_edges_counts(p)
        counts, edges = np.histogram(p, bins=edges)
        n_bins = int(counts.size)
        x_lo, x_hi = float(edges[0]), float(edges[-1])
        bin_w = (x_hi - x_lo) / float(max(n_bins, 1))

        ax.hist(
            p,
            bins=edges,
            color="0.82",
            edgecolor="0.45",
            linewidth=0.25,
            label="Per-event mean mV (outside window), counts",
        )

        xx = np.linspace(x_lo, x_hi, max(200, n_bins * 8))
        sigma = sigma_fit_mv
        if sigma is not None and np.isfinite(sigma) and float(sigma) > 1e-12:
            yy = n_ev * bin_w * norm.pdf(xx, loc=float(mu_fit_mv), scale=float(sigma))
            ax.plot(
                xx,
                yy,
                color="C3",
                lw=2.0,
                label=rf"Gaussian fit ($\mu$={mu_fit_mv:.5g}, $\sigma$={float(sigma):.5g} mV)",
            )
        else:
            sigma_rmse = float(np.sqrt(np.mean((p - mu_fit_mv) ** 2)))
            if sigma_rmse > 1e-12:
                yy = n_ev * bin_w * norm.pdf(xx, loc=float(mu_fit_mv), scale=sigma_rmse)
                ax.plot(
                    xx,
                    yy,
                    color="C3",
                    lw=2.0,
                    label=rf"RMSE width ($\sigma$={sigma_rmse:.5g} mV)",
                )
            else:
                ax.axvline(mu_fit_mv, color="C3", lw=2.0, label=rf"$\mu$={mu_fit_mv:.5g} mV")

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Mean voltage [mV] outside window (one value per GGEM CSV)")
        ax.set_ylabel("Counts per bin")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(
            f"{run_num}  |  Baseline ({method})  |  window [{tmin_us:g}, {tmax_us:g}] µs  "
            f"|  N_events={n_ev}  |  bins={n_bins}"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mean_voltage_outside_window_mv(
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    tmin_us: float,
    tmax_us: float,
) -> float | None:
    """解析窓 ``[tmin_us, tmax_us]`` の外側サンプルにおける平均電圧 [mV]（イベント 1 個あたり 1 スカラー）。"""
    t = np.asarray(t_us, dtype=float)
    v = np.asarray(v_mv, dtype=float)
    m = (t < tmin_us) | (t > tmax_us)
    if not np.any(m):
        return None
    return float(np.mean(v[m]))


def extract_timestamp_from_filename(path: Path, *, pattern: re.Pattern[str], fmt: str) -> datetime | None:
    """ファイル名から正規表現でトークンを取り、``fmt`` で :class:`~datetime.datetime` にパースする。"""
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
    """run 直下の ``CH*`` から GGEM / NIM の波形 CSV 列（ソート済み）を取得する。"""
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
    """``output_root`` 配下に、run 名ベースで衝突しない結果サブディレクトリ名を確保して返す。"""
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
    ggem_peak_minus_nim_fall: bool,
) -> float | None:
    """同一イベントの GGEM 窓内最大時刻と NIM 落下推定時刻の差を ns で返す（符号は ``ggem_peak_minus_nim_fall``）。"""
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
    dt_us = (
        (float(t_max_ggem) - float(t_fall_nim))
        if ggem_peak_minus_nim_fall
        else (float(t_fall_nim) - float(t_max_ggem))
    )
    return float(dt_us * 1e3)


_DEBUG_MATPLOTLIB_WARNED = False


def _catalog_debug_waveform_cap(cfg: dict[str, Any]) -> int | None:
    """``debug_max_waveforms``: 正の整数 = 最大枚数、``None`` / 負 = 無制限、``0`` = プロットなし（ログのみ）。"""
    v = cfg.get("debug_max_waveforms", 50)
    if v is None:
        return None
    n = int(v)
    return None if n < 0 else n


def _debug_plot_waveform(out_path: Path, t_us: np.ndarray, v_mv: np.ndarray, *, title: str) -> bool:
    """デバッグ用に 1 波形を PNG に保存する。matplotlib が無ければ ``False`` を返し一度だけ警告する。"""
    global _DEBUG_MATPLOTLIB_WARNED
    try:
        import matplotlib

        matplotlib.use("Agg")
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.plot(np.asarray(t_us, dtype=float), np.asarray(v_mv, dtype=float), lw=0.7)
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


@dataclass
class EventCatalogRow:
    """イベントカタログの 1 行（CSV 1 行と対応）。"""

    run_num: str
    ggem_csv: str
    timestamp: str
    file_index: int
    spark: int
    fft_is_noise: str
    baseline_mv: float
    integral_mv_us: float
    vmax_mv: float
    dt_ggem_nim_ns: str

    def as_csv_dict(self) -> dict[str, str]:
        """CSV 書き込み用の文字列 dict（列名は :data:`CATALOG_CSV_COLUMNS` に一致）。"""

        def fmt_mv(x: float) -> str:
            """mV 系セル用フォーマット（NaN は空文字）。"""
            if x != x:
                return ""
            return f"{x:.12g}"

        return {
            "run_num": self.run_num,
            "ggem_csv": self.ggem_csv,
            "timestamp": self.timestamp,
            "file_index": str(self.file_index),
            "spark": str(self.spark),
            "fft_is_noise": self.fft_is_noise,
            "baseline_mv": f"{self.baseline_mv:.12g}",
            "integral_mv_us": fmt_mv(self.integral_mv_us),
            "vmax_mv": fmt_mv(self.vmax_mv),
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
    ggem_peak_minus_nim_fall: bool,
    baseline_png_path: Path | None = None,
    debug: bool = False,
    debug_max_waveforms: int | None = 50,
    debug_waveform_dir: Path | None = None,
) -> tuple[list[EventCatalogRow], dict[str, Any]]:
    """1 run 分の GGEM CSV を走査し、ベースライン推定・特徴量付きの :class:`EventCatalogRow` 列と meta を構築する。

    ``debug`` が真のときは各ファイルを標準エラーに出し、``debug_waveform_dir`` が与えられていれば
    最大 ``debug_max_waveforms`` 枚まで GGEM 全レンジ波形を PNG に保存する（``None`` は無制限）。
    """
    loaded = load_ggem_nim_paths(
        run_dir,
        csv_subdir,
        ggem_channel_id=ggem_channel_id,
        nim_channel_id=nim_channel_id,
    )
    if loaded is None:
        return [], {"error": "no_ggem_paths"}
    ggem_paths, nim_paths = loaded

    plot_waveforms = (
        debug
        and debug_waveform_dir is not None
        and (debug_max_waveforms is None or debug_max_waveforms > 0)
    )
    n_debug_plots = 0

    per_event_mean_outside: list[float] = []
    skipped_read_pool = 0
    for fp in tqdm(
        ggem_paths,
        desc=f"{run_num} baseline",
        unit="evt",
        file=sys.stderr,
        mininterval=0.2,
    ):
        if debug:
            print(f"[debug] baseline scan {fp}", file=sys.stderr, flush=True)
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
    mu_fit, fit_info = baseline_gaussian_from_pool(pool)
    sigma_fit = fit_info.get("gaussian_fit_sigma_mv")
    hist_edges_png = fit_info.get("_hist_edges_for_png")

    if baseline_png_path is not None:
        save_baseline_pool_fit_png(
            pool,
            mu_fit,
            _finite_float_or_none(sigma_fit),
            hist_edges_png,
            baseline_png_path,
            run_num=run_num,
            method=str(fit_info.get("method", "")),
            tmin_us=tmin_us,
            tmax_us=tmax_us,
        )

    rows: list[EventCatalogRow] = []
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
        if debug:
            print(f"[debug] catalog k={k} {fp}", file=sys.stderr, flush=True)
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
                    baseline_mv=mu_fit,
                    integral_mv_us=float("nan"),
                    vmax_mv=float("nan"),
                    dt_ggem_nim_ns="",
                )
            )
            continue

        if plot_waveforms and (
            debug_max_waveforms is None or n_debug_plots < debug_max_waveforms
        ):
            assert debug_waveform_dir is not None
            stem = re.sub(r"[^\w\-.]+", "_", fp.name)[:100]
            outp = debug_waveform_dir / f"{k:06d}_{stem}.png"
            title_png = f"{run_num}  {fp.name}  k={k}"
            if _debug_plot_waveform(outp, t_us, v_mv, title=title_png):
                n_debug_plots += 1

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
                    baseline_mv=mu_fit,
                    integral_mv_us=float("nan"),
                    vmax_mv=float("nan"),
                    dt_ggem_nim_ns="",
                )
            )
            continue

        vw_c = np.asarray(vw, dtype=float) - mu_fit
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
                ggem_peak_minus_nim_fall=ggem_peak_minus_nim_fall,
            )
        dt_str = "" if dt_ns is None or not np.isfinite(dt_ns) else f"{float(dt_ns):.12g}"

        rows.append(
            EventCatalogRow(
                run_num=run_num,
                ggem_csv=ggem_name,
                timestamp=ts_str,
                file_index=k,
                spark=int(spark),
                fft_is_noise=fft_lab,
                baseline_mv=mu_fit,
                integral_mv_us=integral_bc,
                vmax_mv=vmax_bc,
                dt_ggem_nim_ns=dt_str,
            )
        )

    bfit: dict[str, Any] = {
        "reference": (
            "Baseline = Gaussian μ from lmfit (bin-integrated normal, total count N = pool size) on a "
            "Freedman–Diaconis histogram of per-event mean voltages outside the analysis window [mV]. "
            "On fit failure or fewer than two histogram bins, falls back to the sample mean. "
            "CSV baseline_mv is μ (or fallback mean). sigma_mV is fit σ when fit_success, else RMSE about μ."
        ),
        "model": fit_info.get("gaussian_fit_model"),
        "method": fit_info.get("method"),
        "mu_mV": mu_fit,
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
        "mu_run_mv": mu_fit,
        "sigma_mle_mv": fit_info.get("gaussian_fit_sigma_mv"),
        "baseline_meta_note": (
            "mu_run_mv matches baseline_gaussian_fit.mu_mV (fit μ or fallback mean). "
            "sigma_mle_mv matches baseline_gaussian_fit.sigma_mV (legacy key name: fit σ when "
            "fit_success, else RMSE about the reported baseline)."
        ),
        "baseline_fit_method": fit_info.get("method"),
        "baseline_fit_png": baseline_png_path.name if baseline_png_path is not None else None,
        "n_baseline_per_event_means": fit_info.get("n_per_event_means", 0),
        "baseline_from_per_event_outside_mean": True,
        "ggem_channel_id": ggem_channel_id,
        "nim_channel_id": nim_channel_id,
        "nim_paths_count": len(nim_paths),
        "edge_mv": edge_mv,
        "nim_width_ns": nim_width_ns,
        "dt_nim_fall_minus_ggem_peak": (not ggem_peak_minus_nim_fall),
        "dt_definition": "t_max_ggem_window_minus_t_fall_nim_ns"
        if ggem_peak_minus_nim_fall
        else "t_fall_nim_minus_t_max_ggem_window_ns",
        "missing_value_encoding": "empty_string_means_nan",
        "n_ggem_files": len(ggem_paths),
        "skipped_read_pool_scan": skipped_read_pool,
        "skipped_read_catalog_scan": skipped_read_catalog,
        "skipped_window": skipped_window,
        "n_spark": n_spark,
        "n_fft_signal": n_fft_signal,
        "n_fft_noise": n_fft_noise,
        "n_fft_unknown_non_spark": n_fft_unknown,
        "cli_debug": debug,
        "debug_max_waveforms": debug_max_waveforms if debug else None,
        "debug_waveform_png_count": n_debug_plots if debug else 0,
        "debug_waveform_dir": str(debug_waveform_dir.resolve())
        if debug and debug_waveform_dir is not None
        else None,
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
        f"[{run_num}] baseline: mu={mu_fit:.6g} mV{sig_part}{src_part}  "
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


def write_catalog(out_csv: Path, out_meta: Path, rows: list[EventCatalogRow], meta: dict[str, Any]) -> None:
    """カタログ CSV とメタ JSON をディスクに書き出す。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CATALOG_CSV_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_dict())
    meta_out = dict(meta)
    meta_out["catalog_csv"] = str(out_csv.name)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)


def read_catalog_rows(path: Path) -> list[dict[str, Any]]:
    """カタログ CSV を読み、各行を列名キーの dict にしたリストで返す。"""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        out: list[dict[str, Any]] = []
        for row in r:
            out.append(dict(row))
    return out


def parse_float_cell(s: str) -> float:
    """カタログの数値セルを ``float`` に（空または nan 表記は ``nan``）。"""
    s = (s or "").strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def parse_fft_noise_cell(s: str) -> bool | None:
    """``fft_is_noise`` 列の ``"0"`` / ``"1"`` / 空を bool または不明 ``None`` に変換する。"""
    s = (s or "").strip()
    if s == "":
        return None
    if s == "0":
        return False
    if s == "1":
        return True
    return None


def parse_timestamp_cell(s: str) -> datetime | None:
    """``timestamp`` 列を ISO 形式として :class:`~datetime.datetime` にパースする。"""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# --- JSON config + CLI -----------------------------------------------------------
_DEFAULT_CATALOG_JSON: dict[str, Any] = {
    "output_root": None,
    "csv_subdir": "csv",
    "ggem_channel_id": 1,
    "nim_channel_id": 2,
    "tmin_us": -0.5,
    "tmax_us": 1.5,
    "spark_threshold_mv": 300.0,
    "spark_min_duration_us": 0.5,
    "noise_dominant_peak_min_mhz": 2.0,
    "rel_tol_dt": 0.02,
    "subtract_mean_fft": True,
    "timestamp_regex": r"(\d{8}_\d{6})",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "edge_mv": -400.0,
    "nim_width_ns": 13.73,
    "dt_nim_fall_minus_ggem_peak": False,
    "baseline_png": True,
    "debug_max_waveforms": 50,
}


def _repo_root() -> Path:
    """このリポジトリの ``analysis/`` ルート（``scripts/`` の親の親）。"""
    return Path(__file__).resolve().parent.parent.parent


def default_catalog_config_path() -> Path:
    """リポジトリ同梱の既定カタログ設定 JSON のパス。"""
    return _repo_root() / "conf" / "event_catalog.default.json"


def load_catalog_build_config(
    path: Path,
    *,
    cli_run_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """カタログ用 JSON を読みデフォルトをマージし、``run_dirs`` / ``output_root`` を絶対パスに解決する。

    ``cli_run_dirs`` が非空のときは、設定ファイルの ``run_dirs`` を置き換える（CLI 優先）。
    """
    base = path.resolve().parent
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg: dict[str, Any] = {**_DEFAULT_CATALOG_JSON, **raw}
    if "dt_nim_fall_minus_ggem_peak" not in raw and "dt_fall_minus_ch1_peak" in raw:
        cfg["dt_nim_fall_minus_ggem_peak"] = bool(raw["dt_fall_minus_ch1_peak"])

    if cli_run_dirs:
        runs: list[Path] = []
        for item in cli_run_dirs:
            p = Path(item).expanduser()
            runs.append(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())
        cfg["run_dirs"] = runs
    elif not cfg.get("run_dirs"):
        raise ValueError(
            "run_dirs is empty: set run_dirs in the config or pass one or more RUN_DIR paths on the command line"
        )
    else:
        runs = []
        for s in cfg["run_dirs"]:
            p = Path(str(s)).expanduser()
            runs.append(p.resolve() if p.is_absolute() else (base / p).resolve())
        cfg["run_dirs"] = runs
    oraw = cfg.get("output_root")
    if oraw is None or str(oraw).strip() == "":
        cfg["output_root"] = _repo_root() / "results"
    else:
        o = Path(str(oraw)).expanduser()
        cfg["output_root"] = o.resolve() if o.is_absolute() else (base / o).resolve()
    return cfg


def main() -> None:
    """JSON 設定に従いイベントカタログを生成する CLI。

    共通約束: 可変長 ``RUN_DIR``、``--conf``、``--debug`` のみ（他は JSON）。詳細は ``cli_common``。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Waveform event catalog from JSON. Pass RUN_DIR paths to override config run_dirs; "
            "omit --conf to use <repo>/conf/event_catalog.default.json."
        ),
    )
    add_standard_cli_arguments(parser)
    args = parser.parse_args()

    if args.conf is not None:
        cfg_path = args.conf.expanduser().resolve()
    else:
        cfg_path = default_catalog_config_path()

    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = load_catalog_build_config(
            cfg_path.resolve(),
            cli_run_dirs=list(args.run_dirs) if args.run_dirs else None,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(1)

    tmin = float(cfg["tmin_us"])
    tmax = float(cfg["tmax_us"])
    if tmin >= tmax:
        print("config: tmin_us must be < tmax_us", file=sys.stderr)
        sys.exit(1)
    if float(cfg["spark_min_duration_us"]) < 0.0:
        print("config: spark_min_duration_us must be >= 0", file=sys.stderr)
        sys.exit(1)
    if float(cfg["noise_dominant_peak_min_mhz"]) < 0.0:
        print("config: noise_dominant_peak_min_mhz must be >= 0", file=sys.stderr)
        sys.exit(1)
    if float(cfg["rel_tol_dt"]) <= 0.0:
        print("config: rel_tol_dt must be > 0", file=sys.stderr)
        sys.exit(1)

    ggem_ch = int(cfg["ggem_channel_id"])
    nim_ch = int(cfg["nim_channel_id"])
    if ggem_ch < 1 or nim_ch < 1:
        print("config: ggem_channel_id and nim_channel_id must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        timestamp_pattern = re.compile(str(cfg["timestamp_regex"]))
    except re.error as exc:
        print(f"config: timestamp_regex invalid: {exc}", file=sys.stderr)
        sys.exit(1)

    run_dirs: list[Path] = cfg["run_dirs"]
    for rd in run_dirs:
        if not rd.is_dir():
            print(f"Not a directory: {rd}", file=sys.stderr)
            sys.exit(1)

    output_root: Path = cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    noise_hz = float(cfg["noise_dominant_peak_min_mhz"]) * 1e6
    subtract_fft = bool(cfg["subtract_mean_fft"])
    ggem_peak_minus_nim_fall = not bool(cfg["dt_nim_fall_minus_ggem_peak"])
    csv_subdir = str(cfg["csv_subdir"])

    used_names: set[str] = set()
    for run_dir in tqdm(run_dirs, desc="runs", unit="run", file=sys.stderr, mininterval=0.2):
        if not discover_channel_roots(run_dir, csv_subdir=csv_subdir):
            print(f"[skip] {run_dir}: no CH*/{csv_subdir}", file=sys.stderr)
            continue
        sub = unique_result_subdir(output_root, run_dir, used_names)
        out_csv = sub / "event_catalog.csv"
        out_meta = sub / "event_catalog.meta.json"
        want_png = bool(cfg["baseline_png"])
        baseline_png = None if not want_png else sub / "event_catalog_baseline.png"
        run_num = run_dir.name
        dbg_dir: Path | None = (sub / "debug_waveforms") if args.debug else None
        if dbg_dir is not None:
            dbg_dir.mkdir(parents=True, exist_ok=True)
        dbg_cap = _catalog_debug_waveform_cap(cfg) if args.debug else None
        rows, meta = process_run_to_rows(
            run_dir,
            run_num=run_num,
            csv_subdir=csv_subdir,
            ggem_channel_id=ggem_ch,
            nim_channel_id=nim_ch,
            tmin_us=tmin,
            tmax_us=tmax,
            spark_threshold_mv=float(cfg["spark_threshold_mv"]),
            spark_min_duration_us=float(cfg["spark_min_duration_us"]),
            noise_dominant_peak_min_hz=noise_hz,
            rel_tol_dt=float(cfg["rel_tol_dt"]),
            subtract_mean_for_fft=subtract_fft,
            timestamp_pattern=timestamp_pattern,
            timestamp_format=str(cfg["timestamp_format"]),
            edge_mv=float(cfg["edge_mv"]),
            nim_width_ns=float(cfg["nim_width_ns"]),
            ggem_peak_minus_nim_fall=ggem_peak_minus_nim_fall,
            baseline_png_path=baseline_png,
            debug=bool(args.debug),
            debug_max_waveforms=dbg_cap,
            debug_waveform_dir=dbg_dir,
        )
        if not rows and meta.get("error") == "no_ggem_paths":
            print(f"[skip] {run_dir}: no GGEM (CH{ggem_ch}) CSV", file=sys.stderr)
            continue
        meta["output_subdir"] = str(sub.relative_to(output_root))
        write_catalog(out_csv, out_meta, rows, meta)
        print(f"[wrote] {out_csv}  ({len(rows)} rows)", flush=True)
        if baseline_png is not None and baseline_png.is_file():
            print(f"[wrote] {baseline_png}", flush=True)
