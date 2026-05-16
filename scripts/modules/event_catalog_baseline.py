"""Run-wide baseline (histogram + lmfit) and optional PNG."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _finite_float_or_none(x: Any) -> float | None:
    """有限 float に正規化し、不可なら None。"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def _require_lmfit() -> None:
    """lmfit 未導入なら ImportError。"""
    try:
        import lmfit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Baseline histogram Gaussian fit requires lmfit. "
            "Install: pip install 'glassgem-analysis[catalog]'"
        ) from e


def _baseline_histogram_edges_counts(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FD ビンでヒストグラム ``(counts, edges)`` を返す。"""
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


def _fit_gaussian_histogram_lmfit(
    p: np.ndarray,
    counts: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, float | None, bool, str | None, Any]:
    """ヒストに GaussianModel を当てはめ center・sigma を返す。"""
    from lmfit.models import GaussianModel

    x = 0.5 * (edges[:-1] + edges[1:])
    y = np.asarray(counts, dtype=np.float64)
    d_lo = float(np.min(p))
    d_hi = float(np.max(p))
    d_span = max(d_hi - d_lo, 1e-12)

    mod = GaussianModel()
    try:
        params = mod.guess(y, x=x)
    except Exception as exc:
        print(f"[baseline lmfit] GaussianModel.guess failed: {exc}", file=sys.stderr, flush=True)
        return float(np.mean(p)), None, False, "guess_failed", None

    params["center"].set(min=d_lo - d_span, max=d_hi + d_span)
    params["sigma"].set(min=1e-9, max=max(5.0 * d_span, 1e-3))
    if "amplitude" in params:
        params["amplitude"].set(min=0.0)

    try:
        out = mod.fit(y, params, x=x, method="leastsq")
    except Exception as exc:
        print(f"[baseline lmfit] GaussianModel.fit raised: {exc}", file=sys.stderr, flush=True)
        return float(np.mean(p)), None, False, "fit_exception", None

    print(
        "[baseline lmfit] GaussianModel (histogram bin centers), method=leastsq\n" + out.fit_report(),
        file=sys.stderr,
        flush=True,
    )

    center = float(out.params["center"].value)
    sigma = float(out.params["sigma"].value)
    reason: str | None = None
    ok = bool(out.success) and np.isfinite(center) and np.isfinite(sigma) and sigma > 1e-8
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

    return center, sigma, True, None, out


def baseline_gaussian_from_pool(pool_mv: np.ndarray) -> tuple[float, dict[str, Any]]:
    """窓外平均プールから run ベースライン [mV] と fit 情報を返す。"""
    p = np.asarray(pool_mv, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = int(p.size)
    pool_mean = float(np.mean(p)) if n > 0 else 0.0

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
            "gaussian_fit_model": "none",
            "gaussian_fit_center_mv": 0.0,
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
            "gaussian_fit_model": "lmfit_GaussianModel_skipped",
            "gaussian_fit_center_mv": v0,
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

    if n_bins < 5:
        print(
            "[baseline lmfit] skipped: fewer than 5 histogram bins (GaussianModel + guess; using pool mean).",
            file=sys.stderr,
            flush=True,
        )
        info = {
            "method": "histogram_insufficient_bins_fallback_mean",
            "n_per_event_means": n,
            "n_bins": n_bins,
            "hist_range_mv": [hist_lo, hist_hi],
            "fit_success": False,
            "fallback_reason": "histogram_lt5_bins",
            "gaussian_fit_model": "lmfit_GaussianModel_skipped",
            "gaussian_fit_center_mv": pool_mean,
            "gaussian_fit_sigma_mv": None,
            "gaussian_fit_sample_mean_mv": pool_mean,
            "gaussian_fit_sigma_pooled_rmse_mv": None,
            "gaussian_fit_sigma_sample_stdev_mv": None,
            "gaussian_fit_sigma_source": "none",
            "_hist_edges_for_png": edges,
        }
        return pool_mean, info

    center_hat, sigma_hat, ok, fail_reason, lm_out = _fit_gaussian_histogram_lmfit(p, counts, edges)

    if ok:
        method = "gaussian_hist_fit"
        sigma_src = "lmfit_GaussianModel"
        center_out = float(center_hat)
        sigma_out = float(sigma_hat) if sigma_hat is not None else None
    else:
        method = "gaussian_hist_fit_fallback_mean"
        sigma_src = "none"
        center_out = float(center_hat)
        sigma_out = None

    info = {
        "method": method,
        "n_per_event_means": n,
        "n_bins": n_bins,
        "hist_range_mv": [hist_lo, hist_hi],
        "fit_success": ok,
        "fallback_reason": fail_reason if not ok else None,
        "gaussian_fit_model": "lmfit_GaussianModel_histogram",
        "gaussian_fit_center_mv": center_out,
        "gaussian_fit_sigma_mv": sigma_out,
        "gaussian_fit_sample_mean_mv": pool_mean,
        "gaussian_fit_sigma_pooled_rmse_mv": None,
        "gaussian_fit_sigma_sample_stdev_mv": None,
        "gaussian_fit_sigma_source": sigma_src,
        "_hist_edges_for_png": edges,
    }
    if lm_out is not None:
        info["gaussian_fit_lmfit_chisqr"] = getattr(lm_out, "chisqr", None)
        info["gaussian_fit_lmfit_redchi"] = getattr(lm_out, "redchi", None)
        info["gaussian_fit_lmfit_nfev"] = getattr(lm_out, "nfev", None)
    return center_out, info


def save_baseline_pool_fit_png(
    pool_mv: np.ndarray,
    center_fit_mv: float,
    sigma_fit_mv: float | None,
    hist_edges: np.ndarray | None,
    out_path: Path,
    *,
    run_num: str,
    method: str,
    tmin_us: float,
    tmax_us: float,
) -> None:
    """プールヒストとガウス曲線を PNG に保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    from modules.project_matplotlib_rc import MPL_RC

    p = np.asarray(pool_mv, dtype=np.float64)
    p = p[np.isfinite(p)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(MPL_RC):
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
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
            ax.axvline(v0, color="C3", lw=2.0, label=rf"center={v0:.5g} mV (N=1)")
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
            c = float(center_fit_mv)

            ax.hist(
                p,
                bins=edges,
                color="0.82",
                edgecolor="0.45",
                linewidth=0.25,
                label="Per-event mean mV (outside window), counts",
            )

            sigma = sigma_fit_mv
            sigma_rmse = float(np.sqrt(np.mean((p - c) ** 2)))
            if sigma is not None and np.isfinite(sigma) and float(sigma) > 1e-12:
                sigma_3 = float(sigma)
                plot_lo, plot_hi = c - 3.0 * sigma_3, c + 3.0 * sigma_3
                xx = np.linspace(plot_lo, plot_hi, max(200, n_bins * 8))
                yy = n_ev * bin_w * norm.pdf(xx, loc=c, scale=sigma_3)
                ax.plot(
                    xx,
                    yy,
                    color="C3",
                    lw=2.0,
                    label=rf"Gaussian fit (center={c:.5g}, $\sigma$={sigma_3:.5g} mV)",
                )
            elif sigma_rmse > 1e-12:
                sigma_3 = sigma_rmse
                plot_lo, plot_hi = c - 3.0 * sigma_3, c + 3.0 * sigma_3
                xx = np.linspace(plot_lo, plot_hi, max(200, n_bins * 8))
                yy = n_ev * bin_w * norm.pdf(xx, loc=c, scale=sigma_rmse)
                ax.plot(
                    xx,
                    yy,
                    color="C3",
                    lw=2.0,
                    label=rf"RMSE width ($\sigma$={sigma_rmse:.5g} mV)",
                )
            else:
                ps = float(np.std(p)) if p.size > 1 else 0.0
                half = max(3.0 * ps, 0.5)
                plot_lo, plot_hi = c - half, c + half
                ax.axvline(c, color="C3", lw=2.0, label=rf"center={c:.5g} mV")

            ax.set_xlim(plot_lo, plot_hi)
            ax.set_xlabel("Mean voltage [mV]")
            ax.set_ylabel("Counts per bin")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title(
                f"{run_num}  |  Baseline ({method})  |  window [{tmin_us:g}, {tmax_us:g}] µs  "
                f"|  N_events={n_ev}  |  bins={n_bins}"
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
