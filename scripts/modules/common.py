"""波形解析用の共通関数（CSV・run レイアウト・窓・スパーク・スペクトル・窓内積分・交差）。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# --- CSV I/O (s, V) -> (us, mV) -------------------------------------------------
DEFAULT_SKIP_HEADER_ROWS = 1
S_TO_US = 1e6
V_TO_MV = 1e3


def read_waveform_csv(
    path: Path | str,
    *,
    skip_header_rows: int = DEFAULT_SKIP_HEADER_ROWS,
    delimiter: str = ",",
    comments: str = "#",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column CSV and return ``(time_us, voltage_mv)``."""
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header_rows,
        comments=comments,
        dtype=float,
        encoding="utf-8",
        invalid_raise=True,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("expected at least 2 columns")
    t_s = data[:, 0].copy()
    v_v = data[:, 1].copy()
    return t_s * S_TO_US, v_v * V_TO_MV


# --- Run ディレクトリレイアウト -------------------------------------------------
_CH_DIR_RE = re.compile(r"^CH(\d+)$", re.IGNORECASE)
_NUM_IN_STEM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def discover_channel_roots(run_dir: Path, *, csv_subdir: str = "csv") -> list[tuple[int, Path]]:
    """``CH*`` ごとの (channel_id, ch_root)。channel_id は 1-based。"""
    found: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _CH_DIR_RE.match(p.name)
        if not m:
            continue
        csv_dir = p / csv_subdir
        if csv_dir.is_dir():
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return found


def sorted_waveform_csvs(csv_dir: Path) -> list[Path]:
    """チャンネル内 ``*.csv`` を stem 先頭の数値でソート。"""
    files = [p for p in csv_dir.glob("*.csv") if p.is_file()]

    def sort_key(fp: Path) -> tuple[int, float, str]:
        hits = _NUM_IN_STEM.findall(fp.stem)
        if hits:
            return (0, float(hits[0]), fp.name)
        return (1, 0.0, fp.name)

    return sorted(files, key=sort_key)


def aligned_csv_paths(
    channel_roots: list[tuple[int, Path]],
    *,
    csv_subdir: str = "csv",
) -> tuple[list[tuple[int, Path, list[Path]]], int]:
    per_ch: list[tuple[int, Path, list[Path]]] = []
    for ch_id, root in channel_roots:
        csv_dir = root / csv_subdir
        paths = sorted_waveform_csvs(csv_dir)
        per_ch.append((ch_id, csv_dir, paths))
    n_events = min((len(paths) for _, _, paths in per_ch), default=0)
    return per_ch, n_events


# --- ピーク ----------------------------------------------------------------------
def time_and_max(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """最大 ``v`` の時刻と値（同値なら先頭のインデックス）。"""
    if t.size == 0 or v.size == 0:
        raise ValueError("empty waveform")
    v = np.asarray(v, dtype=float)
    i = int(np.argmax(v))
    return float(t[i]), float(v[i])


# --- 窓・スパーク ----------------------------------------------------------------
def window_arrays(
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    tmin_us: float,
    tmax_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (t_us >= tmin_us) & (t_us <= tmax_us)
    return t_us[mask], v_mv[mask]


def dt_us(t_us: np.ndarray, *, rel_tol: float) -> float | None:
    """``np.diff(t)`` の平均を代表間隔 [µs] とし、各差分が平均から ``rel_tol`` 以内ならその平均を返す。"""
    if t_us.size < 2:
        return None
    d = np.diff(np.asarray(t_us, dtype=float))
    if d.size == 0:
        return None
    dt_mean = float(np.mean(d))
    if not np.isfinite(dt_mean) or dt_mean <= 0.0:
        return None
    if float(np.max(np.abs(d - dt_mean))) > rel_tol * dt_mean:
        return None
    return dt_mean


def is_spark_event(
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    threshold_mv: float,
    min_duration_us: float,
    rel_tol_dt: float,
) -> bool:
    """``v > threshold`` のサンプル数に :func:`dt_us` の平均間隔 [µs] を掛けた時間が ``min_duration_us`` 以上ならスパーク。"""
    if t_us.size < 2 or v_mv.size < 2:
        return False
    v = np.asarray(v_mv, dtype=float)
    if int(t_us.size) != int(v.size):
        return False
    n_above = int(np.count_nonzero(v > threshold_mv))
    if n_above == 0:
        return False
    sample_dt_us = dt_us(t_us, rel_tol=rel_tol_dt)
    if sample_dt_us is None:
        return False
    return float(n_above) * float(sample_dt_us) >= float(min_duration_us)


# --- スペクトル（Hann + 片側 PSD）------------------------------------------------
def one_sided_psd_m2_per_hz(
    v_mv: np.ndarray,
    *,
    fs_hz: float,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """実信号の片側パワースペクトル密度（入力単位の二乗 / Hz）。

    Hann 窓で端の不連続によるスペクトル漏れを抑え、窓の二乗和でスケールして
    ``|X[k]|^2`` を物理的な PSD に近づける。平均除去は DC およびその近傍への
    リークを抑える用途（``subtract_mean``）。

    ``rfft`` は非負の周波数ビンのみ。片側 PSD では内部ビンに係数 2 を掛け、
    負の周波数側のパワーを折り畳む。DC とナイキスト（偶数長の最終ビン）は
    折り畳みの対がないため係数 1 のまま。
    """
    n = int(v_mv.size)
    if n < 4:
        return None
    w = np.hanning(n)  # 両端を 0 に近づけ、切り出し端のギクシャクを緩和
    x = np.asarray(v_mv, dtype=float).copy()
    if subtract_mean:
        x -= float(np.mean(x))  # 定数オフセット（0 Hz）を弱める
    x *= w
    win_energy = float(np.sum(w * w))  # 窓による振幅減衰の補正に使うエネルギー
    if win_energy <= 0.0 or fs_hz <= 0.0:
        return None
    spec = np.fft.rfft(x)
    # 片側 PSD: 窓付き周期図を fs と窓エネルギーで正規化（単位: (入力)^2/Hz）
    scale = 1.0 / (fs_hz * win_energy)
    psd = np.zeros(spec.shape[0], dtype=float)
    psd[0] = float(np.abs(spec[0]) ** 2) * scale  # DC: 折り畳みなし
    if n % 2 == 0:
        if spec.size > 2:
            psd[1:-1] = 2.0 * (np.abs(spec[1:-1]) ** 2) * scale  # 内部: 負側を足す
        psd[-1] = float(np.abs(spec[-1]) ** 2) * scale  # ナイキスト: 折り畳みなし
    else:
        psd[1:] = 2.0 * (np.abs(spec[1:]) ** 2) * scale  # 奇数長: 最終ビンはナイキスト外
    freq_hz = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freq_hz, psd


def dominant_peak_freq_hz_excluding_dc(freq_hz: np.ndarray, psd: np.ndarray) -> float | None:
    if freq_hz.size < 2 or psd.size < 2:
        return None
    sub = psd[1:]
    if sub.size == 0:
        return None
    k_rel = int(np.argmax(sub))
    k = k_rel + 1
    return float(freq_hz[k])


def fft_is_noise_label(
    tw: np.ndarray,
    vw: np.ndarray,
    *,
    noise_dominant_peak_min_hz: float,
    rel_tol_dt: float,
    subtract_mean_for_fft: bool,
) -> bool | None:
    sample_dt_us = dt_us(tw, rel_tol=rel_tol_dt)
    if sample_dt_us is None or tw.size < 4:
        return None
    fs_hz = 1.0 / (sample_dt_us * 1e-6)
    psd_result = one_sided_psd_m2_per_hz(vw, fs_hz=fs_hz, subtract_mean=subtract_mean_for_fft)
    if psd_result is None:
        return None
    freq_hz, psd = psd_result
    f_dom = dominant_peak_freq_hz_excluding_dc(freq_hz, psd)
    if f_dom is None:
        return None
    return bool(f_dom >= noise_dominant_peak_min_hz)


# --- しきい値交差・整列 ----------------------------------------------------------
def _interp_cross_time(t: np.ndarray, v: np.ndarray, i: int, v_th: float) -> float:
    t0, t1 = float(t[i]), float(t[i + 1])
    v0, v1 = float(v[i]), float(v[i + 1])
    if v1 == v0:
        return 0.5 * (t0 + t1)
    frac = (v_th - v0) / (v1 - v0)
    frac = max(0.0, min(1.0, frac))
    return t0 + frac * (t1 - t0)


def _first_falling_crossing_us(t: np.ndarray, v: np.ndarray, v_th: float) -> float | None:
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    for i in range(v.size - 1):
        if v[i] > v_th and v[i + 1] <= v_th:
            return _interp_cross_time(t, v, i, v_th)
    return None


def _first_rising_crossing_us(t: np.ndarray, v: np.ndarray, v_th: float) -> float | None:
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    for i in range(v.size - 1):
        if v[i] < v_th and v[i + 1] >= v_th:
            return _interp_cross_time(t, v, i, v_th)
    return None


def inferred_fall_time_us(
    t_us: np.ndarray,
    v_mv: np.ndarray,
    *,
    v_th: float,
    nim_width_ns: float,
) -> float | None:
    """``v_th`` をまたぐ最初の立ち下がり時刻 [µs]。無い場合は立ち上がり + ``nim_width_ns`` で推定。"""
    t_f = _first_falling_crossing_us(t_us, v_mv, v_th)
    if t_f is not None:
        return t_f
    t_r = _first_rising_crossing_us(t_us, v_mv, v_th)
    if t_r is None:
        return None
    return float(t_r + nim_width_ns * 1e-3)


def max_v_time_us(tw: np.ndarray, vw: np.ndarray) -> float | None:
    """``vw`` が最大となるサンプルにおける ``tw`` の時刻 [µs]。"""
    if tw.size == 0 or vw.size == 0:
        return None
    v = np.asarray(vw, dtype=float)
    i = int(np.argmax(v))
    return float(tw[i])


def trapz_y_x(y: np.ndarray, x: np.ndarray) -> float:
    """台形則による ``y`` の ``x`` に関する積分。同長1次元（積の単位は呼び出し側の物理に従う）。

    ``numpy.trapz`` を使用（等間隔・非等間隔の両方でそのまま適用可能）。
    """
    return float(np.trapz(y, x))


def align_pair(
    t1: np.ndarray,
    v1: np.ndarray,
    t2: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    n = min(int(t1.size), int(t2.size), int(v1.size), int(v2.size))
    if n < 2:
        return None
    return t1[:n], v1[:n], t2[:n], v2[:n]
