"""Waveform event catalog (one row per GGEM CSV): run-wide baseline, spark, features, GGEM–NIM timing.

Run-wide **baseline** is the ``lmfit.models.GaussianModel`` **center** fit to a **Freedman–Diaconis**
histogram of per-event mean voltages [mV] **outside** the analysis window (bin centers vs counts; ``guess()`` for
initial parameters). On failure or too few bins, the code falls back to the sample mean. Extras:
``pip install -e ".[catalog,viz]"`` for ``lmfit`` and baseline PNG plots.

**CLI（リポジトリ内の解析スクリプト共通）**: 末尾に可変長 ``RUN_DIR``（1 個以上必須）、``--conf``、``--debug``。
run のパスは JSON には含めない。デバッグ時の波形 PNG 枚数上限は ``debug_max_waveforms`` を JSON で指定。

Each row adds spark flag, optional **FFT-based** noise/signal label, integral / vmax, and Δt vs NIM.

GGEM = main GEM waveform channel; NIM = NIM logic channel for edge timing. No extra channel gate
beyond spark for the FFT label path.

実装は ``event_catalog_schema`` / ``event_catalog_baseline`` / ``event_catalog_run`` / ``event_catalog_io`` に分割。
このモジュールは CLI・設定マージと、後方互換の再エクスポートを提供する。
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

from modules.cli_common import add_standard_cli_arguments
from modules.common import discover_channel_roots
from modules.event_catalog_baseline import baseline_gaussian_from_pool
from modules.event_catalog_io import (
    parse_float_cell,
    parse_fft_noise_cell,
    parse_timestamp_cell,
    read_catalog_rows,
    write_catalog,
)
from modules.event_catalog_run import (
    EventCatalogRow,
    _catalog_debug_waveform_cap,
    process_run_to_rows,
    unique_result_subdir,
)
from modules.event_catalog_schema import CATALOG_CSV_COLUMNS

__all__ = [
    "CATALOG_CSV_COLUMNS",
    "EventCatalogRow",
    "_catalog_debug_waveform_cap",
    "baseline_gaussian_from_pool",
    "default_catalog_config_path",
    "load_catalog_build_config",
    "main",
    "parse_float_cell",
    "parse_fft_noise_cell",
    "parse_timestamp_cell",
    "process_run_to_rows",
    "read_catalog_rows",
    "unique_result_subdir",
    "write_catalog",
]

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
    cli_run_dirs: Sequence[Path],
) -> dict[str, Any]:
    """カタログ用 JSON を読みデフォルトをマージし、``run_dirs``（CLI のみ）と ``output_root`` を絶対パスに解決する。

    ``run_dirs`` は JSON には含めない。``cli_run_dirs`` を 1 個以上渡す（カレントディレクトリ基準で相対パス可）。
    旧 JSON に ``run_dirs`` が残っていても無視して削除する。
    """
    base = path.resolve().parent
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg: dict[str, Any] = {**_DEFAULT_CATALOG_JSON, **raw}
    cfg.pop("run_dirs", None)
    if "dt_nim_fall_minus_ggem_peak" not in raw and "dt_fall_minus_ch1_peak" in raw:
        cfg["dt_nim_fall_minus_ggem_peak"] = bool(raw["dt_fall_minus_ch1_peak"])

    if not cli_run_dirs:
        raise ValueError("pass one or more RUN_DIR paths on the command line (run_dirs are not read from JSON)")
    runs: list[Path] = []
    for item in cli_run_dirs:
        p = Path(item).expanduser()
        runs.append(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())
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
            "Waveform event catalog from JSON. Requires one or more RUN_DIR paths; "
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
            cli_run_dirs=list(args.run_dirs),
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
