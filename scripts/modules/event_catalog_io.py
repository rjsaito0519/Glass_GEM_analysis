"""Write/read event catalog CSV and meta JSON."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.event_catalog_run import EventCatalogRow
from modules.event_catalog_schema import CATALOG_CSV_COLUMNS

CATALOG_CSV_NAME = "event_catalog.csv"
CATALOG_META_NAME = "event_catalog.meta.json"


def write_catalog(out_csv: Path, out_meta: Path, rows: list[EventCatalogRow], meta: dict[str, Any]) -> None:
    """カタログ CSV と meta JSON を書き出す。"""
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


def resolve_catalog_paths(paths: Sequence[Path]) -> list[Path]:
    """``event_catalog.csv`` または ``<dir>/event_catalog.csv`` を解決する。"""
    out: list[Path] = []
    for raw in paths:
        p = raw.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".csv":
            out.append(p)
            continue
        if p.is_dir():
            cand = p / CATALOG_CSV_NAME
            if cand.is_file():
                out.append(cand)
            else:
                print(f"[skip] no {CATALOG_CSV_NAME} under {p}", file=sys.stderr, flush=True)
            continue
        print(f"[skip] not a csv file or directory: {p}", file=sys.stderr, flush=True)
    return out


def catalog_headers_ok(fieldnames: Sequence[str] | None) -> bool:
    """カタログ CSV として最低限必要な列があるか。"""
    if not fieldnames:
        return False
    cols = set(fieldnames)
    required = {"run_num", "ggem_csv", "integral_mv_us", "vmax_mv"}
    return required.issubset(cols)


def read_catalog_rows(path: Path) -> list[dict[str, Any]]:
    """カタログ CSV を行 dict のリストで読む。"""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        return [dict(row) for row in r]


def write_merged_catalog_rows(out_csv: Path, rows: list[dict[str, Any]]) -> None:
    """結合カタログ CSV を書き出す（行は列名 dict）。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CATALOG_CSV_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CATALOG_CSV_COLUMNS})


def parse_float_cell(s: str) -> float:
    """数値セルを float に（空は nan）。"""
    s = (s or "").strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def parse_fft_noise_cell(s: str) -> bool | None:
    """``fft_is_noise`` 列を bool / None に変換する。"""
    s = (s or "").strip()
    if s == "":
        return None
    if s == "0":
        return False
    if s == "1":
        return True
    return None


def parse_timestamp_cell(s: str) -> datetime | None:
    """``timestamp`` 列を datetime にパースする。"""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
