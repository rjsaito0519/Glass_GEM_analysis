"""Write/read event catalog CSV and meta JSON."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.event_catalog_run import EventCatalogRow
from modules.event_catalog_schema import CATALOG_CSV_COLUMNS


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


def read_catalog_rows(path: Path) -> list[dict[str, Any]]:
    """カタログ CSV を行 dict のリストで読む。"""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        return [dict(row) for row in r]


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
