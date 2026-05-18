"""Merge multiple decoded event catalog CSV files into one."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from modules.event_catalog_io import (
    CATALOG_CSV_NAME,
    CATALOG_META_NAME,
    catalog_headers_ok,
    resolve_catalog_paths,
    write_merged_catalog_rows,
)


def _analysis_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_output_root() -> Path:
    return _analysis_root() / "results"


def sanitize_output_name(name: str) -> str:
    base = re.sub(r"[^\w\-.]+", "_", name.strip())[:80]
    return base or "merged"


def unique_merge_subdir(output_root: Path, requested_name: str) -> Path:
    """``results/<NAME>/`` を確保する（既存ディレクトリなら別名）。"""
    base = sanitize_output_name(requested_name)
    cand = base
    i = 0
    h = abs(hash(requested_name)) % 1_000_000
    while (output_root / cand).exists():
        i += 1
        cand = f"{base}_{h:x}_{i}"
    return output_root / cand


def merge_catalog_rows(sources: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """入力 CSV を結合し ``(rows, source_strs)`` を返す。"""
    merged: list[dict[str, Any]] = []
    source_strs: list[str] = []
    for path in sources:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not catalog_headers_ok(reader.fieldnames):
                print(f"[skip] not a catalog CSV: {path}", file=sys.stderr, flush=True)
                continue
            rows = [dict(row) for row in reader]
        if not rows:
            print(f"[skip] empty catalog: {path}", file=sys.stderr, flush=True)
            continue
        merged.extend(rows)
        source_strs.append(str(path))
    return merged, source_strs


def merge_catalogs(
    paths: Sequence[Path],
    *,
    output_name: str,
    output_root: Path | None = None,
) -> tuple[Path, Path, int]:
    """カタログを結合して ``results/<name>/`` に書き出す。"""
    catalog_paths = resolve_catalog_paths(paths)
    if not catalog_paths:
        raise ValueError("no catalog CSV paths resolved")

    merged, source_strs = merge_catalog_rows(catalog_paths)
    if not merged:
        raise ValueError("no rows read from input catalogs")

    root = default_output_root() if output_root is None else output_root
    root.mkdir(parents=True, exist_ok=True)
    sub = unique_merge_subdir(root, output_name)
    out_csv = sub / CATALOG_CSV_NAME
    out_meta = sub / CATALOG_META_NAME

    write_merged_catalog_rows(out_csv, merged)

    run_nums = sorted({(r.get("run_num") or "").strip() for r in merged if (r.get("run_num") or "").strip()})
    meta: dict[str, Any] = {
        "merged_from": source_strs,
        "n_rows": len(merged),
        "run_nums": run_nums,
        "output_subdir": sub.name,
        "catalog_csv": CATALOG_CSV_NAME,
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_csv, out_meta, len(merged)


def main() -> None:
    """CLI: ``PATH ... --out NAME``。"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event_catalog.csv or directory containing it",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="NAME",
        help="Subdirectory name under analysis/results/ (event_catalog.csv written inside)",
    )
    args = parser.parse_args()

    try:
        out_csv, out_meta, n_rows = merge_catalogs(args.paths, output_name=args.out)
    except ValueError as exc:
        print(f"[merge_event_catalog] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[wrote] {out_csv}  ({n_rows} rows)", flush=True)
    print(f"[wrote] {out_meta}", flush=True)


if __name__ == "__main__":
    main()
