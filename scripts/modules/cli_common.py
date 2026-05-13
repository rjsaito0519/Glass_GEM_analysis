"""全解析 CLI で共通する引数パターン。

- 末尾の可変長 **RUN_DIR**: 1 個以上指定したときは設定の ``run_dirs`` を上書きする。
- ``--conf PATH``: 既定以外の JSON 設定ファイル。
- ``--debug``: 詳細ログや（設定に従う）波形デバッグ用。それ以外の挙動は JSON のみで渡す。
"""

from __future__ import annotations

import argparse
from pathlib import Path


STANDARD_CLI_EPILOG = (
    "When one or more RUN_DIR paths are given, they replace run_dirs from the JSON config. "
    "If none are given, the config must define a non-empty run_dirs list. "
    "All other behavior is controlled via the config file (not extra CLI flags)."
)


def add_standard_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """``--conf`` / ``--debug`` / 可変長 ``RUN_DIR`` を同一パーサーに付与する。"""
    parser.add_argument(
        "--conf",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON config file (each tool defines its own default path when omitted).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Diagnostic mode: verbose per-file messages and optional per-waveform plots "
            "(see debug_max_waveforms in config)."
        ),
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        metavar="RUN_DIR",
        help="Run data directory(ies); optional if run_dirs is set in the config file.",
    )
    if not parser.epilog:
        parser.epilog = STANDARD_CLI_EPILOG
