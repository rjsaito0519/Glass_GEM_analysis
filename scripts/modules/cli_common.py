"""全解析 CLI で共通する引数パターン。

- 末尾の可変長 **RUN_DIR**（1 個以上必須）: 処理対象の run データディレクトリ（conf には書かない）。
- ``--conf PATH``: 既定以外の JSON 設定ファイル。
- ``--debug``: 詳細ログや（設定に従う）波形デバッグ用。それ以外の挙動は JSON のみで渡す。
"""

from __future__ import annotations

import argparse
from pathlib import Path


STANDARD_CLI_EPILOG = (
    "Provide one or more RUN_DIR paths (run data directories); they are not read from the JSON config. "
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
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="One or more run data directories (required; not stored in the JSON config).",
    )
    if not parser.epilog:
        parser.epilog = STANDARD_CLI_EPILOG
