"""Shared argparse helpers for analysis CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


STANDARD_CLI_EPILOG = (
    "RUN_DIR paths are required and are not read from JSON; other options come from the config file."
)


def add_standard_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """``--conf`` / ``--debug`` / ``RUN_DIR`` をパーサーに付与する。"""
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
        help="Interactive waveform/PSD viewer (n/p keys) after catalog build.",
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
