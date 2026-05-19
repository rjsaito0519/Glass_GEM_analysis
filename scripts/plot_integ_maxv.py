#!/usr/bin/env python3
"""Thin launcher: ensures ``scripts/`` is on ``sys.path`` when run as a file."""

from __future__ import annotations

import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from modules.catalog_plot_integ_maxv import main

if __name__ == "__main__":
    main()
