"""Default matplotlib rcParams for saved figures (use with plt.rc_context(MPL_RC))."""

from __future__ import annotations

from typing import Any

MPL_RC: dict[str, Any] = {
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
}
