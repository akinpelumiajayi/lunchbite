"""
bootstrap.py -- Import-path and console setup. Must run before any `src` import.

The project has no package structure: modules find each other through 26
`sys.path.insert` calls, and `src/graphs/nodes.py` imports its siblings as
top-level names (`from state import ...`). Reproducing exactly the two entries
`src/main.py` puts on the path is therefore not optional, and doing it in one
place keeps the app from growing a 27th variant of the same three lines.

Every Streamlit page starts with `from lunchbite import bootstrap` before it
touches anything in `src`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# app/lunchbite/bootstrap.py -> app/lunchbite -> app -> repo root
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"


def _ensure_paths() -> None:
    # Mirrors src/main.py: `src` first, then `src/graphs`, because the graph
    # modules import each other unqualified.
    for entry in (SRC, SRC / "graphs"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_paths()

# Streamlit's server writes this project's output through a pipe that inherits
# the Windows console code page, so the em-dashes and arrows in the pipeline's
# progress output still raise UnicodeEncodeError without this.
try:
    from console import enable_utf8_stdout

    enable_utf8_stdout()
except Exception:  # pragma: no cover - console setup must never block startup
    pass

__all__ = ["ROOT", "SRC", "DATA"]
