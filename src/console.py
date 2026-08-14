"""
console.py -- Console encoding setup.

Windows consoles default to a legacy code page (cp1252 here), which raises
UnicodeEncodeError on the arrows, dashes, and box-drawing characters used in
this project's progress output. That turns a cosmetic character into a crash
partway through a run.

Call enable_utf8_stdout() at the top of every entry point.
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr, degrading to '?' rather than raising."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream is detached or redirected somewhere that cannot be
            # reconfigured; printing plainly is still better than crashing.
            pass
