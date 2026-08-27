"""
bootstrap.py -- Import-path, secrets and console setup. Runs before any `src` import.

The project has no package structure: modules find each other through 26
`sys.path.insert` calls, and `src/graphs/nodes.py` imports its siblings as
top-level names (`from state import ...`). Reproducing exactly the two entries
`src/main.py` puts on the path is therefore not optional, and doing it in one
place keeps the app from growing a 27th variant of the same three lines.

Every Streamlit page starts with `from lunchbite import bootstrap` before it
touches anything in `src`.
"""

from __future__ import annotations

import os
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


def _ensure_secrets_in_env() -> None:
    """
    Copies Streamlit secrets into os.environ, so `src` keeps reading os.getenv.

    Deployed on Streamlit Community Cloud there is no .env -- the repo's is
    gitignored, and rightly, since it holds the Groq key. Configuration arrives
    through st.secrets instead. But every module under src/ reads os.environ
    directly (GROQ_API_KEY, USE_CROSS_ENCODER_RERANKER and eighteen others), and
    teaching all of them about Streamlit would make the CLI and the benchmark
    depend on the dashboard -- the dependency points the other way everywhere
    else in this project, and it should keep pointing that way. Bridging once,
    here, is the whole change deployment needs.

    An existing environment variable always wins, so a local shell or .env still
    overrides, and `service.shadow_warnings` still reports it when it does. That
    last part needs active work, not just a guard -- see below.
    """
    # Snapshot first. Streamlit promotes secrets into os.environ itself as a
    # side effect of parsing the file (runtime/secrets.py), and it does so with
    # a bare assignment -- no check for an existing value. So by the time the
    # loop below could test `key not in os.environ`, a shell variable of the
    # same name has already been overwritten and the original is unrecoverable.
    # Reading it before the first key access is the only place it still exists.
    preexisting = dict(os.environ)

    try:
        import streamlit as st

        items = list(st.secrets.items())  # first key access: parses, and promotes
    except Exception:
        # No secrets file, or not running under Streamlit at all (the CLI and
        # the benchmark import nothing from app/, but a notebook might). Either
        # way there is nothing to bridge and startup must not fail over it.
        return

    for key, value in items:
        if key in preexisting:
            # Undo Streamlit's promotion. The environment was set by whoever
            # launched the process, and this project's rule -- the one
            # service.shadow_warnings() reports against -- is that they win.
            os.environ[key] = preexisting[key]
        elif isinstance(value, (str, int, float, bool)):
            # Nested sections come back as mappings and are skipped; the flat
            # top-level keys are the ones that name environment variables.
            # Streamlit has already promoted the str/int/float ones, so in
            # practice this line is what carries the TOML booleans, which it
            # deliberately excludes. Writing all of them keeps this correct
            # without depending on which types Streamlit decides to promote.
            os.environ[key] = str(value)


_ensure_secrets_in_env()

# Streamlit's server writes this project's output through a pipe that inherits
# the Windows console code page, so the em-dashes and arrows in the pipeline's
# progress output still raise UnicodeEncodeError without this.
try:
    from console import enable_utf8_stdout

    enable_utf8_stdout()
except Exception:  # pragma: no cover - console setup must never block startup
    pass

__all__ = ["ROOT", "SRC", "DATA"]
