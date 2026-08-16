"""
json_parsing.py -- One place to turn an LLM's response into a dict.

The fence-strip-then-json.loads block used to be copied verbatim into four
modules, each with the same two defects:

  clean.strip("`")  strips backticks from BOTH ends, so a response that opens
                    with ```json but is truncated before its closing fence loses
                    leading characters and then fails to parse for a second,
                    unrelated reason.
  clean[4:]         assumes the fence language tag is exactly "json" -- not
                    "JSON", not "json5", not a bare fence.

The more serious problem was what callers did with a failure. `generate` in
graphs/nodes.py turned a JSONDecodeError into `menus = []`, which is byte-for-byte
what a correct refusal looks like ("I cannot safely recommend anything"). The
metrics then counted a malformed response as an abstention, so a run where the
model kept returning broken JSON scored as a cautious system rather than a broken
one.

So parse failure is returned, not swallowed: callers get (value, error) and are
expected to record the error.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

# Opening fence with an optional language tag, and the closing fence.
_OPEN_FENCE = re.compile(r"^\s*```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?")
_CLOSE_FENCE = re.compile(r"\r?\n?[ \t]*```\s*$")

# First balanced-looking {...} span, for models that wrap JSON in prose.
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def strip_code_fence(raw: str) -> str:
    """Removes a surrounding markdown fence. Leaves unfenced text untouched."""
    if not raw:
        return ""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    text = _OPEN_FENCE.sub("", text)
    text = _CLOSE_FENCE.sub("", text)
    return text.strip()


def parse_json_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (parsed_object, None) on success, or (None, reason) on failure.

    Two-stage: the fenced text as-is, then the first {...} span within it. The
    fallback is for models that answer "Here is the JSON: {...}" despite being
    told not to -- recoverable, and worth recovering, but only after the strict
    read has been tried.
    """
    if raw is None or not str(raw).strip():
        return None, "empty response"

    text = strip_code_fence(str(raw))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        m = _OBJECT.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None, f"unparseable JSON: {e.msg} (line {e.lineno})"
        else:
            return None, f"unparseable JSON: {e.msg} (line {e.lineno})"

    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"
    return parsed, None


def parse_menu_response(raw: str) -> Tuple[list, Optional[str]]:
    """
    Extracts `menu_options` from a generation response.

    Distinguishes the three outcomes the callers previously collapsed into one:
      ([], None)          the model returned a valid, deliberately empty list --
                          a refusal, which is a legitimate safe answer
      ([], "reason")      the response could not be read at all
      ([...], None)       menus were returned
    """
    parsed, err = parse_json_response(raw)
    if err:
        return [], err
    menus = parsed.get("menu_options")
    if menus is None:
        return [], "response contained no 'menu_options' key"
    if not isinstance(menus, list):
        return [], f"'menu_options' was {type(menus).__name__}, expected a list"
    return [m for m in menus if isinstance(m, dict)], None
