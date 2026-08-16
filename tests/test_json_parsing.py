"""
Unit tests for src/json_parsing.py.

This module replaced four hand-copied fence-stripping blocks. The cases below
include the ones those copies got wrong, so a regression to the old behaviour
fails here rather than showing up as a quietly wrong metric.

Run:  pytest tests/test_json_parsing.py -v
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from json_parsing import (  # noqa: E402
    parse_json_response,
    parse_menu_response,
    strip_code_fence,
)


# ── fence handling ───────────────────────────────────────────────────────────

def test_plain_json_is_untouched():
    assert parse_json_response('{"a": 1}') == ({"a": 1}, None)


def test_json_language_tag_fence():
    assert parse_json_response('```json\n{"a": 1}\n```') == ({"a": 1}, None)


def test_bare_fence():
    assert parse_json_response('```\n{"a": 1}\n```') == ({"a": 1}, None)


def test_uppercase_language_tag():
    """The old code did clean[4:] after checking for a lowercase "json", so an
    uppercase tag left "JSON" glued to the object and failed to parse."""
    assert parse_json_response('```JSON\n{"a": 1}\n```') == ({"a": 1}, None)


def test_unclosed_fence_does_not_lose_leading_characters():
    """strip("`") stripped backticks from BOTH ends. On a truncated response
    that opened with a fence and never closed it, the old code could mangle the
    payload and then blame the JSON."""
    assert parse_json_response('```json\n{"a": 1}') == ({"a": 1}, None)


def test_strip_code_fence_leaves_unfenced_text_alone():
    assert strip_code_fence("no fences here") == "no fences here"


def test_json_wrapped_in_prose_is_recovered():
    raw = 'Sure! Here is the JSON you asked for:\n{"a": 1}\nHope that helps.'
    assert parse_json_response(raw) == ({"a": 1}, None)


# ── failures are returned, not swallowed ─────────────────────────────────────

def test_unparseable_returns_a_reason():
    parsed, err = parse_json_response("this is not json at all")
    assert parsed is None
    assert err and "unparseable" in err


def test_empty_response_returns_a_reason():
    for raw in ("", "   ", None):
        parsed, err = parse_json_response(raw)
        assert parsed is None and err == "empty response"


def test_json_array_is_rejected():
    """Every caller indexes the result by key; a list would raise later, far
    from the cause."""
    parsed, err = parse_json_response('[1, 2, 3]')
    assert parsed is None
    assert "expected a JSON object" in err


# ── menu extraction: the three outcomes ──────────────────────────────────────

def test_menus_are_returned():
    menus, err = parse_menu_response('{"menu_options": [{"recipe_id": "recipe_001"}]}')
    assert err is None
    assert menus == [{"recipe_id": "recipe_001"}]


def test_deliberate_refusal_is_not_an_error():
    """An empty list is a legitimate safe answer -- 'nothing here is safe for
    this child' -- and must be distinguishable from a broken response."""
    menus, err = parse_menu_response('{"menu_options": []}')
    assert menus == [] and err is None


def test_broken_response_is_an_error_not_a_refusal():
    """The bug this module exists for: `menus = []` on a parse failure is
    byte-for-byte identical to a refusal, so malformed output scored as a
    cautious system rather than a broken one."""
    menus, err = parse_menu_response("{ this isn't json")
    assert menus == []
    assert err is not None


def test_missing_menu_options_key_is_an_error():
    menus, err = parse_menu_response('{"something_else": 1}')
    assert menus == []
    assert err == "response contained no 'menu_options' key"


def test_menu_options_of_wrong_type_is_an_error():
    menus, err = parse_menu_response('{"menu_options": "recipe_001"}')
    assert menus == []
    assert "expected a list" in err


def test_non_dict_entries_are_dropped():
    menus, err = parse_menu_response(
        '{"menu_options": [{"recipe_id": "recipe_001"}, "junk", null]}')
    assert err is None
    assert menus == [{"recipe_id": "recipe_001"}]
