"""
Unit tests for the per-model Groq client configuration in src/llm_provider.py.

Two things are under test, both of which have a wrong-by-default failure mode:

  reasoning_effort  Groq's reasoning models do not share a vocabulary. gpt-oss
                    takes low|medium|high and rejects "none"; qwen3 takes
                    none|default and rejects "low"; a non-reasoning model
                    rejects the parameter outright. Sending the wrong one is a
                    400 on *every* call, so the value is derived from the model
                    id rather than from the role the model is playing -- a
                    role-keyed default would 400 whatever a user passed to
                    `--model`.

  max_tokens        Groq bills the *reserved* budget, not the completion, so the
                    judge's ceiling decides how many judge calls a day's quota
                    buys. It must not silently inherit the generator's 2000.

No network: ChatGroq is replaced by a recorder that captures the kwargs.

Run:  pytest tests/test_llm_provider_config.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import llm_provider  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

_ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_JUDGE_MODEL",
    "GROQ_REASONING_EFFORT",
    "GROQ_JUDGE_REASONING_EFFORT",
    "JUDGE_MAX_TOKENS",
    "LLM_MAX_TOKENS",
    "JUDGE_TEMPERATURE",
    "LLM_TEMPERATURE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """A real .env is loaded at import time; drop it so defaults are visible."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    return monkeypatch


@pytest.fixture
def captured(monkeypatch):
    """Captures the kwargs _try_groq would hand to ChatGroq."""
    calls = []

    class FakeChatGroq:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_module = type(sys)("langchain_groq")
    fake_module.ChatGroq = FakeChatGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", fake_module)
    return calls


# ── reasoning_effort is chosen by model, not by role ─────────────────────────

def test_gpt_oss_judge_gets_low_not_none(clean_env, captured):
    """"none" is not in gpt-oss's vocabulary; sending it 400s every call."""
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert captured[0]["reasoning_effort"] == "low"


def test_qwen_generator_gets_reasoning_off(clean_env, captured):
    """Thinking tokens would come out of the menus' own token budget."""
    llm_provider._try_groq("qwen/qwen3.6-27b", role="generator")
    assert captured[0]["reasoning_effort"] == "none"


def test_the_same_model_is_configured_the_same_way_in_either_role():
    """
    The regression this file exists for: the setting belongs to the model.
    A role-keyed default would send "none" to gpt-oss whenever it generates,
    and "low" to qwen whenever it judges. Both are 400s.
    """
    for model, expected in (("openai/gpt-oss-120b", "low"), ("qwen/qwen3.6-27b", "none")):
        assert llm_provider._default_reasoning_effort(model) == expected


def test_unknown_model_is_sent_no_reasoning_parameter(clean_env, captured):
    """
    `run_all.py --model <anything>` must not become a 400. A model that does
    not reason rejects reasoning_effort outright, so silence is the only safe
    default for an id we do not recognise.
    """
    llm_provider._try_groq("some-other-provider/plain-model-7b", role="generator")
    assert "reasoning_effort" not in captured[0]


def test_family_is_matched_on_the_id_not_the_full_name(clean_env):
    """Ids carry a vendor prefix and a size suffix; match the family inside."""
    assert llm_provider._default_reasoning_effort("openai/gpt-oss-20b") == "low"
    assert llm_provider._default_reasoning_effort("QWEN/QWEN3.6-27B") == "none"


# ── the env override, including the empty case ───────────────────────────────

def test_env_overrides_the_derived_default(clean_env, captured):
    clean_env.setenv("GROQ_JUDGE_REASONING_EFFORT", "high")
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert captured[0]["reasoning_effort"] == "high"


def test_empty_env_value_suppresses_the_parameter(clean_env, captured):
    """
    The escape hatch for a model that rejects the parameter: an empty value
    means "send nothing", not "fall back to the derived default".
    """
    clean_env.setenv("GROQ_REASONING_EFFORT", "")
    llm_provider._try_groq("qwen/qwen3.6-27b", role="generator")
    assert "reasoning_effort" not in captured[0]


def test_generator_and_judge_overrides_do_not_leak_into_each_other(clean_env, captured):
    clean_env.setenv("GROQ_REASONING_EFFORT", "default")
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert captured[0]["reasoning_effort"] == "low"


# ── token budgets ────────────────────────────────────────────────────────────

def test_judge_does_not_inherit_the_generator_budget(clean_env, captured):
    """
    Groq bills the reservation. The judge sharing the generator's budget is what
    once exhausted the daily cap after 42 of ~321 calls and left the reported
    means resting on the survivors.
    """
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    llm_provider._try_groq("qwen/qwen3.6-27b", role="generator")
    judge_kwargs, generator_kwargs = captured
    assert judge_kwargs["max_tokens"] == 450
    assert generator_kwargs["max_tokens"] == 1200


def test_a_full_run_fits_the_daily_token_cap(clean_env, captured):
    """
    The reservation is not a style choice, it is what decides whether a run
    finishes. 30 cases x 3 LLM arms, billed at (measured median prompt 562 +
    the reservation), has to stay inside Groq's 200k tokens/day for the
    generator. At the old 2000 this sum is ~231k: the run dies around case 26
    and the metrics describe whatever completed first.
    """
    llm_provider._try_groq("qwen/qwen3.6-27b", role="generator")
    reserved = captured[0]["max_tokens"]
    assert 90 * (562 + reserved) < 200_000


def test_judge_does_not_sample(clean_env, captured):
    """A scorer that samples is not reproducible between runs."""
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert captured[0]["temperature"] == 0.0


def test_malformed_budget_falls_back_instead_of_crashing_the_run(clean_env, captured):
    clean_env.setenv("JUDGE_MAX_TOKENS", "not-a-number")
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert captured[0]["max_tokens"] == 450


# ── the models the project actually ships with ───────────────────────────────

def test_shipped_defaults_are_two_different_families(clean_env, captured):
    """
    The judge exists to score without self-preferencing bias, which requires an
    independent lineage -- not merely a different size. Read through get_llm /
    get_judge_llm so this asserts on the defaults the project actually ships,
    not on names repeated here: if those two ever converge on one family, the
    claim the report makes about the judge becomes false.
    """
    _, generator_name = llm_provider.get_llm(prefer="groq")
    _, judge_name = llm_provider.get_judge_llm(prefer="groq")
    generator = captured[0]["model"]
    judge = captured[1]["model"]
    assert generator.split("/")[0] != judge.split("/")[0], (
        f"generator and judge share a family: {generator_name}, {judge_name}"
    )


def test_no_reasoning_format_is_ever_sent(clean_env, captured):
    """
    gpt-oss rejects reasoning_format entirely (it uses include_reasoning), and
    the two parameters are mutually exclusive on the models that do take it.
    """
    llm_provider._try_groq("openai/gpt-oss-120b", role="judge")
    assert "reasoning_format" not in captured[0]


# ── --provider must reach the judge, not only the generator ──────────────────
#
# `--provider ollama` used to apply to the generator alone. The evaluator called
# get_judge_llm() with no preference, and that resolves to Groq whenever
# GROQ_API_KEY is set, so an explicitly local run still sent every judge call to
# the cloud. Silent when the key was good; empty judge scores when it was not.

def test_judge_honours_an_explicit_provider(clean_env, monkeypatch):
    """A Groq key is set, yet prefer='ollama' must not resolve to Groq."""
    import llm_provider

    monkeypatch.setattr(llm_provider, "_ollama_reachable", lambda: True)
    _, name = llm_provider.get_judge_llm(prefer="ollama")
    assert name.startswith("ollama/"), f"judge ignored the preference: {name}"


def test_evaluator_threads_the_provider_to_the_judge(monkeypatch):
    """The wiring, not just the resolver: evaluate() must pass it down."""
    sys.path.insert(0, os.path.join(ROOT, "benchmark"))
    import evaluator

    evaluator.set_judge_provider("ollama")
    assert evaluator._judge_cache.get("prefer") == "ollama"

    # Switching preference must drop the cached client, or a run would keep
    # using whichever provider happened to be resolved first.
    evaluator._judge_cache["llm"] = object()
    evaluator.set_judge_provider("groq")
    assert "llm" not in evaluator._judge_cache
    evaluator.set_judge_provider(None)


def test_ollama_client_is_json_constrained_and_has_a_timeout(monkeypatch):
    """
    Both roles ask for JSON and parse it. A 3B local model asked for JSON
    unaided returned Python source code that *generates* it — 26 parse failures
    across 16 cases in run 20260820_100542, every error in that run.

    The timeout matters as much: ChatOllama takes none by default, so a local
    call had no deadline at all.
    """
    import llm_provider

    monkeypatch.setattr(llm_provider, "_ollama_reachable", lambda: True)
    for role in ("generator", "judge"):
        llm = llm_provider._try_ollama("llama3.2", role)
        assert getattr(llm, "format", None) == "json", f"{role} not JSON-constrained"
        timeout = (getattr(llm, "client_kwargs", {}) or {}).get("timeout")
        assert timeout and timeout > 0, f"{role} has no request timeout"


def test_json_mode_can_be_disabled_for_debugging(monkeypatch):
    import llm_provider

    monkeypatch.setattr(llm_provider, "_ollama_reachable", lambda: True)
    monkeypatch.setenv("OLLAMA_JSON_MODE", "false")
    llm = llm_provider._try_ollama("llama3.2", "generator")
    assert getattr(llm, "format", None) != "json"
