"""
Tests for finishing a run on a different generator than it started on.

The 2026-08-21 run spent Groq's 200,000 tokens/day at ADV-01 of 36 and stopped
with 15 case-runs on disk. Ollama was running locally and has no token budget, so
the cheapest way to finish was to resume on it — 22 case-runs for no cloud spend.

That is a legitimate thing to do and a dangerous thing to do quietly. Before this,
`metadata.model` was one string written from whichever provider the resuming
process happened to resolve, so a file holding 14 qwen3.6-27b case-runs and 22
llama3.2 ones came out stamped `ollama/llama3.2` — and the report printed that as
the run's model while asserting the arms shared "an identical LLM backbone".

The properties under test:
  1. every case-run records the model that produced it;
  2. a resume that would change generator is refused unless it is asked for;
  3. when it is asked for, the file names every generator in it and the report
     says so rather than claiming one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "report"))


def _stub_graphs(monkeypatch, provider_name="ollama/llama3.2"):
    """Four graphs that answer instantly, and a named provider to attribute to."""
    import llm_provider
    from graphs import build_graphs

    monkeypatch.setattr(llm_provider, "get_llm",
                        lambda prefer=None: (MagicMock(), provider_name))
    monkeypatch.setattr(llm_provider, "configure_langsmith", lambda *a, **k: False)

    def make(_llm=None):
        g = MagicMock()
        g.invoke.side_effect = lambda state: {"final_menus": [], "latency_ms": {}}
        return g

    # Driven off build_graphs.BUILDER_NAMES rather than a literal list, so a
    # new arm is stubbed here automatically instead of reaching a real model.
    for name in build_graphs.BUILDER_NAMES.values():
        monkeypatch.setattr(build_graphs, name, make)


def _partial_file(tmp_path, model, n=3, name="run_partial.json"):
    """A results file from an earlier attempt, stamped with the model that made it."""
    from benchmark_cases import BENCHMARK_CASES

    rows = [
        {"case_id": c.case_id, "repeat": 0, "category": c.category,
         "description": c.description,
         "neural_rag": {"final_menus": []}, "neurosymbolic": {"final_menus": []},
         "no_rag": {"final_menus": []}}
        for c in BENCHMARK_CASES[:n]
    ]
    p = tmp_path / name
    p.write_text(json.dumps({"metadata": {"model": model}, "results": rows}),
                 encoding="utf-8")
    return p


# ── provenance ───────────────────────────────────────────────────────────────

def test_every_case_run_records_the_model_that_produced_it(monkeypatch, tmp_path):
    import runner

    _stub_graphs(monkeypatch, "ollama/llama3.2")
    out = runner.run_benchmark(output_dir=str(tmp_path), limit=2)

    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert [r["generator"] for r in payload["results"]] == ["ollama/llama3.2"] * 2
    assert payload["metadata"]["generators"] == {"ollama/llama3.2": 2}


def test_rows_from_an_older_file_inherit_that_file_s_model(tmp_path):
    """
    A resumed file names one model in its metadata and none on its rows. That
    name is the provenance of every row in it, so it is copied down rather than
    left blank — otherwise resuming loses the history of the half already paid for.
    """
    import runner

    partial = _partial_file(tmp_path, "groq/qwen/qwen3.6-27b", n=3)
    kept = runner._load_completed(str(partial))

    assert len(kept) == 3
    assert {r["generator"] for r in kept} == {"groq/qwen/qwen3.6-27b"}


# ── the guard ────────────────────────────────────────────────────────────────

def test_resuming_on_a_different_generator_is_refused_by_default(monkeypatch, tmp_path):
    """
    Silently allowed, this produces a file whose single `model` field is false for
    most of its rows. The run is stopped so the mix is a decision, not a default.
    """
    import runner

    partial = _partial_file(tmp_path, "groq/qwen/qwen3.6-27b", n=3)
    _stub_graphs(monkeypatch, "ollama/llama3.2")

    with pytest.raises(SystemExit) as excinfo:
        runner.run_benchmark(output_dir=str(tmp_path), resume_from=str(partial), limit=5)

    msg = str(excinfo.value)
    assert "RESUME REFUSED" in msg
    assert "groq/qwen/qwen3.6-27b" in msg and "ollama/llama3.2" in msg
    assert "--allow-provider-change" in msg


def test_resuming_on_the_same_generator_is_not_refused(monkeypatch, tmp_path):
    """The guard must not fire on the ordinary case: same model, budget refilled."""
    import runner

    partial = _partial_file(tmp_path, "groq/qwen/qwen3.6-27b", n=3)
    _stub_graphs(monkeypatch, "groq/qwen/qwen3.6-27b")

    out = runner.run_benchmark(output_dir=str(tmp_path), resume_from=str(partial), limit=5)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["metadata"]["generators"] == {"groq/qwen/qwen3.6-27b": 5}
    assert payload["metadata"]["model"] == "groq/qwen/qwen3.6-27b"


def test_a_file_predating_provenance_is_not_treated_as_a_provider_change(monkeypatch, tmp_path):
    """
    An old file records no generator at all. That is missing evidence, not proof
    of a different model, and refusing on it would block every resume of a file
    written before this feature — including a resume onto the very same model.
    """
    import runner

    p = tmp_path / "old.json"
    p.write_text(json.dumps({"metadata": {}, "results": [
        {"case_id": "STD-01", "repeat": 0, "neural_rag": {"final_menus": []},
         "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}}]}),
        encoding="utf-8")
    _stub_graphs(monkeypatch, "ollama/llama3.2")

    out = runner.run_benchmark(output_dir=str(tmp_path), resume_from=str(p), limit=3)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    # Not refused — but the gap is still recorded rather than papered over.
    assert payload["metadata"]["generators"] == {"unknown": 1, "ollama/llama3.2": 2}


# ── what the mixed file says about itself ────────────────────────────────────

def test_an_allowed_mix_names_every_generator_in_the_metadata(monkeypatch, tmp_path):
    import runner

    partial = _partial_file(tmp_path, "groq/qwen/qwen3.6-27b", n=3)
    _stub_graphs(monkeypatch, "ollama/llama3.2")

    out = runner.run_benchmark(output_dir=str(tmp_path), resume_from=str(partial),
                               limit=5, allow_provider_change=True)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    meta = payload["metadata"]

    assert meta["generators"] == {"groq/qwen/qwen3.6-27b": 3, "ollama/llama3.2": 2}
    # The single `model` string stays a string for existing readers, but it can
    # no longer name one model for a file produced by two.
    assert meta["model"].startswith("mixed:")
    assert "groq/qwen/qwen3.6-27b (3)" in meta["model"]
    assert "ollama/llama3.2 (2)" in meta["model"]


def _render(tmp_path, meta, results):
    """Render a report from a results file, the way generate_report is really called."""
    from generate_report import generate_report

    base = {"pipelines": ["no_llm", "neural_rag", "neurosymbolic", "no_rag"],
            "adversarial_injection_applied_to": ["neural_rag", "neurosymbolic", "no_rag"]}
    base.update(meta)
    run = tmp_path / "run_mixed.json"
    run.write_text(json.dumps({"metadata": base, "results": results}), encoding="utf-8")
    out = tmp_path / "REPORT.md"
    generate_report(str(run), out_path=str(out))
    return out.read_text(encoding="utf-8")


def _case(cid, cat, gen):
    row = {"case_id": cid, "description": "d", "category": cat, "generator": gen,
           "expected_unsafe_ids": [], "profile": {"age": 7, "allergies": ["milk"]}}
    for mode in ("no_llm", "neural_rag", "neurosymbolic", "no_rag"):
        row[mode] = {"final_menus": []}
    return row


def test_the_report_discloses_a_mixed_run_instead_of_claiming_one_backbone(tmp_path):
    """
    The claim being guarded: "using an identical LLM backbone ... across arms".
    True within a case in every run; false across the suite in a mixed one.
    """
    md = _render(
        tmp_path,
        {"model": "mixed: groq/qwen3.6-27b (2), ollama/llama3.2 (2)",
         "generators": {"groq/qwen3.6-27b": 2, "ollama/llama3.2": 2}},
        [_case("STD-01", "standard", "groq/qwen3.6-27b"),
         _case("STD-02", "standard", "groq/qwen3.6-27b"),
         _case("ADV-01", "adversarial", "ollama/llama3.2"),
         _case("ADV-02", "adversarial", "ollama/llama3.2")],
    )

    assert "TWO GENERATORS IN THIS RUN" in md
    # and it says WHICH cases came from which, since the split follows category
    assert "`groq/qwen3.6-27b` — 2 case-runs: standard 2" in md
    assert "`ollama/llama3.2` — 2 case-runs: adversarial 2" in md
    # the unqualified claim is gone, and the qualified one has replaced it
    assert "identical LLM backbone, retrieval stack, and recipe corpus" not in md
    assert "identical LLM backbone within each case" in md


def test_a_single_generator_run_carries_no_such_warning(tmp_path):
    """The disclosure must not fire on a normal run, or it stops being read."""
    md = _render(
        tmp_path,
        {"model": "groq/qwen3.6-27b", "generators": {"groq/qwen3.6-27b": 2}},
        [_case("STD-01", "standard", "groq/qwen3.6-27b"),
         _case("STD-02", "standard", "groq/qwen3.6-27b")],
    )

    assert "TWO GENERATORS" not in md
    assert "identical LLM backbone, retrieval stack, and recipe corpus" in md
