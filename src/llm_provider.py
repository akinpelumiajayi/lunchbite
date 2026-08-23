"""
llm_provider.py -- Single source of truth for all LLM configuration.

Provider resolution (automatic, reads .env):
  1. Groq  -- if GROQ_API_KEY is set
  2. Ollama -- if Groq absent and Ollama is reachable at OLLAMA_BASE_URL

TWO SEPARATE MODELS are used:
  Generator model  -- fast, lower-cost (GROQ_MODEL / OLLAMA_MODEL)
                      used for lunch recommendation generation
  Judge model      -- independent lineage (GROQ_JUDGE_MODEL / OLLAMA_JUDGE_MODEL)
                      used for LLM-as-judge evaluation metrics
  The two come from DIFFERENT MODEL FAMILIES -- Qwen generates, an OpenAI
  open-weight model judges -- which is what actually blunts self-preferencing
  bias. Two models of the same lineage share pretraining data and RLHF
  conventions, so a judge drawn from the generator's family still recognises
  and rewards its own house style, however much larger it is.

No Anthropic dependency anywhere in this project.
"""

from __future__ import annotations
import os
import sys
import socket
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"


# ── Load .env ─────────────────────────────────────────────────────────────────

# Keys the user is told, by every error message in this project, to edit in .env.
# If the process environment already defines one of these with a DIFFERENT value,
# .env is silently ignored and the edit appears to do nothing.
_DOTENV_SHADOW_WATCH = ("GROQ_API_KEY", "LANGSMITH_API_KEY")


def _warn_if_env_shadows_dotenv() -> None:
    """
    Say so, loudly, when an exported variable overrules the .env line for it.

    load_dotenv(override=False) is the right precedence -- an explicit export
    should beat a file -- but it is invisible, and invisible precedence is how
    run 20260819_212458 died: a stale user-level GROQ_API_KEY from an old `setx`
    shadowed a freshly pasted, perfectly valid key in .env. Every 401 in that run
    pointed at a key the user had already replaced.
    """
    if not _ENV_PATH.exists():
        return
    try:
        text = _ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in _DOTENV_SHADOW_WATCH:
            continue
        value = value.strip().strip('"').strip("'")
        live = os.environ.get(key, "")
        if not value or not live or live == value:
            continue
        for msg in (
            f"WARNING: {key} in the environment SHADOWS the one in .env -- "
            f"the .env value is NOT being used.",
            f"  using   (environment): {live[:8]}... ({len(live)} chars)",
            f"  ignored (.env line)  : {value[:8]}... ({len(value)} chars)",
            "  If you just edited .env, the edit has no effect until the "
            "environment copy is cleared:",
            f"    this shell only : $env:{key}=$null      (PowerShell)",
            rf'    permanently     : reg delete "HKCU\Environment" /v {key} /f',
        ):
            print(msg, file=sys.stderr)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(_ENV_PATH), override=False)
        _warn_if_env_shadows_dotenv()
    except ImportError:
        if _ENV_PATH.exists():
            with open(_ENV_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            _warn_if_env_shadows_dotenv()


_load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_temperature(role: str = "generator") -> float:
    """Judge scoring defaults to 0.0 — a scorer should not sample."""
    if role == "judge":
        var, default = "JUDGE_TEMPERATURE", "0.0"
    else:
        var, default = "LLM_TEMPERATURE", "0.1"
    try:
        return float(os.environ.get(var, default))
    except ValueError:
        return float(default)


def _get_max_tokens(role: str = "generator") -> int:
    """
    Sized per role, because Groq bills the *reserved* max_tokens against the
    rate-limit budget, not the tokens actually generated.

    The judge once shared the generator's 2000 — a value sized for multi-menu
    JSON — while its prompts ask for one score and one sentence. Every judge
    call therefore reserved ~2081 tokens, exhausting the daily cap after 42 of
    the ~321 calls a 30-case run needs; the published means then rested on the
    handful that survived.

    450 is sized for the current judge, gpt-oss-120b: a reasoning model, so the
    reply is preceded by hidden reasoning tokens (observed completions
    198–228). At ~581 reserved tokens a call, its 200,000 tokens/day allows
    ~344 — a full run fits, with no truncation mid-verdict.

    1200 is sized the same way for the generator, and the generator needed it
    too after the move to qwen3.6-27b: its cap is 200,000 tokens/day where the
    retired llama-3.1-8b-instant had 500,000. Measured over 18 calls spanning
    all five case categories, prompts run 250–1907 tokens (median 562) and
    completions 7–751. At the old 2000 reservation a 30-case run bills
    90 x ~2562 = ~231k and dies around case 26 — the same quota collapse, moved
    from the judge to the generator. At 1200 it bills ~158k and completes, with
    1.6x headroom over the longest menu JSON observed.
    """
    if role == "judge":
        var, default = "JUDGE_MAX_TOKENS", "450"
    else:
        var, default = "LLM_MAX_TOKENS", "1200"
    try:
        return int(os.environ.get(var, default))
    except ValueError:
        return int(default)


def generator_max_tokens() -> int:
    """The generator's reserved max_tokens, for callers recording run metadata."""
    return _get_max_tokens("generator")


def judge_max_tokens() -> int:
    """The judge's reserved max_tokens, for callers recording run metadata."""
    return _get_max_tokens("judge")


# Which reasoning_effort a model accepts is a property of the MODEL, not of the
# role it is playing, and the vocabularies do not overlap: gpt-oss takes
# low|medium|high and rejects "none", qwen3 takes none|default and rejects
# "low". A model that does not reason at all rejects the parameter outright.
#
# So the default is looked up from the model id, and anything unrecognised gets
# no parameter sent. Keying this on the role instead would send "none" to
# whatever a user passed to `--model`, and 400 the whole run on a model that
# happens not to reason.
_REASONING_DEFAULTS = (
    ("gpt-oss", "low"),    # lowest setting it has; the judge's 450 tokens must
                           # cover hidden reasoning as well as the verdict
    ("qwen3", "none"),     # generation fills a JSON template — thinking tokens
                           # would come out of the menus' LLM_MAX_TOKENS budget
)


def _default_reasoning_effort(model: str) -> str:
    """The effort setting for a model id, or "" when it should not be sent."""
    name = model.lower()
    for family, effort in _REASONING_DEFAULTS:
        if family in name:
            return effort
    return ""


def _get_reasoning_effort(model: str, role: str = "generator") -> str:
    """
    How much hidden reasoning the model may spend before answering.

    Overridable per role, since that is how the two models are configured
    everywhere else. Set the variable to an empty value to suppress the
    parameter entirely.
    """
    var = "GROQ_JUDGE_REASONING_EFFORT" if role == "judge" else "GROQ_REASONING_EFFORT"
    if var in os.environ:
        return os.environ[var].strip()
    return _default_reasoning_effort(model)


def _ollama_reachable() -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# ── Provider builders ─────────────────────────────────────────────────────────

def _get_timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT_SECS", "60"))
    except ValueError:
        return 60.0


def _get_max_retries() -> int:
    try:
        return int(os.environ.get("LLM_MAX_RETRIES", "4"))
    except ValueError:
        return 4


def provider_available() -> bool:
    """True when some provider could serve a request. No client is constructed."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip()) or _ollama_reachable()


def _try_groq(model: str, role: str = "generator") -> Optional[Any]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq
        # Retries and a timeout matter here: an unretried 429 mid-benchmark used
        # to be recorded as a pipeline error and then dropped from the metrics,
        # quietly shrinking the sample a run was scored on.
        kwargs: dict = dict(
            api_key=key,
            model=model,
            temperature=_get_temperature(role),
            max_tokens=_get_max_tokens(role),
            timeout=_get_timeout(),
            max_retries=_get_max_retries(),
        )
        # Only sent when the model is known to accept it — see
        # _REASONING_DEFAULTS. An empty value means "omit", not "use a default".
        effort = _get_reasoning_effort(model, role)
        if effort:
            kwargs["reasoning_effort"] = effort
        return ChatGroq(**kwargs)
    except ImportError:
        raise RuntimeError("langchain-groq not installed. Run: pip install langchain-groq")


def _try_ollama(model: str, role: str = "generator") -> Optional[Any]:
    if not _ollama_reachable():
        return None
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except ImportError:
            raise RuntimeError("langchain-ollama not installed. Run: pip install langchain-ollama")
    kwargs: dict = {}
    if _ollama_json_mode():
        # Ollama constrains sampling to valid JSON when `format="json"`.
        #
        # Both roles in this project ask for JSON and parse it, and a small local
        # model is far worse at holding that format unaided than the cloud one.
        # Run 20260820_100542 (llama3.2, 3B) recorded 26 failures across 16 cases
        # -- every single error in the run -- all of them variants of
        # "unparseable JSON: Expecting ',' delimiter". The arms were not
        # unsafe, they were unreadable, so the judge had almost nothing to score.
        #
        # This does not guarantee the right SHAPE, only valid syntax; a response
        # missing `menu_options` still fails downstream, and that is a clearer
        # failure than a truncated brace.
        kwargs["format"] = "json"

    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=_get_temperature(role),
        num_predict=_get_max_tokens(role),
        # Bound the KV cache. llama3.1 and llama3.2 both advertise a 131,072-token
        # context, and Ollama sizes its allocation from that unless told otherwise:
        # llama-server was observed committing 9.7 GB for an 8B Q4 model whose
        # weights are 4.9 GB, the balance being context it never used.
        #
        # On a machine with 7.7 GB of RAM that is what exhausts the Windows commit
        # charge, and the failure is not a clean error -- Python reports
        # "The paging file is too small for this operation" (os error 1455), while
        # the same exhaustion inside Ollama's native backend kills the process
        # outright, which Git Bash prints as `Segmentation fault`.
        #
        # 8192 is chosen against measured demand, not guessed: the widest prompt
        # this pipeline builds is ~3,500 tokens (21 candidates after two refine
        # passes) plus a 1,200-token reply, so ~4,700 with headroom to spare.
        num_ctx=_get_ollama_num_ctx(),
        **kwargs,
        # ChatOllama takes no `timeout`; the underlying httpx client does, via
        # client_kwargs. Without this there is NO timeout on a local call at all
        # -- a model that is thrashing against a full page file blocks the run
        # indefinitely rather than failing and being recorded. The default is
        # deliberately far longer than the Groq one: a 3B model on CPU can take
        # minutes for a single generation, where the cloud takes seconds.
        client_kwargs={"timeout": _get_ollama_timeout()},
    )


def ollama_memory_warning() -> Optional[str]:
    """
    Warn when the configured Ollama models cannot fit in RAM together.

    Returns a message to print, or None when there is nothing to say. Advisory
    only: reported sizes are on-disk quantised weights and actual footprint
    varies with context length, so this is a heads-up before a long run, not a
    gate on starting one.
    """
    if not _ollama_reachable():
        return None
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import json as _json
        import urllib.request
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as r:
            tags = _json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

    sizes = {}
    for m in tags.get("models", []):
        name = m.get("name", "")
        sizes[name] = m.get("size", 0)
        sizes.setdefault(name.split(":")[0], m.get("size", 0))

    gen = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()
    judge = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1").strip()
    gen_b, judge_b = sizes.get(gen, 0), sizes.get(judge, 0)
    if not gen_b or not judge_b:
        return None

    total_b = _total_ram_bytes()
    if not total_b:
        return None

    gb = 1024 ** 3
    need = (gen_b + judge_b) / gb
    have = total_b / gb
    if need < have * 0.75:
        return None
    return (
        f"WARNING: Ollama generator ({gen}, {gen_b / gb:.1f} GB) and judge "
        f"({judge}, {judge_b / gb:.1f} GB) total {need:.1f} GB against {have:.1f} GB of RAM.\n"
        f"  Ollama holds a model resident for 5 minutes after its last call, so the two can\n"
        f"  be in memory at once and the allocation fails inside Ollama's native backend --\n"
        f"  which surfaces as `Segmentation fault`, naming nothing.\n"
        f"  This run evicts the generator before judging, which usually suffices. If it\n"
        f"  still dies, either:\n"
        f"    - use a smaller judge:  OLLAMA_JUDGE_MODEL=llama3.2  in .env\n"
        f"    - skip judging:         python run_all.py --provider ollama --no-judge\n"
        f"    - judge on Groq:        leave GROQ_API_KEY set and omit --provider for the judge"
    )


def _total_ram_bytes() -> int:
    """Total physical RAM, or 0 when it cannot be determined on this platform."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        # Advisory probe only -- a machine whose RAM cannot be read still runs.
        return 0


def _get_ollama_num_ctx() -> int:
    """Context window for local models. See the note in _try_ollama for the sizing."""
    try:
        return int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
    except ValueError:
        return 8192


def _ollama_json_mode() -> bool:
    """Set OLLAMA_JSON_MODE=false to sample freely (for debugging raw output)."""
    return os.environ.get("OLLAMA_JSON_MODE", "true").strip().lower() not in ("false", "0", "no")


def _get_ollama_timeout() -> float:
    try:
        return float(os.environ.get("OLLAMA_TIMEOUT_SECS", "300"))
    except ValueError:
        return 300.0


def unload_ollama_model(model: str) -> bool:
    """
    Ask Ollama to evict a model from memory now.

    Why this exists
    ---------------
    Ollama holds a model resident for `keep_alive` (5 minutes by default) after
    its last call. The benchmark generates with OLLAMA_MODEL and then judges with
    OLLAMA_JUDGE_MODEL, so the judge's weights are loaded while the generator's
    are still held -- llama3.2 (2.0 GB) plus llama3.1 (4.9 GB) is ~7 GB, which
    does not fit a machine with 7.7 GB of RAM. The allocation fails inside
    Ollama's native backend, and the run dies with a segmentation fault rather
    than anything that names memory as the cause.

    Evicting between phases costs one reload and makes the two models sequential
    rather than simultaneous. Best-effort: a failure here is not worth aborting a
    run over, since the next call simply proceeds as before.
    """
    if not _ollama_reachable():
        return False
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=_json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(prefer: Optional[str] = None) -> Tuple[Any, str]:
    """
    Returns (llm, provider_name) for the GENERATOR model.
    Generator: fast, lower-cost model for lunch recommendation text.
    """
    groq_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()
    return _resolve(groq_model, ollama_model, prefer, role="generator")


def get_judge_llm(prefer: Optional[str] = None) -> Tuple[Any, str]:
    """
    Returns (llm, provider_name) for the JUDGE model.
    MUST come from a different model family than the generator — see the module
    docstring — to avoid self-preferencing bias. Family, not size, is the
    property being enforced: gpt-oss-120b is a mixture-of-experts model with
    ~5.1B active parameters per token, so it is a comparable judge, not a
    larger one.
    """
    groq_model = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b").strip()
    ollama_model = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1").strip()
    return _resolve(groq_model, ollama_model, prefer, role="judge")


def _resolve(
    groq_model: str,
    ollama_model: str,
    prefer: Optional[str],
    role: str,
) -> Tuple[Any, str]:
    tried = []

    if prefer in (None, "groq"):
        llm = _try_groq(groq_model, role)
        if llm is not None:
            return llm, f"groq/{groq_model}"
        tried.append(f"groq (GROQ_API_KEY not set)")

    if prefer in (None, "ollama"):
        llm = _try_ollama(ollama_model, role)
        if llm is not None:
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return llm, f"ollama/{ollama_model}@{base}"
        tried.append("ollama (not reachable at OLLAMA_BASE_URL)")

    raise RuntimeError(
        f"No LLM provider available for {role} model. Tried: {', '.join(tried)}\n\n"
        "Options:\n"
        "  A) Groq (cloud, free):  Add GROQ_API_KEY=gsk_... to .env\n"
        "                          Get key at https://console.groq.com\n"
        "  B) Ollama (local):      Install from https://ollama.com\n"
        "                          Run: ollama pull llama3.2 && ollama serve\n"
        "  C) Mock (testing only): python3 run_all.py --mock"
    )


def verify_credentials() -> Tuple[bool, str]:
    """
    One cheap round trip to confirm the provider will actually accept the key.

    `provider_available()` above answers a different and much weaker question --
    is GROQ_API_KEY non-empty -- and the gap between the two cost the whole of run
    20260819_082917. The key had expired; all 90 generator calls and all 30 judge
    calls returned 401 `expired_api_key`; and the harness still spent three
    minutes producing a report in which the only arm with any data was `no_llm`,
    the one that never calls a model.

    Deliberately the model-listing endpoint rather than a one-token completion, so
    the check costs nothing against the daily token budget that _get_max_tokens
    goes to such lengths to protect.

    Returns (ok, detail). `ok` is False ONLY for a definite credential rejection:
    a timeout or a 5xx says nothing about the key, and blocking a run on a flaky
    probe would be a worse failure than the one this prevents.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        # An Ollama-only setup authenticates with nothing, so _ollama_reachable()
        # is already the entire check.
        return True, "no Groq key set — nothing to verify"

    try:
        from groq import Groq
    except ImportError:
        return True, "groq SDK not importable — credential check skipped"

    try:
        Groq(api_key=key, timeout=_get_timeout()).models.list()
        return True, "Groq key accepted"
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        from rate_limit import is_auth_failure
        if is_auth_failure(detail):
            return False, detail
        return True, f"inconclusive, run will proceed ({detail[:120]})"


def configure_langsmith(project_name: Optional[str] = None) -> bool:
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = (
        project_name or os.environ.get("LANGCHAIN_PROJECT", "lunch-rag-benchmark")
    )
    return True


def use_cross_encoder_reranker() -> bool:
    """
    Cross-encoder reranking is on by default. Set USE_CROSS_ENCODER_RERANKER=false
    to skip the stage and keep RRF fusion order.

    Note there is no corresponding embeddings flag: neural embeddings are the
    only backend. The former USE_HUGGINGFACE_EMBEDDINGS switch was inert — it
    was read only for the status printout while retrieval always used TF-IDF,
    so runs reported a retriever they were not using.
    """
    return os.environ.get("USE_CROSS_ENCODER_RERANKER", "true").lower() == "true"


def print_provider_status() -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
    groq_judge = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_judge = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1")
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    reranker = use_cross_encoder_reranker()

    print("LLM provider configuration:")
    if groq_key:
        print(f"  Groq generator : CONFIGURED  model={groq_model}")
        print(f"  Groq judge     : CONFIGURED  model={groq_judge}  (different family from generator)")
    else:
        print("  Groq           : not configured  (set GROQ_API_KEY in .env)")

    if _ollama_reachable():
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  Ollama generator: RUNNING  model={ollama_model} @ {base}")
        print(f"  Ollama judge    : RUNNING  model={ollama_judge} @ {base}")
    else:
        print("  Ollama         : not running  (start: ollama serve)")

    if langsmith_key:
        project = os.environ.get("LANGCHAIN_PROJECT", "lunch-rag-benchmark")
        print(f"  LangSmith      : CONFIGURED  project={project}")
    else:
        print("  LangSmith      : not configured  (optional, set LANGSMITH_API_KEY in .env)")

    # Retrieval status. Reports configured models and the real state of the
    # index, without loading the models (that would trigger a download here).
    from huggingface_upgrade.huggingface_embeddings import DEFAULT_HF_MODEL
    from huggingface_upgrade.reranker import DEFAULT_CE_MODEL, rerank_top_k

    embed_model = os.environ.get("HF_EMBEDDING_MODEL", DEFAULT_HF_MODEL)
    ce_model = os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_CE_MODEL)

    print(f"  Embeddings     : {embed_model}  (neural, sentence-transformers)")
    if reranker:
        print(f"  Cross-encoder  : ENABLED  model={ce_model}  top_k={rerank_top_k()}")
    else:
        print("  Cross-encoder  : disabled  (USE_CROSS_ENCODER_RERANKER=false)")

    try:
        from vector_store import get_collection
        print(f"  Vector index   : {get_collection().count()} chunks")
    except Exception as e:
        print(f"  Vector index   : unavailable ({type(e).__name__}) — run: python src/setup_database.py")
