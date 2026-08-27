# Deploying LunchBite

The dashboard deploys to **Streamlit Community Cloud**. This document is the
whole procedure, plus the reasoning behind the two choices that are not obvious
(why not Vercel, and why the index builds itself on boot).

---

## Why not Vercel or Cloudflare

Both were attempted; neither can host this app, and the failure is structural
rather than a configuration problem worth another attempt.

The Vercel error — `Found src/main.py but it does not export a top-level "app",
"application", or "handler" variable` — is the builder assuming `src/main.py` is
a serverless function entrypoint and looking for a WSGI/ASGI object. It is not
one: `src/main.py` orchestrates the LangGraph pipeline and exposes
`recommend_lunches`, not a web framework. Exporting an `app` from it would clear
that message and then fail for three larger reasons:

1. **Streamlit is a stateful, long-lived WebSocket server.** Session state and
   the rerun model need a process that outlives a request. Vercel functions are
   request/response and frozen in between. There is no supported way to run
   Streamlit on Vercel.
2. **Size.** `sentence-transformers` brings PyTorch; PyPI's default Linux torch
   wheel bundles the CUDA runtime and runs to several hundred megabytes on its
   own. Vercel's Python function limit is 250 MB unzipped.
3. **Cold start.** Loading MiniLM and the cross-encoder is tens of seconds. The
   function timeout is well below that on the Hobby plan.

Cloudflare Workers (the `wrangler` dependency in `package.json`) fails harder
still — no PyTorch at all.

`package.json`, `package-lock.json` and `node_modules/` are leftovers from those
attempts. Nothing in this project is a Node app; `node_modules/` is now
gitignored, and the two manifests can be deleted.

---

## Deploying

### 1. Confirm what gets committed

`vectordb/` is gitignored and stays that way — see below. `data/` is committed
and must be, since the index is built from it.

```bash
git status --short          # node_modules/ should no longer appear
git ls-files data | head     # the three JSON files must be tracked
```

### 2. Push to GitHub

Streamlit Cloud deploys from a GitHub repository, so `main` has to be pushed and
the repo readable by your Streamlit account.

### 3. Create the app

At <https://share.streamlit.io> → **New app**:

| Field | Value |
|---|---|
| Repository | your fork/remote of this project |
| Branch | `main` |
| Main file path | `app/LunchBite.py` |
| Python version | **3.12** (matches local development) |

### 4. Set the secrets

Before the first run finishes, open **Advanced settings → Secrets** (or
**Settings → Secrets** afterwards) and paste the contents of
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with your
real Groq key substituted.

Do not commit a filled-in `secrets.toml`. `.gitignore` blocks it.

### 5. First boot takes a few minutes

In order: pip install (torch dominates), then the embedding model and
cross-encoder download from Hugging Face (~90 MB each), then the vector index
builds. The app shows a spinner naming each of those. Subsequent restarts reuse
the container's caches and are quick.

---

## The two things deployment changed in the code

### Secrets reach `src/` through the environment

Everything under `src/` reads configuration with `os.environ` — `GROQ_API_KEY`,
`USE_CROSS_ENCODER_RERANKER`, and eighteen others. On Streamlit Cloud there is
no `.env` to load them from.

`app/lunchbite/bootstrap.py` copies every flat key in `st.secrets` into
`os.environ` at import, before any `src` module is touched. An already-set
environment variable wins, so local `.env` and shell behaviour are unchanged,
and `service.shadow_warnings()` still reports a shell value shadowing a file
value.

That last guarantee costs more than it looks. Streamlit promotes secrets into
`os.environ` itself while parsing the file, with a bare assignment and no check
for an existing value (`streamlit/runtime/secrets.py`). By the time any code of
ours could test whether a variable was already set, a shell variable of the same
name has been overwritten and the original is gone. `bootstrap` therefore
snapshots `os.environ` *before* the first key access and restores the
pre-existing values afterwards. It also writes the keys Streamlit skips —
`USE_CROSS_ENCODER_RERANKER = true` as a TOML boolean is not promoted, because
Streamlit excludes `bool` by design.

The alternative — teaching `llm_provider` and friends about `st.secrets` — would
make the CLI, the benchmark and the eval harness depend on the dashboard.
Everywhere else in this project that dependency points the other way.

### The index builds itself on first use

`vectordb/` is generated output and stays gitignored: regenerating it from
`data/` is exactly what `src/setup_database.py` is for, and a committed binary
index would drift from the JSON it was built from with nothing to catch it.

But a cloud container starts with no index and no shell to run
`setup_database.py` from. And the failure mode is quiet rather than loud:
`vector_store.get_collection()` calls `get_or_create_collection`, so a missing
index does not raise — the app comes up holding an **empty** collection and
serves BM25-only retrieval while presenting itself as the full pipeline.

`service.ensure_index()` closes that. It runs once per server, builds the
collection when it is missing or empty, and is a no-op locally where
`setup_database.py` has already run. It is cheap: the corpus is the three JSON
files under `data/` — a few hundred chunks — and the embedding model it needs is
the one the first query would load anyway.

---

## If the deployed app is killed for memory

The app holds two models: MiniLM for embeddings and the ms-marco cross-encoder
for reranking. If the container is killed, set in Secrets:

```toml
USE_CROSS_ENCODER_RERANKER = "false"
```

Retrieval then keeps RRF fusion order instead of reranking. That is a genuine
quality loss and it should be stated in the report if a deployed run is cited —
but it is a running app rather than a dead one, and it is the single largest
memory saving available without changing the pipeline.

---

## Local development is unchanged

```bash
pip install -r requirements.txt -r requirements-dev.txt
python src/setup_database.py
streamlit run app/LunchBite.py
```

`requirements.txt` now holds only what the deployed app imports, because it is
what Streamlit Cloud installs. Jupyter, matplotlib and pytest moved to
`requirements-dev.txt`. `requirements.lock.txt` is unchanged and still pins the
exact versions the published benchmark results were produced with.
