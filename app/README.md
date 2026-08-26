# LunchBite

A Streamlit dashboard over the lunch recommendation pipeline in `src/`.

```bash
pip install -r requirements.txt
python src/setup_database.py        # once, if vectordb/ is not built
streamlit run app/LunchBite.py
```

Opens at <http://localhost:8501>.

## What it shows

| Page | Purpose |
|---|---|
| **LunchBite** | Child profile in, recommended lunches out. Each card carries why it fits, ingredients, nutrition against the age-band guideline, allergens confirmed absent, and the verified source. |
| **Safety report** | What the guardrail rejected and on what grounds, before and after the model was asked. This is the page the project's argument lives on. |
| **Pipeline trace** | Retrieval rankings (BM25, embeddings, RRF, cross-encoder), per-stage timings, and the model's raw reply. |
| **Recipe explorer** | All 29 recipes, filterable. No API key or prior run needed. |
| **System health** | Is a model reachable, is the index built, and which gate modes are in force. |

## No photographs

`data/recipes.json` has no image field for any of the 29 recipes. A stock photo
of a similar dish would assert something about the recommendation that is not
true, so LunchBite shows a `meal_category` mark and the real figures instead.

## Running without an API key

The **No LLM (rule-based)** arm runs retrieval and the deterministic guardrail
with no model, no key and no network. It is selected automatically when no
provider is configured, and the other four arms are hidden rather than offered
and then failing.

## Layout

```
app/
  LunchBite.py            entry point; profile form and recommendations
  pages/                  the four sub-pages, in sidebar order
  lunchbite/
    bootstrap.py          sys.path wiring; must be imported before anything in src/
    service.py            the only module that touches the pipeline
    vocab.py              form options derived from the guardrail vocabularies
    theme.py              palette, meal-category accents, pipeline-arm metadata
    components.py         cards, chips, nutrient meters, charts
```

Two constraints hold the design together:

**The display contract is not reimplemented.** A terminal pipeline state becomes
a result through `main.shape_result`, the same function `recommend_lunches` ends
in. `src/main.py` documents what it cost last time two implementations of that
mapping drifted apart: the CLI and the benchmark measured different systems, so
no reported number described what a user actually got.

**Form options come from the safety module, not from a list here.** `vocab.py`
builds them out of `guardrails.ALLERGY_SYNONYMS`, `DIET_SPECS` and
`document_loader.ALL_14_ALLERGENS`, so the form cannot offer a term the gate
cannot enforce, and a new synonym in the guardrail appears in the form without
anyone remembering to add it.

## Tests

```bash
pytest tests/test_app_service.py -q
```

Includes a parity test that `shape_result` and `recommend_lunches` agree, and an
end-to-end `AppTest` run of the whole page in the no-LLM arm that checks the
recommended lunches against the recipe records rather than against the model's
claims about them.
