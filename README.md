# trend-to-rule

[![CI](https://github.com/mofuteq/trend-to-rule/actions/workflows/ci.yml/badge.svg)](https://github.com/mofuteq/trend-to-rule/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green)
![SQLite](https://img.shields.io/badge/checkpoints-SQLite-blue)
![Pydantic AI](https://img.shields.io/badge/LLM-Pydantic%20AI-purple)
![OpenRouter](https://img.shields.io/badge/provider-OpenRouter-black)
![Langfuse](https://img.shields.io/badge/tracing-Langfuse-orange)
![Tavily](https://img.shields.io/badge/search-Tavily-teal)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

`trend-to-rule` is a search-native Agentic RAR runtime for turning noisy trend
narratives into decision-grade rules.

In fast-moving domains, signals often split into two layers: the **canonical** —
patterns that hold across cycles — and the **emerging** — what is loud right now.
Most systems surface one side or the other. `trend-to-rule` retrieves both,
compares them, and extracts their shared structure as an inspectable rule for a
human decision-maker.

The system is not a recommender, not a forecaster, and not a styling AI. It does
not tell the user what to wear, buy, publish, or believe. It returns structured
decision-support artifacts: typed sources, extracted claims, common rules,
conflicts, gaps, tradeoffs, and visual references.

Fashion and styling are the current evaluation domain — not because the system
has any commitment to fashion as a subject, but because taste is a domain where
many people have judgments they cannot fully articulate. The gap between “this
looks right” and “I can explain why” is unusually visible in fashion, which makes
it a useful place to test whether an agent can lend articulation without taking
over the judgment.

The architecture is domain-agnostic. Wherever short-term signals can obscure
long-term structure — product strategy, content selection, capital allocation,
or other taste-driven domains — the same pattern applies.

See [`examples/sample-output.md`](./examples/sample-output.md) for a full run on
a fashion query.

**RAR** means **Retrieval Augmented Reasoning**: retrieval is treated as part of
the reasoning workflow rather than passive context lookup.

**Search-native** means the system uses live web search as its primary evidence
source, not a prebuilt corpus, local vector database, or fine-tuned model
knowledge.

LangGraph checkpoints are persisted to local SQLite by default at
`.data/langgraph/checkpoints.sqlite`, so the default runtime does not require a
database service.

## Output Boundary

`trend-to-rule` returns frames, not verdicts.

The final answer is written as continuous prose, but it is generated from
structured intermediate artifacts. The system reasons through claims, canonical
and emerging evidence, conflicts, gaps, tradeoffs, and common rules, then renders
without exposing that internal schema as the user-facing answer.

Every final answer must include an `Interpreted Rules` section, localized as
`解釈ルール` for Japanese output. Those rules use observation-grounded language:

> When an observable condition appears, it may signal an underlying
> interpretation.

The system should not collapse evidence into a recommendation. The human remains
the decision-maker; the agent lends a reference frame and then stops.

## Current Retrieval Backend

Tavily is the default and only text evidence backend in this repository.

The app runs two text searches per in-scope request:

- `canonical_query`
- `emerging_query`

Those queries come from `RequestAnalysis.candidate_queries`. Raw Tavily payloads
are never passed directly to the LLM. Search results are normalized into stable
`WebSource` models first:

```python
class WebSource(BaseModel):
    source_id: str
    query_kind: Literal["canonical", "emerging"]
    title: str
    url: str
    snippet: str
    published_at: str | None = None
    score: float | None = None
    provider: Literal["tavily"] = "tavily"
```

Sources are deduplicated by normalized URL, then rendered into
`canonical_context` and `emerging_context` for the existing RAR stages:

```text
retrieve_supporting_context
  -> extract_claims
  -> extract_structured_draft
  -> generate_decision_support
  -> generate_query
  -> render_image_query
  -> search_images
```

If `TAVILY_API_KEY` is missing, or if no text evidence can be retrieved, the
workflow abstains instead of producing a confident evidence-based answer.

## Visual References

Visual retrieval also uses Tavily, but it is downstream of rule generation.

The workflow does not send the final answer directly to image search. It first
converts the rule into an `ExampleQuerySpec`, renders a compact image query, and
then requests Tavily image candidates. The pipeline normalizes those candidates,
deduplicates image URLs and page/title pairs, and selects the top candidates in
Tavily-provided order.

Visual references are optional supporting examples, not the core reasoning
source. No local embedding model is required for the default runtime.

## Architecture

```mermaid
flowchart TD
    E[Streamlit app.py] --> W[chat_workflow LangGraph]
    W --> A[analyze_request]
    A --> R{route_by_scope}
    R -->|out of scope| OOS[out_of_scope_response]
    R -->|in scope| Q[canonical_query / emerging_query]
    Q --> T[Tavily text search]
    T --> S[normalize to WebSource]
    S --> D[dedupe by normalized URL]
    D --> C[canonical_context / emerging_context]
    C --> EC[extract_claims]
    EC --> ESD[extract_structured_draft]
    ESD --> GDS[generate_decision_support]
    GDS --> GQ[generate_query]
    GQ --> RIQ[render_image_query]
    RIQ --> SI[Tavily image search]
    SI --> E

    subgraph Observability
      LF[Langfuse Cloud tracing]
    end
    W --> LF
```

The LangGraph workflow is implemented in
[`src/services/chat_workflow.py`](./src/services/chat_workflow.py). It keeps
request analysis, scope routing, the out-of-scope path, the structured RAR
stages, visual retrieval, Langfuse tracing, and the SQLite checkpoint backend.

In-scope runs log text evidence metadata to Langfuse:

- `text_retrieval_backend="tavily"`
- `canonical_source_count`
- `emerging_source_count`
- `total_source_count`

Out-of-scope requests do not run `retrieve_supporting_context` and do not call
Tavily text search.

## Directory Layout

```text
trend-to-rule/
├── src/
│   ├── app.py
│   ├── Dockerfile
│   ├── core/
│   ├── pipeline/
│   ├── prompt_template/
│   ├── services/
│   ├── storage/
│   └── ui/
├── pyproject.toml
├── uv.lock
└── README.md
```

- `src/core/`: runtime config, domain models, text/query helpers.
- `src/services/web_search.py`: Tavily text evidence search and `WebSource`
  normalization.
- `src/services/image_search.py`: Tavily image search, candidate normalization,
  deduplication, and top-candidate selection.
- `src/services/chat_workflow.py`: LangGraph workflow orchestration.
- `src/services/chat.py`: LLM-backed request analysis, claim extraction,
  structured draft, decision support, and query generation.
- `src/prompt_template/`: prompts for each structured stage.
- `src/storage/`: LMDB-backed chat persistence.
- `src/ui/`: Streamlit rendering and session state.

## Environment Setup

Install dependencies with:

```bash
uv sync
```

Create `src/.env` for local and Docker runs.

The app uses Pydantic AI with OpenRouter-specific provider/model classes at the
LLM boundary. The workflow architecture remains provider-independent, but the
runtime does not treat OpenAI-compatible APIs as the architectural contract.

Set the OpenRouter fields together: `OPENROUTER_MODEL`, `OPENROUTER_API_KEY`,
`OPENROUTER_OUTPUT_RETRIES`, and `OPENROUTER_REASONING_EFFORT`.

Use the model id from OpenRouter, such as `google/gemini-3-flash-preview` or
`nvidia/nemotron-3-super-120b-a12b:free`. Do not include the LiteLLM-style
`openrouter/` prefix.

```dotenv
OPENROUTER_MODEL=google/gemini-3-flash-preview
OPENROUTER_API_KEY=
OPENROUTER_OUTPUT_RETRIES=3
OPENROUTER_REASONING_EFFORT=low

TAVILY_API_KEY=
TAVILY_TEXT_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=basic
TAVILY_INCLUDE_RAW_CONTENT=false
TAVILY_IMAGE_FETCH_LIMIT=10
TAVILY_IMAGE_LIMIT=3
TAVILY_INCLUDE_IMAGE_DESCRIPTIONS=true

CHAT_DB_PATH=.data/chat_db
T2R_DEFAULT_WORKSPACE=demo
APP_LOG_LEVEL=INFO
APP_ENV=development

LANGFUSE_BASE_URL="https://cloud.langfuse.com"
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=

LANGGRAPH_SQLITE_PATH=.data/langgraph/checkpoints.sqlite
```

Key settings:

- `OPENROUTER_MODEL`: OpenRouter model identifier, e.g.
  `google/gemini-3-flash-preview` or
  `nvidia/nemotron-3-super-120b-a12b:free`.
- `OPENROUTER_API_KEY`: required OpenRouter API key.
- `OPENROUTER_OUTPUT_RETRIES`: maximum structured-output validation retries
  before Pydantic AI raises a model-behavior error. Default: `3`.
- `OPENROUTER_REASONING_EFFORT`: OpenRouter reasoning effort passed through
  `openrouter_reasoning`. One of `minimal`, `low`, `medium`, `high`, `xhigh`.
  Default: `low`.
- `TAVILY_API_KEY`: required for in-scope text evidence retrieval and also used
  by visual retrieval.
- `TAVILY_TEXT_MAX_RESULTS`: per-lane text result cap. Default: `5`.
- `TAVILY_SEARCH_DEPTH`: Tavily search depth. Default: `basic`.
- `TAVILY_INCLUDE_RAW_CONTENT`: whether Tavily may return raw page content.
  Default: `false`.
- `TAVILY_IMAGE_FETCH_LIMIT`: raw image candidate count requested from Tavily.
- `TAVILY_IMAGE_LIMIT`: final visual reference card count selected from Tavily
  results.
- `TAVILY_INCLUDE_IMAGE_DESCRIPTIONS`: whether Tavily should return image
  descriptions when available.
- `LANGFUSE_BASE_URL`: defaults to Langfuse Cloud.
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`: tracing is disabled if either
  key is empty.
- `LANGGRAPH_SQLITE_PATH`: local SQLite file for LangGraph checkpoints.
  Default: `.data/langgraph/checkpoints.sqlite`.

## Run Locally

Launch the Streamlit app with uv:

```bash
uv run streamlit run src/app.py
```

## Run With Docker

Build the standalone app image:

```bash
docker build -f src/Dockerfile -t trend-to-rule .
```

Run the Streamlit app:

```bash
docker run --rm \
  --env-file src/.env \
  -p 8501:8501 \
  -v "$(pwd)/.data:/app/.data" \
  trend-to-rule
```

The app is available at `http://localhost:8501`.

## Observability

`trend-to-rule` uses Langfuse Cloud as the default observability backend for development
and OSS demos. Set `LANGFUSE_BASE_URL="https://cloud.langfuse.com"` with a
Langfuse Cloud public/secret key pair in `src/.env`; tracing activates
automatically when both keys are present.

Each Streamlit chat turn is captured as a single `chat_turn` trace with native
LangGraph callback events. In-scope requests show `analyze_request`,
`route_by_scope`, `retrieve_supporting_context`, and the RAR nodes.
Out-of-scope requests show `analyze_request`, `route_by_scope`, and
`out_of_scope_response`.

LLM calls in `src/services/llm_client.py` are recorded as generation spans with
backend, model name, input messages, output, token usage, sampling config,
reasoning effort, and structured-output retry metadata.

See [docs/langfuse.md](./docs/langfuse.md) for the current observability setup.

## License

This project is licensed under the MIT License.
