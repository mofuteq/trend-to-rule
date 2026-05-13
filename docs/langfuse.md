# Langfuse Cloud observability

RepoA uses Langfuse Cloud as the default observability backend for development
and OSS demos. Self-hosted Langfuse was useful as an infrastructure spike and
as a bridge to private workplace deployments, but it is now deprecated for this
repository. RepoA now prioritizes workflow/runtime design over observability
infrastructure operations.

Tracing is fully optional: when `LANGFUSE_PUBLIC_KEY` or
`LANGFUSE_SECRET_KEY` is missing, `src/services/tracing.py` degrades to a
no-op and no Langfuse network calls are made.

## Enabling tracing

Set the following in `src/.env`:

```dotenv
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
APP_ENV=development
```

`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are required only when tracing
is enabled. Leave either key empty to disable tracing.

`LANGFUSE_BASE_URL` is the only supported Langfuse endpoint variable in RepoA.

## Trace shape

Each Streamlit chat turn is captured as a single `chat_turn` trace with nested
spans for retrieval, reasoning, and LLM generations. LLM generation spans carry
the model name, input/output, token usage, and sampling config.

In-scope requests show `analyze_request`, `route_by_scope`,
`retrieve_supporting_context`, and the Fixed RAR nodes. Out-of-scope requests
show `analyze_request`, `route_by_scope`, and `out_of_scope_response`.

LangGraph runs still pass Langfuse's LangChain callback handler into
`graph.invoke()` so the in-scope graph view remains visible in Langfuse. This
does not change the workflow behavior or SQLite checkpoint backend.

## RepoA identification

RepoA traces include stable tags and metadata so they can be separated from
other personal OSS apps in the same Langfuse account:

- Tags: `repoa`, `trend-to-rule`, `oss-demo`, `chat_turn`,
  `chat_workflow`, `langgraph`, workflow version, vertical, and scope status.
- Metadata: `app_id=repoa`, `app_name=trend-to-rule`, `repo=RepoA`,
  `observability_backend=langfuse-cloud`, `environment=<APP_ENV>`, chat id,
  session id, thread id, workspace/user id, and request details.

Use `APP_ENV` for environment-like grouping such as `development`, `demo`, or
`local`. Use Langfuse tags, metadata filters, projects, or dashboards to keep
RepoA traces distinct from other OSS experiments.

## Self-hosted Langfuse

Local self-hosted Langfuse is deprecated for RepoA. Do not use the local stack
for normal development or OSS demos. If a private deployment needs Langfuse,
use `LANGFUSE_BASE_URL` for that deployment and provide keys from that Langfuse
project.
