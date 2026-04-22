# Langfuse self-hosting and tracing

The Streamlit app and pipeline scripts can optionally report traces to
[Langfuse](https://langfuse.com/). Each chat turn is captured as a single
`chat_turn` trace with nested spans for retrieval, reasoning, and LLM
generations (model name, input/output, token usage, and sampling config).

Tracing is fully optional: when `LANGFUSE_PUBLIC_KEY` or
`LANGFUSE_SECRET_KEY` is missing, `src/services/tracing.py` degrades to a
no-op and no network calls are made.

## Enabling tracing

Set the following in `src/.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse-web:3000
```

`LANGFUSE_HOST` defaults to `https://cloud.langfuse.com`. Point it at
`http://langfuse-web:3000` when using the bundled self-hosted stack below,
or at your own Langfuse URL.

## Self-hosted stack via a Compose overlay

The base Compose file keeps the app/Qdrant/SearXNG stack lightweight.
Langfuse lives in a separate overlay `docker-compose.langfuse.yml`, so you
only pay the extra resources when you actually want tracing.

Start both stacks together:

```bash
docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up -d
```

Stop them:

```bash
docker compose -f docker-compose.yml -f docker-compose.langfuse.yml down
```

The overlay declares an `app` override that adds `depends_on: langfuse-web`
and injects `LANGFUSE_HOST=http://langfuse-web:3000`, so the Streamlit
container reaches Langfuse over the Compose network automatically.

### Services in the overlay

| Service | Image | Purpose |
| --- | --- | --- |
| `langfuse-web` | `langfuse/langfuse:3` | UI + ingestion API (port `3000`) |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | Async ingestion worker |
| `langfuse-postgres` | `postgres:17` | Metadata store |
| `langfuse-clickhouse` | `clickhouse/clickhouse-server:24.8` | Traces / observations store |
| `langfuse-valkey` | `valkey/valkey:8` | Queue + cache (Redis-protocol-compatible) |
| `langfuse-seaweedfs` | `chrislusf/seaweedfs:4.20` | S3-compatible event/media store (master + volume + filer + S3 gateway in one process) |
| `langfuse-seaweedfs-init` | `amazon/aws-cli:2.17.0` | One-shot sidecar that creates the `langfuse` bucket once SeaweedFS is healthy |

## Bootstrap workflow

1. Start the overlay with the command above and open `http://localhost:3000`.
2. Create an account, an organization, and a project through the Langfuse UI,
   or pre-provision them via the `LANGFUSE_INIT_*` variables below.
3. Copy the generated `Public Key` and `Secret Key` from the project settings.
4. Add them to `src/.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`,
   then restart the `app` service so the new keys are picked up.

## Recommended `src/.env` entries

```dotenv
# Langfuse client (read by the Streamlit app)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse-web:3000

# Langfuse server secrets (change these in any non-local deployment)
LANGFUSE_NEXTAUTH_URL=http://localhost:3000
LANGFUSE_NEXTAUTH_SECRET=replace-with-long-random-string
LANGFUSE_SALT=replace-with-long-random-string
LANGFUSE_ENCRYPTION_KEY=<openssl rand -hex 32>
LANGFUSE_POSTGRES_PASSWORD=postgres
LANGFUSE_CLICKHOUSE_PASSWORD=clickhouse
LANGFUSE_VALKEY_AUTH=valkey
LANGFUSE_S3_ACCESS_KEY_ID=langfuse
LANGFUSE_S3_SECRET_ACCESS_KEY=langfusesecret

# Optional: pre-provision an org/project/user on first boot
# LANGFUSE_INIT_ORG_ID=trend-to-rule
# LANGFUSE_INIT_ORG_NAME=trend-to-rule
# LANGFUSE_INIT_PROJECT_ID=trend-to-rule
# LANGFUSE_INIT_PROJECT_NAME=trend-to-rule
# LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-local-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-local-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# LANGFUSE_INIT_USER_EMAIL=admin@example.com
# LANGFUSE_INIT_USER_NAME=Admin
# LANGFUSE_INIT_USER_PASSWORD=change-me
```

Generate a 32-byte hex encryption key with:

```bash
openssl rand -hex 32
```

## Persistence

All Langfuse state is bind-mounted from the host under `.data/langfuse/`
(git-ignored), alongside Qdrant's `.data/qdrant/`:

- `.data/langfuse/postgres/` — metadata store
- `.data/langfuse/clickhouse/` — traces and observations
- `.data/langfuse/clickhouse-logs/` — ClickHouse server logs
- `.data/langfuse/valkey/` — queue + cache
- `.data/langfuse/seaweedfs/` — event and media blob store

On first boot Docker creates these directories on the host as `root:root`.
Postgres, ClickHouse, Valkey, and SeaweedFS all initialise as root before
dropping to their service user, so they chown their own data dirs and work
out of the box. If you run a variant that cannot chown (for example a
rootless Docker setup), pre-create the directories with appropriate
ownership before the first `up`.

## SeaweedFS notes

SeaweedFS is started in all-in-one `server -s3` mode (master, volume, filer,
and S3 gateway in one process). Credentials are written to
`/etc/seaweedfs/s3config.json` at container start from
`LANGFUSE_S3_ACCESS_KEY_ID` / `LANGFUSE_S3_SECRET_ACCESS_KEY`, so no secrets
are baked into the repo.

Because SeaweedFS does not auto-create buckets on first `PutObject`, the
`langfuse-seaweedfs-init` sidecar runs once to create the `langfuse` bucket
via the AWS CLI. `langfuse-web` and `langfuse-worker` wait for that sidecar
to complete successfully (`service_completed_successfully`) before starting,
so event uploads have a writable bucket from the first request.
