#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${T2R_IMAGE:-trend-to-rule:dev}"
NETWORK="${T2R_DOCKER_NETWORK:-trend-to-rule-dev}"
API_CONTAINER="${T2R_API_CONTAINER:-trend-to-rule-api}"
UI_CONTAINER="${T2R_UI_CONTAINER:-trend-to-rule-ui}"
ENV_FILE="${T2R_ENV_FILE:-src/.env}"
DATA_DIR="${T2R_DATA_DIR:-.data}"
API_PORT="${T2R_API_PORT:-8000}"
UI_PORT="${T2R_UI_PORT:-8501}"

usage() {
  cat <<USAGE
Usage: scripts/run-local-containers.sh [start|stop|restart|status|logs]

Environment overrides:
  T2R_IMAGE             Docker image tag. Default: trend-to-rule:dev
  T2R_DOCKER_NETWORK    Docker network name. Default: trend-to-rule-dev
  T2R_API_CONTAINER     FastAPI container name. Default: trend-to-rule-api
  T2R_UI_CONTAINER      Streamlit container name. Default: trend-to-rule-ui
  T2R_ENV_FILE          Env file path. Default: src/.env
  T2R_DATA_DIR          Host data dir mounted into FastAPI. Default: .data
  T2R_API_PORT          Host FastAPI port. Default: 8000
  T2R_UI_PORT           Host Streamlit port. Default: 8501
USAGE
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

read_env_value() {
  local wanted_key="$1"
  local line key value
  [[ -f "$ENV_FILE" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" == *"="* ]] || continue
    key="$(trim "${line%%=*}")"
    [[ "$key" == "$wanted_key" ]] || continue
    value="$(trim "${line#*=}")"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    printf '%s' "$value"
    return 0
  done <"$ENV_FILE"
}

docker_env_args() {
  local langfuse_base_url
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing env file: $ENV_FILE" >&2
    exit 1
  fi

  ENV_ARGS=(--env-file "$ENV_FILE")

  # Docker's --env-file keeps quote characters. Override this one documented
  # quoted dotenv value so Langfuse receives a valid URL inside containers.
  langfuse_base_url="$(read_env_value "LANGFUSE_BASE_URL")"
  if [[ -n "$langfuse_base_url" ]]; then
    ENV_ARGS+=(-e "LANGFUSE_BASE_URL=$langfuse_base_url")
  fi
}

ensure_network() {
  docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local attempt
  for attempt in {1..30}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

stop_containers() {
  docker rm -f "$UI_CONTAINER" "$API_CONTAINER" >/dev/null 2>&1 || true
}

start_containers() {
  docker_env_args
  mkdir -p "$DATA_DIR"

  docker build -f src/Dockerfile -t "$IMAGE" .
  ensure_network
  stop_containers

  docker run -d --rm \
    --name "$API_CONTAINER" \
    --network "$NETWORK" \
    "${ENV_ARGS[@]}" \
    -v "$ROOT_DIR/$DATA_DIR:/app/.data" \
    -p "$API_PORT:8000" \
    "$IMAGE" \
    uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 >/dev/null

  wait_for_url "FastAPI" "http://127.0.0.1:$API_PORT/health"

  docker run -d --rm \
    --name "$UI_CONTAINER" \
    --network "$NETWORK" \
    "${ENV_ARGS[@]}" \
    -e "T2R_API_BASE_URL=http://$API_CONTAINER:8000" \
    -p "$UI_PORT:8501" \
    "$IMAGE" >/dev/null

  wait_for_url "Streamlit" "http://127.0.0.1:$UI_PORT/_stcore/health"

  echo "FastAPI:   http://localhost:$API_PORT"
  echo "Streamlit: http://localhost:$UI_PORT"
}

case "${1:-start}" in
  start)
    start_containers
    ;;
  stop)
    stop_containers
    ;;
  restart)
    stop_containers
    start_containers
    ;;
  status)
    docker ps --filter "name=$API_CONTAINER" --filter "name=$UI_CONTAINER" \
      --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
    ;;
  logs)
    docker logs -f "$API_CONTAINER" &
    api_logs_pid=$!
    docker logs -f "$UI_CONTAINER" &
    ui_logs_pid=$!
    trap 'kill "$api_logs_pid" "$ui_logs_pid" >/dev/null 2>&1 || true' INT TERM EXIT
    wait
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
