FROM debian:stable-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system trend-to-rule \
    && useradd --system --create-home --gid trend-to-rule --shell /bin/bash trend-to-rule

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ENV HOME="/home/trend-to-rule"
ENV PATH="/usr/local/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR="/home/trend-to-rule/.local/share/uv/python"
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN chown -R trend-to-rule:trend-to-rule /app "${HOME}"

USER trend-to-rule

RUN uv python install 3.13.12
RUN uv sync --frozen --python 3.13.12

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
