FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS base
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./

FROM base AS backend
# Hermes Agent only installs in editable mode from a checkout (see README)
ARG HERMES_COMMIT=c0ef5af5205dbdc25e9e44acd66873b4bb3503c9
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git init -q vendor/hermes-agent \
    && git -C vendor/hermes-agent fetch -q --depth 1 https://github.com/NousResearch/hermes-agent.git $HERMES_COMMIT \
    && git -C vendor/hermes-agent checkout -q FETCH_HEAD
RUN uv sync --frozen --no-install-project --extra backend
COPY agent_backend agent_backend
COPY api_common api_common
EXPOSE 8000
CMD ["uvicorn", "agent_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend
RUN uv sync --frozen --no-install-project --extra frontend
COPY api_common api_common
COPY frontend_demo frontend_demo
EXPOSE 8501
CMD ["streamlit", "run", "frontend_demo/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
