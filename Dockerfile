FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./

FROM base AS backend
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
