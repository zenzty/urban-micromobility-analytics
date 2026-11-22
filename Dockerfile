FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install locked third-party runtime dependencies first for Docker layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application source and install the local package in editable mode (preserves config.PROJECT_ROOT)
COPY src ./src
COPY scripts ./scripts
COPY data/.gitkeep ./data/.gitkeep
RUN uv sync --locked --no-dev

# Run unprivileged with write access scoped only to /app/data (avoids duplicating /app/.venv layers)
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/raw /app/data/processed \
    && chown -R appuser:appuser /app/data

USER appuser

# Build validated Parquet datasets during image build and clear raw archives
RUN python scripts/build_dataset.py && rm -rf data/raw/*

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"10000\")}/liveness', timeout=4)"]

CMD ["python", "-c", "import gc, os, panel as pn; from micromobility_analytics.ui.app import create_dashboard_app, get_cached_datasets; pn.config.reuse_sessions = 'warm'; pn.config.global_loading_spinner = True; get_cached_datasets(); gc.collect(); pn.serve({'': create_dashboard_app}, port=int(os.environ.get('PORT', '10000')), address='0.0.0.0', websocket_origin=['*'], use_xheaders=True, liveness=True, warm=True, keep_alive_milliseconds=15000, unused_session_lifetime_milliseconds=300000, check_unused_sessions_milliseconds=60000, show=False)"]
