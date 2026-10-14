# ============================================================
# Stage 1 — dependency builder (uv)
# ============================================================
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy only dependency manifests first — leverages layer cache
COPY pyproject.toml uv.lock* ./

# Install dependencies into an isolated virtual environment
RUN uv sync --no-dev --frozen --no-install-project

# ============================================================
# Stage 2 — runtime (non-root, minimal image)
# ============================================================
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy the venv from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Copy application source and config
COPY app/     /app/app/
COPY config.yaml /app/config.yaml

# Ensure the venv's binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Default to JSON logging in containers (Rule 6)
ENV SAATH__LOGGING__FORMAT=json
ENV SAATH__LOGGING__LEVEL=INFO

# Drop to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]
