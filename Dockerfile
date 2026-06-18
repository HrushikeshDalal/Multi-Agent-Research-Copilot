# ── Stage 1: dependency builder ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed for some compiled packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly to the runtime stage
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as a non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Drop to non-root
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Default command — overridden by docker-compose for the Streamlit service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]