# =============================================================================
# Dockerfile — Code Review Environment for LLM Agents
# =============================================================================
# Production image with FastAPI server for OpenEnv compatibility.
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies (cached layer)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . .

# Make startup script executable
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose the server port (OpenEnv standard: 8000 internal, 7860 mapped by HF)
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------------------------
# Entry point — run server.py directly (avoids server/ package name conflict)
# ---------------------------------------------------------------------------
CMD ["python", "server.py"]
