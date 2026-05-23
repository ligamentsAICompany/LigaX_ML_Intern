FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create user with UID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy application code (includes backend/static/index.html)
COPY agent/ ./agent/
COPY backend/ ./backend/
COPY configs/ ./configs/

RUN mkdir -p /app/session_logs && chown -R user:user /app

USER user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
WORKDIR /app/backend
CMD ["bash", "start.sh"]
