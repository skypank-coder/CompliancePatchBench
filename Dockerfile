FROM python:3.11-slim

# Install system dependencies for the health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Health check uses the same PORT the app uses (${PORT} from the container env)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /bin/sh -c 'curl -fsS "http://127.0.0.1:${PORT:-7860}/health" >/dev/null || exit 1'

EXPOSE 7860

# Listen on 0.0.0.0:7860 by default. Set PORT for platforms that inject a different value.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
