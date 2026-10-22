# ============================================================================
# deployment/Dockerfile — MOSAIC API container (built for Cloud Run)
#
# BUILD (from the project root, not from deployment/):
#   docker build -f deployment/Dockerfile -t mosaic-api .
#
# RUN LOCALLY:
#   docker run --rm -p 8000:8000 --env-file .env mosaic-api
#
# DEPLOY TO CLOUD RUN:
#   gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/mosaic-api
#   gcloud run deploy mosaic-api \
#       --image gcr.io/$GCP_PROJECT_ID/mosaic-api \
#       --region us-central1 \
#       --add-cloudsql-instances $GCP_PROJECT_ID:us-central1:INSTANCE \
#       --set-env-vars "DB_HOST=/cloudsql/$GCP_PROJECT_ID:us-central1:INSTANCE" \
#       --set-secrets   "OPENAI_API_KEY=openai-api-key:latest,DB_PASSWORD=db-password:latest"
# ============================================================================

FROM python:3.12-slim
# slim keeps the image small. 3.12 is chosen deliberately: the codebase
# uses `str | None` unions and built-in generics freely, which need 3.10+.

# ── PYTHON RUNTIME BEHAVIOUR ────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    # Write logs straight through instead of buffering them. Without this,
    # Cloud Run shows nothing until the buffer flushes — so the logs of a
    # crash arrive after the crash, or not at all.
    PYTHONDONTWRITEBYTECODE=1 \
    # No .pyc files. The filesystem is read-only and discarded anyway.
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── DEPENDENCIES FIRST, FOR LAYER CACHING ───────────────────────────────────
# requirements.txt is copied and installed BEFORE the source. Docker caches
# each layer, so editing an agent re-runs only the (fast) code copy below,
# not the (slow) dependency install.
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── APPLICATION SOURCE ──────────────────────────────────────────────────────
# Copied as separate directories rather than "COPY . ." so that notebooks,
# local .env files and test fixtures never end up in the image.
COPY config/     ./config/
COPY ingestion/  ./ingestion/
COPY processing/ ./processing/
COPY memory/     ./memory/
COPY tools/      ./tools/
COPY graph/      ./graph/
COPY agents/     ./agents/
COPY api/        ./api/
COPY evaluation/ ./evaluation/

# ── NON-ROOT USER ───────────────────────────────────────────────────────────
# Nothing in this container needs root. If the process is ever compromised,
# it should not be able to write to the image or escalate.
RUN useradd --create-home --shell /bin/bash mosaic \
    && chown -R mosaic:mosaic /app
USER mosaic

EXPOSE 8000

# ── START ───────────────────────────────────────────────────────────────────
# Cloud Run injects the port to listen on as $PORT and it is not always
# 8000, so the shell form is used here to let $PORT expand at runtime.
# The fallback keeps `docker run` working locally where $PORT is unset.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
