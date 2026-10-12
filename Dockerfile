# syntax=docker/dockerfile:1
#
# Delivery-fleet allocation engine - one image, one port.
#
#   stage frontend   builds the React SPA with Vite (npm ci -> dist/)
#   stage runtime    Python 3.11 + runtime deps + engine + tests + SPA + RL artifact
#   stage test       adds pytest/httpx and runs the suite (`docker build --target test`)
#
# Build the production image:
#   docker build -t delivery-fleet-allocation-engine:1.0.0 .
# Run it:
#   docker run --rm -p 8000:8000 delivery-fleet-allocation-engine:1.0.0
#   -> UI and API on http://localhost:8000

# ---------------------------------------------------------------- frontend --
FROM node:20.19-alpine AS frontend

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build && ls -la dist/assets

# ----------------------------------------------------------------- runtime --
FROM python:3.11.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    RL_POLICY_PATH=/app/backend/artifacts/rl_policy.json

WORKDIR /app/backend
COPY backend/requirements.txt backend/requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# engine, tests, pytest config and the committed policy artifact
COPY backend/pyproject.toml ./
COPY backend/app ./app
COPY backend/tests ./tests
COPY backend/artifacts ./artifacts
# the SPA is served by FastAPI from the same origin (backend/app/main.py looks
# for <repo root>/frontend/dist, so the path below must match)
COPY --from=frontend /build/frontend/dist /app/frontend/dist

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=8s --retries=5 \
    CMD ["python", "-c", "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3).status==200 else 1)"]

# uvicorn serves both the API and the built SPA on one port
CMD ["python", "-m", "app.main"]

# -------------------------------------------------------------------- test --
FROM runtime AS test
USER root
RUN pip install --no-cache-dir -r requirements-dev.txt
USER appuser
CMD ["python", "-m", "pytest", "-q"]
