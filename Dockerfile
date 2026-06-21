# ---- Stage 1: build the React frontend into static files ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend that also serves the built frontend ----
FROM python:3.11-slim
WORKDIR /app

# System libs needed by opencv-python-headless and tensorflow at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py .
COPY backend/best_model.keras .

# Bring in the built frontend; main.py serves this directory at "/" if it exists
COPY --from=frontend-build /frontend/dist ./static

# Most hosts (Render, Railway, Fly.io) inject $PORT at runtime; default to 8000 for local docker run.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
