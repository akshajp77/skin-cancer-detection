# Deploying the Skin Cancer Detection App

This is now a **single-service deployment**: one Docker container runs the FastAPI backend,
which also serves the built React frontend directly. One host, one URL, no separate
frontend/backend deploys, no CORS config, no environment variables to wire up between services.

## How it works

`Dockerfile` (repo root) is a two-stage build:
1. **Stage 1** builds the React app (`npm run build`) into static files.
2. **Stage 2** installs the Python backend and copies those static files into `backend/static/`.

At runtime, `main.py` serves `/predict` and `/health` as API routes, and mounts everything else
(`/`, `/assets/*`, etc.) to serve the built frontend from that `static/` folder. The frontend
calls `/predict` as a relative path — since it's served by the same process, there's no
cross-origin URL to configure.

In local dev, `vite.config.js` proxies `/predict` and `/health` to `http://127.0.0.1:8000`, so
the same relative-path code works whether you're running the Vite dev server + FastAPI
separately, or the built single-container version.

## Deploy to Render (or Railway / Fly.io — same Dockerfile works on any of them)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect the repo.
3. Render should auto-detect `render.yaml` and use it. If not, set manually:
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Docker Build Context:** `.` (repo root — important, since the build needs both
     `frontend/` and `backend/`)
   - **Plan:** Free (note: 512MB RAM is tight for TensorFlow; upgrade to Starter, $7/mo,
     if you see memory errors. Free tier also spins down after 15 min idle — first request
     after that takes ~30-60s to wake up.)
4. Deploy. Render builds the frontend, then the backend image, then starts the container.
5. Once live, open the URL Render gives you (e.g. `https://skin-cancer-detection.onrender.com`)
   directly in your browser — that's the actual app, frontend and all. No second deploy needed.

## Local development

Run frontend and backend as two processes (normal Vite + FastAPI dev workflow):

```bash
# Terminal 1
cd backend
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8000

# Terminal 2
cd frontend
npm install
npm run dev                # http://127.0.0.1:5173 - open this one in your browser
```

The Vite dev server proxies `/predict` calls to the backend automatically, so you don't need
any extra config to make this work locally.

## Verifying the deployed app

1. Open your Render URL. You should see the actual React UI, not a JSON response.
2. `curl https://your-app.onrender.com/health` → `{"status":"ok","model_loaded":true}`
3. Upload a lesion image through the UI — first request after idle may take ~30-60s on the
   free tier (cold start), after that it should respond in a couple seconds.
