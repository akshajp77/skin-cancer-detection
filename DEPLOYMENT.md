# Deploying the Skin Cancer Detection App

This covers what changed to make the app deployable, and the steps to actually get it live.

## What was wrong before

1. **Frontend hardcoded `http://127.0.0.1:8000`** as the backend URL in `App.jsx`. No matter
   where you deployed the frontend, it would only ever try to reach your own laptop — so
   it silently fell back to the local mock simulator. This is why the Vercel deploy "didn't work."
2. **`opencv-python` (not headless)** pulls in GUI/X11 libraries that most container hosts
   (Render, Railway, Fly.io) don't have installed, causing `ImportError: libGL.so.1` at runtime.
3. **No Dockerfile.** Render/Railway can build Python apps without one, but their auto-detection
   for TensorFlow + OpenCV is unreliable — a Dockerfile gives you a guaranteed-reproducible build.
4. **CORS was wide open (`*`)** with no way to restrict it later via config.
5. **No `$PORT` handling** — most hosts assign a port dynamically via an env var; the app
   needs to listen on whatever they hand it.

## What changed

- `frontend/src/App.jsx` — backend URL now reads from `VITE_API_URL` (env var), falls back to
  localhost for local dev.
- `frontend/.env.example` — documents the var.
- `backend/requirements.txt` — `opencv-python` → `opencv-python-headless`, `tensorflow` →
  `tensorflow-cpu` (smaller, faster build, no GPU needed for inference).
- `backend/main.py` — CORS origins now configurable via `ALLOWED_ORIGINS` env var (comma-separated,
  defaults to `*`); listens on `$PORT`; added a `/health` route for the host's health checks.
- `backend/Dockerfile` + `.dockerignore` — reproducible container build.
- `render.yaml` — optional, lets Render auto-configure the service from the repo.
- Removed `backend/__pycache__` from git tracking (it was accidentally committed).

## Step 1 — Deploy the backend (Render)

Railway works identically; just skip the Render-specific UI steps below and use Railway's
"Deploy from GitHub" with the same Dockerfile.

1. Push these changes to `github.com/akshajp77/skin-cancer-detection`.
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect the repo.
3. Render should detect `render.yaml` automatically. If not, set manually:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Dockerfile Path:** `backend/Dockerfile`
   - **Plan:** Free (note: free tier spins down after 15 min idle — first request after
     idle takes ~30-60s to wake up, and has 512MB RAM which is tight for TF + the model;
     upgrade to Starter ($7/mo) if you hit memory errors)
4. Deploy. Once live, note the URL Render gives you, e.g. `https://skin-cancer-detection-api.onrender.com`.
5. Test it: `curl https://your-app.onrender.com/health` should return `{"status":"ok","model_loaded":true}`.

## Step 2 — Deploy the frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the repo.
2. **Root Directory:** `frontend`
3. Vercel auto-detects Vite (build command `npm run build`, output `dist`) — no changes needed.
4. Under **Environment Variables**, add:
   - `VITE_API_URL` = `https://your-app.onrender.com` (the URL from Step 1, no trailing slash)
5. Deploy.

## Step 3 — Lock down CORS (optional but recommended)

Once you have your real Vercel URL, go back to Render → your service → **Environment** and set:

```
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

Redeploy the backend. This restricts API access to just your frontend instead of any website.

## Verifying end to end

1. Open your Vercel URL.
2. Upload a lesion image.
3. If it calls the real backend, you'll see a short delay (cold start on free tier) then a
   real Grad-CAM heatmap. If it instantly shows a result with no network delay, it silently
   fell back to the local simulator — check the browser console for a fetch error and confirm
   `VITE_API_URL` is set correctly in Vercel and the backend `/health` endpoint responds.
