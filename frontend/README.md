# BrainSentinel frontend

React + Vite single-page UI for the BrainSentinel research prototype (anomaly-score review tool, not a diagnostic tool).
All results, scores, images and strings come from the backend API; nothing is mocked.

## Run

```
npm install
npm run dev      # http://localhost:5173
npm run build    # production build in dist/
npm test         # vitest unit tests
```

The backend (FastAPI, port 8000) must be running. The Vite dev server proxies `/api` and `/health` to
`http://127.0.0.1:8000` (see `vite.config.js`), so the app uses relative URLs. For a production build, serve `dist/`
behind a reverse proxy that forwards the same two paths to the backend.

Navigation is hash based: `#/` dashboard, `#/upload`, `#/report/<report_id>`.
