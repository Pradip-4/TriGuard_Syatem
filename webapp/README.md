# TriGuard Web Console

A clean, minimal browser interface for the TriGuard authentication system
(face + voice + lip). It runs the **real** TriGuard pipeline — nothing is
simulated.

## Run it

```powershell
cd <project root>         # the folder that contains config.json
python webapp/app.py
```

Then open **<http://127.0.0.1:5000>** in your browser.

> First startup loads the face / voice / lip models once (~45–55 s on this
> machine). After that, every session reuses the loaded models.

## What you see

- **Live camera preview** while idle and during capture.
- **Challenge sentence + 4-second countdown** (set in `config.json` →
  `modules.lip.countdown_seconds`).
- **Per-module status** (Face / Voice / Lip) with PASS/FAIL and scores.
- **Result + timing breakdown** (countdown, recording, each module, total).

## Configuration

- `TRIGUARD_PORT` — port (default `5000`).
- `TRIGUARD_HOST` — bind host (default `127.0.0.1`).
- `TRIGUARD_CAMERA` — camera index override (default: from `config.json`).
- `config.json → system.capture_max_width` — frames are downscaled to this
  width before recording/analysis (default `640`) for faster CPU processing
  without recognition-quality loss. Set `0` to keep native resolution.

## Endpoints

| Method | Path              | Purpose                                        |
|--------|-------------------|------------------------------------------------|
| GET    | `/`               | The web UI                                     |
| GET    | `/api/preview.mjpg` | MJPEG live preview stream                    |
| GET    | `/api/events`     | SSE stream: stages, countdown, module status, result |
| POST   | `/api/start`      | Start an authentication session               |
| GET    | `/api/status`     | JSON snapshot: status, preload time, last result, session history |
| GET    | `/api/snapshot.jpg` | JPEG of the most recent capture frame (shown in the result card) |

## UI features (v2)

- **Professional dark console** — glassy cards, gradient accents, countdown ring,
  animated 4-step session stepper, per-module progress + score chips.
- **Session history** — the last 6 results are kept server-side and survive page
  refreshes (verdict chips: identity · time · score).
- **Capture snapshot** — the last recorded frame is shown beside each result
  (purely local, never leaves the machine).

## Notes for the demo

- Module **preload happens once** at server start — don't restart the server
  between demos.
- Analysis runs on CPU on this machine, so allow up to ~1–2 minutes after
  recording for the result card.
- If a module fails (e.g. no face detected), the session still finishes and
  shows a FAIL — the pipeline never hangs.