# Web dashboard (React + FastAPI)

Browser UI for the sign-to-speech pipeline, with a separate **Speech → Sign**
screen marked as coming soon.

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + Vite |
| Backend | FastAPI + WebSocket |
| ML | Existing Python modules (`modules/`, `pipeline/`) |

## Install

```bash
./.venv/bin/pip install -r app/requirements-web.txt
cd app/web && npm install
```

## Run

**Development** (React hot reload on port 5173, API on 8000):

```bash
./.venv/bin/python scripts/run_web_app.py --dev
```

Open **http://127.0.0.1:5173**

**Production-style** (single server on 8000):

```bash
./.venv/bin/python scripts/run_web_app.py
```

Open **http://127.0.0.1:8000**

## Layout

- **Centre:** live webcam with hand box and hold-progress bar
- **Left:** Alphabet / Words toggle, current sign, confidence, committed tokens
- **Words mode:** Record sign → Stop & detect (WLASL), then Finish & speak
- **Right:** gloss, English, emotion, latency, and Play buttons for EN/NE audio
- **Top toggle:** Sign → Speech | Speech → Sign (coming soon)

## API

- `GET /api/health`
- `GET /api/models`
- `GET /api/audio/{path}` — serves cached files from `logs/tts/`
- `WS /ws/sign-to-speech` — JPEG frames in, recognition state out; `finish` runs NLP → emotion → TTS

Optional: `export TF_CPP_MIN_LOG_LEVEL=2` to quiet TensorFlow logs on first load.
