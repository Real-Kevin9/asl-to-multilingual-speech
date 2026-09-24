from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.session import (  # noqa: E402
    RealtimeSession,
    parse_nlp_mode,
    safe_audio_path,
)
from pipeline.model_selection import format_model_report, resolve_models  # noqa: E402

WEB_DIST = REPO_ROOT / "app" / "web" / "dist"

app = FastAPI(
    title="ASL Multilingual Speech",
    description="Web dashboard for sign-to-speech with speech-to-sign coming soon.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionConfig(BaseModel):
    backend: str = "auto"
    nlp: str = "off"
    use_nlp_model: Optional[bool] = None
    languages: list[str] = Field(default_factory=lambda: ["en", "ne"])
    hold_time: float = 0.9


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "modes": ["sign_to_speech", "speech_to_sign_coming_soon"]}


@app.get("/api/models")
def models(backend: str = "auto", nlp: str = "auto") -> dict:
    selection = resolve_models(recognition_backend=backend, nlp_mode=nlp)
    return {
        "report": format_model_report(selection),
        "selection": selection.as_dict(),
    }


@app.get("/api/audio/{file_path:path}")
def audio(file_path: str) -> FileResponse:
    resolved = safe_audio_path(file_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    media = "audio/mpeg" if resolved.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(resolved, media_type=media)


@app.websocket("/ws/sign-to-speech")
async def sign_to_speech_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session: Optional[RealtimeSession] = None

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")

            if msg_type == "configure":
                cfg = message.get("config") or {}
                if session is not None:
                    session.close()
                session = RealtimeSession(
                    backend=cfg.get("backend", "auto"),
                    recognition_mode=str(cfg.get("recognition_mode", "alphabet")),
                    use_nlp_model=parse_nlp_mode(
                        cfg.get("nlp", "off"),
                        cfg.get("use_nlp_model"),
                    ),
                    hold_time=float(cfg.get("hold_time", 0.9)),
                    languages=tuple(cfg.get("languages") or ("en", "ne")),
                )
                try:
                    info = session.load()
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    session = None
                    continue
                await websocket.send_json({"type": "ready", "models": info})
                continue

            if session is None:
                session = RealtimeSession()
                info = session.load()
                await websocket.send_json({"type": "ready", "models": info})

            try:
                if msg_type == "frame":
                    state = await asyncio.to_thread(
                        session.process_frame, message.get("image", "")
                    )
                    await websocket.send_json({"type": "state", **state})
                elif msg_type == "clear":
                    session.clear()
                    await websocket.send_json({"type": "state", **session.snapshot()})
                elif msg_type == "start_sign":
                    await websocket.send_json({"type": "state", **session.start_sign()})
                elif msg_type == "commit_sign":
                    state = await asyncio.to_thread(session.commit_sign)
                    await websocket.send_json({"type": "state", **state})
                elif msg_type == "cancel_sign":
                    await websocket.send_json({"type": "state", **session.cancel_sign()})
                elif msg_type == "finish":
                    result = await asyncio.to_thread(session.finish)
                    await websocket.send_json({"type": "result", **result})
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {msg_type}"}
                    )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.close()


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404)
        target = WEB_DIST / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
