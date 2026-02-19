from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from .backends.registry import create_backend
from .orchestrator import run_session
from .aligner import get_aligner_from_env
from .math_normalizer import get_math_normalizer_from_env

app = FastAPI()
_BACKEND = None
_BACKEND_ERROR: Optional[str] = None
_BACKEND_LOCK = asyncio.Lock()

_ALIGNER = None
_ALIGNER_ERROR: Optional[str] = None
_ALIGNER_LOCK = asyncio.Lock()

_MATH_NORMALIZER = None
_MATH_NORMALIZER_ERROR: Optional[str] = None
_MATH_NORMALIZER_LOCK = asyncio.Lock()


async def _init_backend() -> None:
    global _BACKEND, _BACKEND_ERROR
    async with _BACKEND_LOCK:
        if _BACKEND is not None or _BACKEND_ERROR is not None:
            return
        backend_name = os.environ.get("TTS_BACKEND", "piper")
        try:
            _BACKEND = create_backend(backend_name)
            try:
                await asyncio.to_thread(_BACKEND.warmup)
            except Exception:
                pass
        except Exception as exc:
            _BACKEND_ERROR = str(exc)


async def _init_aligner() -> None:
    global _ALIGNER, _ALIGNER_ERROR
    async with _ALIGNER_LOCK:
        if _ALIGNER is not None or _ALIGNER_ERROR is not None:
            return
        try:
            _ALIGNER = get_aligner_from_env()
        except Exception as exc:
            _ALIGNER_ERROR = str(exc)

async def _init_math_normalizer() -> None:
    global _MATH_NORMALIZER, _MATH_NORMALIZER_ERROR
    async with _MATH_NORMALIZER_LOCK:
        if _MATH_NORMALIZER is not None or _MATH_NORMALIZER_ERROR is not None:
            return
        try:
            _MATH_NORMALIZER = get_math_normalizer_from_env()
        except Exception as exc:
            _MATH_NORMALIZER_ERROR = str(exc)


@app.on_event("startup")
async def _startup() -> None:
    await _init_backend()
    await _init_aligner()
    await _init_math_normalizer()


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _MATH_NORMALIZER
    if _MATH_NORMALIZER is not None:
        close_fn = getattr(_MATH_NORMALIZER, "close", None)
        if callable(close_fn):
            try:
                await close_fn()
            except Exception:
                pass


async def _get_backend():
    if _BACKEND is None and _BACKEND_ERROR is None:
        await _init_backend()
    return _BACKEND, _BACKEND_ERROR


async def _get_aligner():
    if _ALIGNER is None and _ALIGNER_ERROR is None:
        await _init_aligner()
    return _ALIGNER, _ALIGNER_ERROR

async def _get_math_normalizer():
    if _MATH_NORMALIZER is None and _MATH_NORMALIZER_ERROR is None:
        await _init_math_normalizer()
    return _MATH_NORMALIZER, _MATH_NORMALIZER_ERROR


@app.get("/")
async def index() -> FileResponse:
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.websocket("/tts")
async def tts_websocket(ws: WebSocket) -> None:
    await ws.accept()
    backend, backend_error = await _get_backend()
    if backend is None:
        reason = f"Backend init failed: {backend_error}"
        if len(reason) > 120:
            reason = reason[:117] + "..."
        await ws.close(code=1011, reason=reason)
        return
    aligner, aligner_error = await _get_aligner()
    if aligner_error:
        reason = f"Aligner init failed: {aligner_error}"
        if len(reason) > 120:
            reason = reason[:117] + "..."
        await ws.close(code=1011, reason=reason)
        return
    math_normalizer, math_normalizer_error = await _get_math_normalizer()
    if math_normalizer_error:
        reason = f"Math normalizer init failed: {math_normalizer_error}"
        if len(reason) > 120:
            reason = reason[:117] + "..."
        await ws.close(code=1011, reason=reason)
        return
    profile_enabled = os.environ.get("TTS_PROFILE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    await run_session(
        ws,
        backend,
        aligner=aligner,
        math_normalizer=math_normalizer,
        profile_enabled=profile_enabled,
    )
