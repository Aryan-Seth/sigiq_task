from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .text_normalize import NormalizedText, normalize_for_tts

_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)")


class _SreWorkerClient:
    def __init__(self) -> None:
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.pending: Dict[str, asyncio.Future[dict]] = {}
        self.reader_task: Optional[asyncio.Task[None]] = None
        self.stderr_task: Optional[asyncio.Task[None]] = None
        self.write_lock = asyncio.Lock()
        self.start_lock = asyncio.Lock()
        self.timeout_s = max(0.05, float(os.environ.get("TTS_SRE_TIMEOUT_S", "0.8")))

    async def _start(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            return
        async with self.start_lock:
            if self.proc is not None and self.proc.returncode is None:
                return
            worker_path = Path(__file__).resolve().parent / "tools" / "sre_worker.mjs"
            self.proc = await asyncio.create_subprocess_exec(
                "node",
                str(worker_path),
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.reader_task = asyncio.create_task(self._read_stdout())
            self.stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self.proc is not None
        stream = self.proc.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            req_id = str(payload.get("id", ""))
            fut = self.pending.pop(req_id, None)
            if fut is not None and not fut.done():
                fut.set_result(payload)
        self._fail_pending("sre worker exited")

    async def _read_stderr(self) -> None:
        assert self.proc is not None
        stream = self.proc.stderr
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            txt = line.decode("utf-8", errors="replace").strip()
            if txt:
                print(f"[sre-worker] {txt}", flush=True)

    def _fail_pending(self, message: str) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(message))
        self.pending.clear()

    async def convert_many(self, exprs: List[str], options: Dict[str, str]) -> List[str]:
        if not exprs:
            return []
        await self._start()
        assert self.proc is not None
        assert self.proc.stdin is not None
        req_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self.pending[req_id] = fut
        payload = {"id": req_id, "exprs": exprs, "options": options}
        async with self.write_lock:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self.proc.stdin.drain()
        reply = await asyncio.wait_for(fut, timeout=self.timeout_s)
        if "error" in reply:
            raise RuntimeError(str(reply.get("error")))
        spoken = reply.get("spoken", [])
        if not isinstance(spoken, list):
            return [str(spoken)]
        return [str(x) for x in spoken]

    async def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.returncode is None:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=1.5)
        except Exception:
            try:
                self.proc.kill()
                await self.proc.wait()
            except Exception:
                pass
        if self.reader_task is not None:
            self.reader_task.cancel()
        if self.stderr_task is not None:
            self.stderr_task.cancel()
        await asyncio.gather(
            *(t for t in [self.reader_task, self.stderr_task] if t is not None),
            return_exceptions=True,
        )
        self.proc = None
        self.reader_task = None
        self.stderr_task = None
        self._fail_pending("sre worker closed")


@dataclass
class MathNormalizer:
    mode: str
    sre: Optional[_SreWorkerClient] = None
    options: Optional[Dict[str, str]] = None

    async def normalize(self, text: str) -> NormalizedText:
        if self.mode == "off":
            return NormalizedText(text=text, source_indices=list(range(len(text))))
        if self.mode == "sre":
            if self.sre is None:
                return normalize_for_tts(text)
            exprs = [m.group(1) for m in _INLINE_MATH_RE.finditer(text)]
            if not exprs:
                return normalize_for_tts(text)
            try:
                spoken = await self.sre.convert_many(exprs, self.options or {})
            except Exception:
                return normalize_for_tts(text)
            idx = 0

            def _next_expr(_expr: str) -> str:
                nonlocal idx
                if idx < len(spoken):
                    out = spoken[idx]
                    idx += 1
                    return out
                return ""

            return normalize_for_tts(text, latex_expr_to_speech=_next_expr)
        return normalize_for_tts(text)

    async def close(self) -> None:
        if self.sre is not None:
            await self.sre.close()


def _get_mode() -> str:
    mode = os.environ.get("TTS_MATH_NORMALIZER", "sre").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return "off"
    if mode in {"sre", "latex-to-speech"}:
        return "sre"
    return "rule"


def get_math_normalizer_from_env() -> MathNormalizer:
    mode = _get_mode()
    if mode != "sre":
        return MathNormalizer(mode=mode)
    options: Dict[str, str] = {}
    domain = os.environ.get("TTS_SRE_DOMAIN", "clearspeak").strip()
    style = os.environ.get("TTS_SRE_STYLE", "").strip()
    locale = os.environ.get("TTS_SRE_LOCALE", "").strip()
    if domain:
        options["domain"] = domain
    if style:
        options["style"] = style
    if locale:
        options["locale"] = locale
    return MathNormalizer(mode="sre", sre=_SreWorkerClient(), options=options)
