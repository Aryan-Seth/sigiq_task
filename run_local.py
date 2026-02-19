from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.request import urlretrieve


def _healthcheck(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"http://{host}:{port}/", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.25)
    raise RuntimeError(f"Server did not become healthy: {last_exc}")


def _require_python_module(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing Python dependency '{module_name}'. {install_hint}"
        ) from exc


def _require_command(cmd: str, install_hint: str) -> None:
    from shutil import which

    if which(cmd) is None:
        raise RuntimeError(f"Missing command '{cmd}'. {install_hint}")


def _ensure_math_stack(project_dir: Path) -> None:
    _require_command("node", "Install Node.js 18+ to enable LaTeX-to-speech conversion.")
    pkg_json = project_dir / "package.json"
    if not pkg_json.is_file():
        raise RuntimeError(f"Missing package.json at {pkg_json}")

    node_module = project_dir / "node_modules" / "latex-to-speech"
    if node_module.is_dir():
        return
    print("Installing JS math dependencies (npm install)...", flush=True)
    subprocess.run(["npm", "install"], cwd=str(project_dir), check=True)


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        print(f"Downloading {url} -> {dst}", flush=True)
        urlretrieve(url, str(tmp))
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _ensure_piper_medium_files(project_dir: Path) -> tuple[Path, Path]:
    voice = project_dir / "en_US-lessac-medium.onnx"
    config = project_dir / "en_US-lessac-medium.onnx.json"
    if voice.is_file() and config.is_file():
        return voice, config

    voice_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    )
    config_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    )

    errors: list[str] = []
    if not voice.is_file():
        try:
            _download_file(voice_url, voice)
        except Exception as exc:
            errors.append(f"voice download failed: {exc}")
    if not config.is_file():
        try:
            _download_file(config_url, config)
        except Exception as exc:
            errors.append(f"config download failed: {exc}")

    if not voice.is_file() or not config.is_file():
        raise RuntimeError(
            "Piper medium files are missing and auto-download failed. "
            f"voice_exists={voice.is_file()} config_exists={config.is_file()} "
            f"errors={errors}"
        )
    return voice, config


def _build_env(args: argparse.Namespace, project_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TTS_BACKEND"] = args.backend
    env["ALIGNER"] = args.aligner
    env["TTS_MATH_NORMALIZER"] = args.math_normalizer
    env["TTS_PROFILE"] = "1"

    if args.backend == "piper":
        _require_python_module("piper", "Install with `pip install piper-tts`.")
        voice, config = _ensure_piper_medium_files(project_dir)
        env["PIPER_VOICE"] = str(voice)
        env["PIPER_CONFIG"] = str(config)

    if args.backend == "kokoro":
        _require_python_module("kokoro_onnx", "Install with `pip install kokoro-onnx`.")
        model = project_dir / "models" / "kokoro" / "kokoro-v1.0.onnx"
        voices = project_dir / "models" / "kokoro" / "voices-v1.0.bin"
        if not model.is_file() or not voices.is_file():
            raise FileNotFoundError(
                f"Kokoro files missing: {model} and/or {voices}. "
                "Download them first."
            )
        env["KOKORO_MODEL"] = str(model)
        env["KOKORO_VOICES"] = str(voices)
        env["KOKORO_VOICE"] = args.kokoro_voice
        env["KOKORO_LANG"] = args.kokoro_lang
        env["KOKORO_SPEED"] = str(args.kokoro_speed)

    if args.aligner == "whisperx":
        _require_python_module("whisperx", "Install with `pip install whisperx`.")

    if args.math_normalizer == "sre":
        _ensure_math_stack(project_dir)

    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production launcher for TTS WebSocket server."
    )
    parser.add_argument(
        "--profile",
        choices=["latency", "quality"],
        default="quality",
        help="latency: piper+rule+none, quality: piper+sre+whisperx",
    )
    parser.add_argument(
        "--backend", choices=["dummy", "piper", "kokoro", "say"], default=None
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--aligner", default=None)
    parser.add_argument("--math-normalizer", default=None)
    parser.add_argument("--kokoro-voice", default="af_sarah")
    parser.add_argument("--kokoro-lang", default="en-us")
    parser.add_argument("--kokoro-speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.backend is None:
        args.backend = "piper" if args.profile == "latency" else "piper"
    if args.aligner is None:
        args.aligner = "none" if args.profile == "latency" else "whisperx"
    if args.math_normalizer is None:
        args.math_normalizer = "rule" if args.profile == "latency" else "sre"

    project_dir = Path(__file__).resolve().parent
    app_file = project_dir / "app" / "server.py"
    if not app_file.is_file():
        raise FileNotFoundError(f"Could not find app entrypoint: {app_file}")

    env = _build_env(args, project_dir)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.server:app",
        "--app-dir",
        str(project_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    proc = subprocess.Popen(cmd, cwd=str(project_dir), env=env)
    try:
        _healthcheck(args.host, args.port, timeout_s=45.0)
        print(f"Server healthy at http://{args.host}:{args.port}/")
        print(f"WS endpoint: ws://{args.host}:{args.port}/tts")
        print(
            "Running with "
            f"backend={args.backend}, aligner={args.aligner}, math={args.math_normalizer}"
        )
        proc.wait()
        return int(proc.returncode or 0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
