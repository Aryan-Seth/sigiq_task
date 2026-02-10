#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/content}"
REPO_ROOT="${REPO_ROOT:-$WORK_ROOT/sigiq_takehome}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-$WORK_ROOT/CosyVoice}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG_FILE="${LOG_FILE:-$WORK_ROOT/tts_ws_server.log}"

usage() {
  cat <<EOF
Usage: bash colab_t4_cosyvoice.sh <install|start|smoke>

Environment overrides:
  WORK_ROOT=/content
  REPO_ROOT=/content/sigiq_takehome
  COSYVOICE_REPO_DIR=/content/CosyVoice
  PORT=8000
  HOST=0.0.0.0
EOF
}

prepare_requirements_for_py312() {
  local req_in="$1"
  local req_out="$2"
  # grpcio==1.57.0 has no stable py3.12 wheel path.
  # Deepspeed/openai-whisper/pyworld/tensorrt are not required for this websocket inference path.
  awk '
    /^--extra-index-url/ { print; next }
    /^grpcio==/ { print "grpcio>=1.62.2"; next }
    /^grpcio-tools==/ { print "grpcio-tools>=1.62.2"; next }
    /^deepspeed==/ { next }
    /^openai-whisper==/ { next }
    /^pyworld==/ { next }
    /^tensorrt-cu12==/ { next }
    /^tensorrt-cu12-bindings==/ { next }
    /^tensorrt-cu12-libs==/ { next }
    { print }
  ' "$req_in" > "$req_out"
}

prepare_requirements_minimal_fallback() {
  local req_in="$1"
  local req_out="$2"
  awk '
    /^--extra-index-url/ { print; next }
    /^deepspeed==/ { next }
    /^openai-whisper==/ { next }
    /^pyworld==/ { next }
    /^grpcio==/ { next }
    /^grpcio-tools==/ { next }
    /^fastapi==/ { next }
    /^fastapi-cli==/ { next }
    /^gradio==/ { next }
    /^gdown==/ { next }
    /^lightning==/ { next }
    /^tensorboard==/ { next }
    /^matplotlib==/ { next }
    /^pyarrow==/ { next }
    /^tensorrt-cu12==/ { next }
    /^tensorrt-cu12-bindings==/ { next }
    /^tensorrt-cu12-libs==/ { next }
    { print }
  ' "$req_in" > "$req_out"
}

install() {
  if [[ ! -d "$REPO_ROOT" ]]; then
    echo "Repo not found at REPO_ROOT=$REPO_ROOT" >&2
    exit 1
  fi

  nvidia-smi || true

  if [[ ! -d "$COSYVOICE_REPO_DIR" ]]; then
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_REPO_DIR"
  else
    (
      cd "$COSYVOICE_REPO_DIR"
      git submodule update --init --recursive
    )
  fi

  python -m pip install -U pip setuptools wheel

  local py_minor
  py_minor="$(python - <<'PY'
import sys
print(sys.version_info.minor)
PY
)"

  local req_path="$COSYVOICE_REPO_DIR/requirements.txt"
  local req_to_install="$req_path"
  local tmp_req=""
  if [[ "$py_minor" -ge 12 ]]; then
    tmp_req="$(mktemp -t cosyvoice_req.XXXXXX.txt)"
    prepare_requirements_for_py312 "$req_path" "$tmp_req"
    req_to_install="$tmp_req"
    echo "Using patched CosyVoice requirements for Python 3.$py_minor"
  fi

  if ! python -m pip install --prefer-binary -r "$req_to_install"; then
    echo "Primary CosyVoice dependency install failed; retrying minimal fallback set..." >&2
    local tmp_req_fallback
    tmp_req_fallback="$(mktemp -t cosyvoice_req_fallback.XXXXXX.txt)"
    prepare_requirements_minimal_fallback "$req_to_install" "$tmp_req_fallback"
    python -m pip install --prefer-binary -r "$tmp_req_fallback"
    rm -f "$tmp_req_fallback"
  fi
  python -m pip install -r "$REPO_ROOT/requirements.runtime.txt"

  if [[ -n "$tmp_req" && -f "$tmp_req" ]]; then
    rm -f "$tmp_req"
  fi
}

start() {
  if [[ ! -d "$REPO_ROOT" ]]; then
    echo "Repo not found at REPO_ROOT=$REPO_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$COSYVOICE_REPO_DIR" ]]; then
    echo "CosyVoice repo not found at COSYVOICE_REPO_DIR=$COSYVOICE_REPO_DIR" >&2
    echo "Run: bash colab_t4_cosyvoice.sh install" >&2
    exit 1
  fi

  (
    cd "$REPO_ROOT"
    export TTS_BACKEND=cosyvoice
    export COSYVOICE_REPO_DIR="$COSYVOICE_REPO_DIR"
    export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-FunAudioLLM/CosyVoice2-0.5B}"
    export COSYVOICE_MODE="${COSYVOICE_MODE:-sft}"
    export TTS_PROFILE="${TTS_PROFILE:-1}"
    nohup python -m uvicorn app.server:app \
      --host "$HOST" \
      --port "$PORT" \
      --log-level warning >"$LOG_FILE" 2>&1 &
  )

  sleep 2
  pgrep -fl "uvicorn app.server:app" || true
  tail -n 30 "$LOG_FILE" || true
}

smoke() {
  python - <<'PY'
import asyncio
import json
import os
import websockets

PORT = int(os.environ.get("PORT", "8000"))

async def run():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/tts", max_size=None) as ws:
        await ws.send(json.dumps({"text": " ", "flush": False}))
        await ws.send(json.dumps({"text": "Hello from Colab T4 CosyVoice.", "flush": False}))
        await ws.send(json.dumps({"text": "", "flush": True}))
        await ws.send(json.dumps({"text": "", "flush": False}))
        msg = await ws.recv()
        print(str(msg)[:200])

asyncio.run(run())
PY
}

cmd="${1:-}"
case "$cmd" in
  install)
    install
    ;;
  start)
    start
    ;;
  smoke)
    smoke
    ;;
  *)
    usage
    exit 1
    ;;
esac
