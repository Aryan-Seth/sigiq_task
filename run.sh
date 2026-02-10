#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-warning}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN to a valid interpreter, e.g. PYTHON_BIN=python3.11 ./run.sh"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing runtime dependencies..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.runtime.txt >/dev/null

# Default runtime config
export TTS_BACKEND="${TTS_BACKEND:-piper}"
export TTS_PROFILE="${TTS_PROFILE:-1}"
export TTS_MATH_NORMALIZER="${TTS_MATH_NORMALIZER:-rule}"

case "${TTS_BACKEND}" in
  dummy|piper|say|kokoro|cosyvoice)
    ;;
  *)
    echo "Unsupported TTS_BACKEND=${TTS_BACKEND}"
    echo "Supported backends: dummy, piper, say, kokoro, cosyvoice"
    exit 1
    ;;
esac

if [[ "${TTS_BACKEND}" == "piper" ]]; then
  if [[ -z "${PIPER_VOICE:-}" ]]; then
    if [[ -f "$ROOT_DIR/en_US-lessac-low.onnx" ]]; then
      export PIPER_VOICE="$ROOT_DIR/en_US-lessac-low.onnx"
    elif [[ -f "$ROOT_DIR/en_US-lessac-medium.onnx" ]]; then
      export PIPER_VOICE="$ROOT_DIR/en_US-lessac-medium.onnx"
    elif [[ -f "$ROOT_DIR/en_US-lessac-high.onnx" ]]; then
      export PIPER_VOICE="$ROOT_DIR/en_US-lessac-high.onnx"
    else
      echo "No bundled Piper voice found. Set PIPER_VOICE to a local .onnx file."
      exit 1
    fi
  fi

  if [[ -z "${PIPER_CONFIG:-}" && -f "${PIPER_VOICE}.json" ]]; then
    export PIPER_CONFIG="${PIPER_VOICE}.json"
  fi
fi

if [[ "${TTS_BACKEND}" == "say" ]]; then
  if ! command -v say >/dev/null 2>&1; then
    echo "TTS_BACKEND=say requires macOS 'say' command."
    exit 1
  fi
  if ! command -v afconvert >/dev/null 2>&1; then
    echo "TTS_BACKEND=say requires 'afconvert' command."
    exit 1
  fi
  export SAY_VOICE="${SAY_VOICE:-Eddy (English (US))}"
  export SAY_RATE="${SAY_RATE:-190}"
fi

if [[ "${TTS_BACKEND}" == "kokoro" ]]; then
  if [[ -z "${KOKORO_MODEL:-}" ]]; then
    if [[ -f "$ROOT_DIR/models/kokoro/kokoro-v1.0.int8.onnx" ]]; then
      export KOKORO_MODEL="$ROOT_DIR/models/kokoro/kokoro-v1.0.int8.onnx"
    elif [[ -f "$ROOT_DIR/models/kokoro/kokoro-v1.0.onnx" ]]; then
      export KOKORO_MODEL="$ROOT_DIR/models/kokoro/kokoro-v1.0.onnx"
    else
      echo "Kokoro model not found."
      echo "Download one to models/kokoro/, e.g.:"
      echo "  curl -L -o models/kokoro/kokoro-v1.0.int8.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
      exit 1
    fi
  fi

  if [[ -z "${KOKORO_VOICES:-}" ]]; then
    if [[ -f "$ROOT_DIR/models/kokoro/voices-v1.0.bin" ]]; then
      export KOKORO_VOICES="$ROOT_DIR/models/kokoro/voices-v1.0.bin"
    else
      echo "Kokoro voices file not found."
      echo "Download it with:"
      echo "  curl -L -o models/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
      exit 1
    fi
  fi
  export KOKORO_VOICE="${KOKORO_VOICE:-af_sarah}"
  export KOKORO_LANG="${KOKORO_LANG:-en-us}"
  export KOKORO_SPEED="${KOKORO_SPEED:-1.0}"
fi

if [[ "${TTS_BACKEND}" == "cosyvoice" ]]; then
  export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-FunAudioLLM/CosyVoice2-0.5B}"
  export COSYVOICE_MODE="${COSYVOICE_MODE:-sft}"
  export COSYVOICE_SPEED="${COSYVOICE_SPEED:-1.0}"
  export COSYVOICE_STREAM_CHUNK_MS="${COSYVOICE_STREAM_CHUNK_MS:-120}"

  if [[ -z "${COSYVOICE_REPO_DIR:-}" && -d "$ROOT_DIR/CosyVoice" ]]; then
    export COSYVOICE_REPO_DIR="$ROOT_DIR/CosyVoice"
  fi

  if [[ -z "${COSYVOICE_REPO_DIR:-}" ]]; then
    echo "Warning: COSYVOICE_REPO_DIR is not set."
    echo "If backend init fails, set it to a local FunAudioLLM/CosyVoice checkout."
  fi
fi

if [[ "${TTS_MATH_NORMALIZER}" == "sre" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "TTS_MATH_NORMALIZER=sre requires npm (Node.js)."
    echo "Either install Node.js or use TTS_MATH_NORMALIZER=rule."
    exit 1
  fi
  if [[ ! -d "$ROOT_DIR/node_modules" ]]; then
    echo "Installing Node dependencies for SRE mode..."
    npm install >/dev/null
  fi
fi

echo "Starting server on http://localhost:${PORT}"
echo "Open http://localhost:${PORT}/"
exec python -m uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
