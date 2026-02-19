# TTS WebSocket (Piper Medium + WhisperX + SRE)

This repo is configured to run one stack:
- TTS backend: `piper` (`en_US-lessac-medium`)
- Aligner: `whisperx`
- Math/LaTeX normalization: `sre` (`latex-to-speech` worker via Node)

## 1) Prerequisites

- Python 3.10+ recommended
- Node.js 18+ (required for SRE worker)
- `ffmpeg` available on PATH (recommended for WhisperX tooling)

## 2) Setup

From `tts_ws/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
```

## 3) Required Piper Medium Files

Place these files in `tts_ws/`:

- `en_US-lessac-medium.onnx`
- `en_US-lessac-medium.onnx.json`

If they are already present, skip this step.

## 4) Run Server (Piper + WhisperX + SRE)

```bash
export TTS_BACKEND=piper
export PIPER_VOICE="$PWD/en_US-lessac-medium.onnx"
export PIPER_CONFIG="$PWD/en_US-lessac-medium.onnx.json"
export ALIGNER=whisperx
export TTS_MATH_NORMALIZER=sre
export TTS_PROFILE=1

python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

## 5) Test

In a new terminal (same env):

```bash
source .venv/bin/activate
python test_client.py --text "The product of three and seven is \(3 \times 7 = 21\)."
```

Open UI:

- `http://127.0.0.1:8000/`

WebSocket endpoint:

- `ws://127.0.0.1:8000/tts`

## Troubleshooting

- `Backend init failed: piper-tts is not installed`
  - Re-run `pip install -r requirements.txt`
- `Aligner init failed`
  - Confirm `whisperx` and torch installed in `.venv`
- `Math normalizer init failed`
  - Confirm `npm install` was run and `node` is on PATH
- Very high TTFT
  - WhisperX alignment is expensive; this setup prioritizes alignment quality over minimum latency.
