# TTS WebSocket Scaffold

Minimal ElevenLabs-style TTS WebSocket clone scaffold with streaming audio + alignment.

## Quickstart

```bash
cd tts_ws
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run server
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd tts_ws
source .venv/bin/activate
python test_client.py
```

The client writes `output.wav` (44.1 kHz, mono, 16-bit) in `tts_ws/`.

To use real text:

```bash
python test_client.py --text "Your text here."
```

Or from a file:

```bash
python test_client.py --file /path/to/text.txt
```

Sample file:

```bash
python test_client.py --file sample_text.txt
```

Metrics (chars as tokens):

```bash
python test_client.py --file /path/to/text.txt --runs 50
```

Alignment evaluation (requires whisperx + torch):

```bash
# run server first
python align_eval.py --file sample_text.txt --chunk-size 512 --delay 0
```

## Setup and running with `uv`

`uv` gives fast, repeatable installs without touching global Python.

```bash
# install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /Users/aryanseth/sigiq_takehome/tts_ws

# create and populate a virtual env with project deps
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# run the server (dummy backend)
uvicorn app.server:app --host 0.0.0.0 --port 8000

# in another shell, same env, run the client
python test_client.py --text "Hello streaming TTS"
```

### Piper backend
```bash
export TTS_BACKEND=piper
export PIPER_VOICE=/absolute/path/to/en_US-lessac-medium.onnx  # adjust to your voice file
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### WhisperX alignment (optional, slower TTFT)
Requires torch/torchaudio compatible with your platform/GPU:
```bash
# example CPU install; pick wheels from https://pytorch.org/get-started/locally/
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install whisperx

export ALIGNER=whisperx
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Then run the client as usual. For alignment evaluation:
```bash
python align_eval.py --file sample_text.txt --chunk-size 512 --delay 0
```

## Piper backend

Install Piper and download a voice:

```bash
pip install piper-tts
python -m piper.download_voices en_US-lessac-medium
```

Set the voice path and run the server:

```bash
export TTS_BACKEND=piper
export PIPER_VOICE=/path/to/en_US-lessac-medium.onnx
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Optional tuning via env vars: `PIPER_CONFIG`, `PIPER_VOLUME`, `PIPER_LENGTH_SCALE`,
`PIPER_NOISE_SCALE`, `PIPER_NOISE_W_SCALE`, `PIPER_NORMALIZE_AUDIO`, `PIPER_USE_CUDA`.

## Notes

- WebSocket route: `ws://localhost:8000/tts`
- Backend selection: set `TTS_BACKEND=dummy` (default)
- Alignment: optional `ALIGNER=whisperx` (requires `whisperx` + torch and a downloaded align model)
