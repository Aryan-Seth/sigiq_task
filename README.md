# TTS WebSocket Scaffold

Minimal ElevenLabs-style TTS WebSocket clone scaffold with streaming audio + alignment.

## Quickstart

```bash
cd tts_ws
./run.sh
```

In another terminal:

```bash
cd tts_ws
source .venv/bin/activate
python test_client.py
```

The client writes `output.wav` (44.1 kHz, mono, 16-bit) in `tts_ws/`.

## One-Click Run (No Model Downloads)

`run.sh` is the intended one-command entrypoint for this repo:

```bash
cd tts_ws
chmod +x run.sh
./run.sh
```

What it does:
- creates/uses `.venv`
- installs runtime Python dependencies from `requirements.runtime.txt`
- uses bundled Piper voice files already in this repo (no model download)
- starts `uvicorn` server at `http://localhost:8000/`

Useful overrides:

```bash
PORT=9000 ./run.sh
TTS_MATH_NORMALIZER=rule ./run.sh
TTS_MATH_NORMALIZER=sre ./run.sh   # requires Node + npm
PIPER_VOICE=/absolute/path/to/voice.onnx ./run.sh
TTS_BACKEND=say ./run.sh
TTS_BACKEND=kokoro KOKORO_MODEL=/abs/path/kokoro-v1.0.int8.onnx KOKORO_VOICES=/abs/path/voices-v1.0.bin ./run.sh
TTS_BACKEND=cosyvoice COSYVOICE_REPO_DIR=/abs/path/CosyVoice COSYVOICE_MODEL_DIR=FunAudioLLM/CosyVoice2-0.5B ./run.sh
```

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

## Additional Local Backends (On-Device)

### macOS `say` backend
Uses Apple local TTS voices (no cloud, no model download).

```bash
export TTS_BACKEND=say
export SAY_VOICE="Eddy (English (US))"   # any voice from: say -v '?'
export SAY_RATE=190
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### Kokoro ONNX backend
Runs fully local ONNX inference (no cloud).

Download model files once:
```bash
mkdir -p models/kokoro
curl -L -o models/kokoro/kokoro-v1.0.int8.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -L -o models/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Run:
```bash
export TTS_BACKEND=kokoro
export KOKORO_MODEL=$PWD/models/kokoro/kokoro-v1.0.int8.onnx
export KOKORO_VOICES=$PWD/models/kokoro/voices-v1.0.bin
export KOKORO_VOICE=af_sarah
export KOKORO_LANG=en-us
export KOKORO_SPEED=1.0
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### CosyVoice backend (high quality, GPU-friendly)
This backend uses FunAudioLLM/CosyVoice and can run locally or on Colab T4.

Required env vars:
- `COSYVOICE_REPO_DIR`: local checkout of `FunAudioLLM/CosyVoice`
- `COSYVOICE_MODEL_DIR`: local model path or repo id (default `FunAudioLLM/CosyVoice2-0.5B`)
- `COSYVOICE_MODE`: `sft` (default), `zero_shot`, `cross_lingual`, or `instruct2`

Run with SFT mode:
```bash
export TTS_BACKEND=cosyvoice
export COSYVOICE_REPO_DIR=/absolute/path/to/CosyVoice
export COSYVOICE_MODEL_DIR=FunAudioLLM/CosyVoice2-0.5B
export COSYVOICE_MODE=sft
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

For `zero_shot`, `cross_lingual`, or `instruct2`, also set:
- `COSYVOICE_PROMPT_WAV=/path/to/reference_16k.wav`
- `COSYVOICE_PROMPT_TEXT=...` (required in `zero_shot`)
- `COSYVOICE_INSTRUCT_TEXT=...` (required in `instruct2`)

## Math Notation Handling

The server can normalize mathematical notation into spoken text before synthesis.
This is enabled by default (`TTS_MATH_NORMALIZE=1`).

Examples:
- `\(3 \times 7 = 21\)` -> `3 times 7 equals 21`
- `\(a^2 + b^2 = c^2\)` -> `a squared plus b squared equals c squared`
- `\(\frac{d}{dx} e^x = e^x\)` -> `d over d x e to the power of x equals e to the power of x`

Modes:
- `TTS_MATH_NORMALIZER=rule` (default): in-process Python rule-based verbalization (lowest overhead).
- `TTS_MATH_NORMALIZER=sre`: uses `latex-to-speech` (SRE) via a persistent Node worker.
- `TTS_MATH_NORMALIZER=off`: disables math verbalization.

SRE options (only in `sre` mode):
- `TTS_SRE_DOMAIN` (default: `clearspeak`)
- `TTS_SRE_STYLE` (optional)
- `TTS_SRE_LOCALE` (optional)

Disable it with:
```bash
export TTS_MATH_NORMALIZE=0
```

## Basic Web UI

After starting the server, open:

```text
http://localhost:8000/
```

The page streams chunked text over WebSocket, plays returned PCM chunks, and renders live subtitles from alignment data.
It also shows:
- live client metrics (`TTFT`, total latency, chars/sec, alignment coverage, avg misalignment)
- server profile metrics (`model_compute_ms`, `post_model_to_audio_out_ms`, queue/send gaps)
- progressive LaTeX rendering of source text in sync with alignment indices.
- chunk scheduling controls:
  - `Fixed`: constant chunk size
  - `Ramp`: increasing chunk sizes (`chunk_start -> ... -> chunk_max`) with multiplier `growth`

## Colab T4: CosyVoice WebSocket Server

In Colab (GPU runtime), run:

```bash
%%bash
set -euo pipefail
nvidia-smi

# 1) Clone this repo to /content/sigiq_takehome (or upload it there) before running.
if [[ ! -d /content/sigiq_takehome/tts_ws ]]; then
  echo "Place this repo at /content/sigiq_takehome first." >&2
  exit 1
fi

# 2) Clone CosyVoice and submodules.
if [[ ! -d /content/CosyVoice ]]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /content/CosyVoice
else
  cd /content/CosyVoice
  git submodule update --init --recursive
fi

# 3) Install dependencies.
python -m pip install -U pip
python -m pip install -r /content/CosyVoice/requirements.txt
python -m pip install -r /content/sigiq_takehome/tts_ws/requirements.runtime.txt
```

Or use the helper script:

```bash
cd /content/sigiq_takehome/tts_ws
bash colab_t4_cosyvoice.sh install
```

Start the websocket server:

```bash
%%bash
set -euo pipefail
cd /content/sigiq_takehome/tts_ws
export TTS_BACKEND=cosyvoice
export COSYVOICE_REPO_DIR=/content/CosyVoice
export COSYVOICE_MODEL_DIR=FunAudioLLM/CosyVoice2-0.5B
export COSYVOICE_MODE=sft
export TTS_PROFILE=1
nohup python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 > /content/tts_ws_server.log 2>&1 &
sleep 2
tail -n 30 /content/tts_ws_server.log || true
```

Or:

```bash
cd /content/sigiq_takehome/tts_ws
bash colab_t4_cosyvoice.sh start
```

Quick websocket smoke test from Colab:

```python
import asyncio, json, websockets
async def _t():
    async with websockets.connect("ws://127.0.0.1:8000/tts", max_size=None) as ws:
        await ws.send(json.dumps({"text":" ","flush":False}))
        await ws.send(json.dumps({"text":"Hello from CosyVoice on T4.","flush":False}))
        await ws.send(json.dumps({"text":"", "flush":True}))
        await ws.send(json.dumps({"text":"", "flush":False}))
        msg = await ws.recv()
        print(str(msg)[:200])
asyncio.run(_t())
```

## Notes

- WebSocket route: `ws://localhost:8000/tts`
- Backend selection: `TTS_BACKEND=dummy|piper|say|kokoro|cosyvoice` (default `dummy`)
- Alignment: optional `ALIGNER=whisperx` (requires `whisperx` + torch and a downloaded align model)
- `TTS_PROFILE` defaults to `1`; set `TTS_PROFILE=0` to disable server profiling payloads/logs.
- `benchmark_ttft.py` supports ramped chunks via:
  - `--chunk-mode ramp --chunk-start 4 --chunk-max 32 --chunk-growth 2`
  - or explicit schedule: `--chunk-mode ramp --chunk-plan 4,8,32` (holds at `32`)
- `benchmark_ttft.py` also supports multi-case corpora via `--cases-file <path>` (one case per line or `name<TAB>text`).
- Receiver idle timeout defaults are adaptive in benchmarks:
  - no aligner: `2.0s`
  - aligner enabled: `8.0s`
  - override with `--receiver-idle-timeout`.
- `benchmark_model_pareto.py` supports local cross-backend sweeps with:
  - `--piper-voices ...`
  - `--say-voices ... --say-rate ...`
  - `--kokoro-voice-list ... --kokoro-model ... --kokoro-voices-bin ...`

## Polished Report + Figures

Generate a publication-ready report and charts from latest experiment JSON files:

```bash
.venv/bin/python experiments/build_polished_report.py
```

Outputs:
- Report: `experiments/POLISHED_REPORT.md`
- Figures:
  - `experiments/figures/fig_01_backend_pareto.png`
  - `experiments/figures/fig_02_ttft_by_mode_length.png`
  - `experiments/figures/fig_03_aligner_breakdown.png`
  - `experiments/figures/fig_04_coverage_by_category.png`
