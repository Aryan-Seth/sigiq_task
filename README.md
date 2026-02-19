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

## How We Got Here (Experiments Summary)

We evaluated multiple backend/normalizer/aligner combinations before fixing this default stack.

- Backend sweeps: `benchmark_model_pareto.py`
- TTFT/latency sweeps: `benchmark_ttft.py`
- Math normalization sweeps: `benchmark_math_normalizers.py`
- Alignment quality checks: `align_eval.py`
- Aggregated outputs and plots: `experiments/results/` and `experiments/figures/`

Selection rationale for `piper + whisperx + sre`:
- `piper` gave the best local throughput and low TTFT on this hardware.
- `whisperx` provided the strongest character-level timing quality.
- `sre` improved spoken output for LaTeX/math expressions over rule-only normalization.

## Profiler Setup

Server-side profiler is built into the websocket orchestrator and emitted per session as:
- `[tts-profile]` (human-readable)
- `[tts-profile-json]` (structured metrics)

Key metrics include:
- `ttft_audio_out_ms`
- `model_compute_ms`
- `gap_math_normalize_ms`
- `gap_aligner_ms`
- `packaging_only_ms`
- `post_model_total_incl_aligner_ms`

Enable/disable profiling:

```bash
export TTS_PROFILE=1
```

## Code Structure

- `app/server.py`: startup/init and websocket route wiring.
- `app/orchestrator.py`: receive/synthesize/send loops and profiler markers.
- `app/backends/`: backend implementations (`piper`, `kokoro`, etc.).
- `app/math_normalizer.py`: math normalization mode selection (`rule`/`sre`/`off`).
- `app/text_normalize.py`: rule-based text/math normalization helpers.
- `app/aligner.py`: aligner factory and WhisperX integration.
- `app/tools/sre_worker.mjs`: Node worker for LaTeX-to-speech conversion.
- `app/static/index.html`: streaming UI with metrics/captions.

## Alternatives Considered

- `kokoro + whisperx + sre`:
  - Better voice quality in some cases.
  - Higher TTFT on local CPU and slower first chunk in our tests.
- `piper + none + rule`:
  - Fastest latency profile.
  - Lower alignment quality and weaker spoken math handling.
- `dummy` backend:
  - Useful for transport/protocol debugging only, not meaningful for quality evaluation.

## Troubleshooting

- `Backend init failed: piper-tts is not installed`
  - Re-run `pip install -r requirements.txt`
- `Aligner init failed`
  - Confirm `whisperx` and torch installed in `.venv`
- `Math normalizer init failed`
  - Confirm `npm install` was run and `node` is on PATH
- Very high TTFT
  - WhisperX alignment is expensive; this setup prioritizes alignment quality over minimum latency.
