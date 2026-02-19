# TTS Sweeps, Experiments, Ideas, and Results

_Last updated: 2026-02-10 PST_

## Scope
This document tracks latency/alignment experiments for the `tts_ws` backend and the current optimization backlog.

## Metrics Tracked
- `client_ttft_recv_ms`: client first non-whitespace text send -> first audio chunk received
- `server_ttft_audio_out_ms`: server first non-whitespace text -> first audio chunk ready
- `alignment_coverage`: matched chars / input chars
- `alignment_abs_tail_error_ms`: absolute difference between alignment tail and audio timeline tail
- `model_compute_ms`, `post_model_to_audio_out_ms`, `gap_buffer_to_synth_ms`, `gap_audio_out_to_send_ms`
- UI-only derived metric: `avg_misalignment_ms = alignment_abs_tail_error_ms / alignment_total_reported_chars`

## Completed Sweeps

### 1) Math normalizer mode sweep (`off`, `rule`, `sre`)
Command:
```bash
.venv/bin/python benchmark_math_normalizers.py \
  --backend piper --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-modes off,rule,sre --runs 2 \
  --json-out experiments/results/math_modes_r2.json
```
Result summary (all cases combined):
- `off`: client TTFT p50 `279.6 ms`, server TTFT p50 `300.8 ms`
- `rule`: client TTFT p50 `208.4 ms`, server TTFT p50 `230.2 ms`
- `sre`: client TTFT p50 `312.1 ms`, server TTFT p50 `364.1 ms`

Takeaway:
- `rule` mode is best latency/quality tradeoff for current stack.
- `sre` gives different phrasing but currently adds startup/IPC overhead.

### 2) TTFT vs text length (rule mode, baseline knobs)
Command:
```bash
.venv/bin/python benchmark_ttft.py \
  --uri ws://127.0.0.1:8013/tts \
  --backend piper --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-normalizer rule --lengths 50,200,800 --runs-per-length 3 --warmup-runs 1 \
  --chunk-size 22 --delay 0.025 \
  --json-out experiments/results/ttft_rule_baseline_r3_port8013.json
```
Key p50 results:
- len 50: client `46.3 ms`, server `44.7 ms`, alignment coverage `100.0%`
- len 200: client `47.1 ms`, server `45.7 ms`, alignment coverage `99.5%`
- len 800: client `44.5 ms`, server `43.1 ms`, alignment coverage `99.2%`

Takeaway:
- TTFT is mostly length-invariant for Piper once first chunking threshold is met.

### 3) TTFT vs text length (rule mode, aggressive low-latency knobs)
Command:
```bash
.venv/bin/python benchmark_ttft.py \
  --uri ws://127.0.0.1:8015/tts \
  --backend piper --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-normalizer rule --lengths 50,200,800 --runs-per-length 3 --warmup-runs 1 \
  --chunk-size 22 --delay 0.025 \
  --min-chars-to-synth 1 --idle-trigger-ms 5 --first-segment-limit 40 \
  --json-out experiments/results/ttft_rule_lowlat_r3_port8015.json
```
Key p50 results:
- len 50: client `45.8 ms`, server `44.1 ms`
- len 200: client `51.3 ms`, server `49.7 ms`
- len 800: client `46.4 ms`, server `45.0 ms`

Takeaway:
- For current workload this knob set is not consistently better than baseline; benefit is small/noisy.

### 4) Math normalization overhead check (`off` vs `rule` under low-latency knobs)
Command:
```bash
.venv/bin/python benchmark_ttft.py \
  --uri ws://127.0.0.1:8014/tts \
  --backend piper --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-normalizer off --lengths 50,200,800 --runs-per-length 3 --warmup-runs 1 \
  --chunk-size 22 --delay 0.025 \
  --min-chars-to-synth 1 --idle-trigger-ms 5 --first-segment-limit 40 \
  --json-out experiments/results/ttft_off_lowlat_r3_port8014.json
```
Key p50 results:
- len 50: client `41.5 ms`, server `40.2 ms`
- len 200: client `47.0 ms`, server `45.6 ms`
- len 800: client `45.9 ms`, server `44.4 ms`

Takeaway:
- Turning math normalization off saves a few ms; `rule` overhead is modest.

### 5) Chunk-size sensitivity sweep (len=200, rule mode)
Commands:
```bash
# smaller chunks
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8016/tts --backend piper \
  --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-normalizer rule --lengths 200 --runs-per-length 3 --warmup-runs 1 \
  --chunk-size 8 --delay 0.025 --min-chars-to-synth 1 --idle-trigger-ms 5 --first-segment-limit 40 \
  --json-out experiments/results/ttft_rule_chunk8_len200_r3_port8016.json

# larger chunks
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8017/tts --backend piper \
  --piper-voice en_US-lessac-low.onnx --piper-config en_US-lessac-low.onnx.json \
  --math-normalizer rule --lengths 200 --runs-per-length 3 --warmup-runs 1 \
  --chunk-size 64 --delay 0.025 --min-chars-to-synth 1 --idle-trigger-ms 5 --first-segment-limit 40 \
  --json-out experiments/results/ttft_rule_chunk64_len200_r3_port8017.json
```
Key p50 results:
- `chunk-size=8`: client TTFT `45.8 ms`, server TTFT `43.6 ms`
- `chunk-size=64`: client TTFT `194.3 ms`, server TTFT `190.2 ms`

Takeaway:
- Large input chunks with fixed inter-chunk delay significantly hurt TTFT.

## Quality/Correctness Work Completed
- Fixed subtitle tripling (`TTThhheee`) by changing alignment window split to emit chars once by start-time.
- Fixed math subtitle truncation caused by over-aggressive UI dedupe on `char_indices`.
- Added progressive LaTeX reveal panel synced to alignment indices.
- Added UI metrics dashboard with client + server profiling fields.

## Optimization Ideas Backlog

### A) Backend/model
- Evaluate higher-quality TTS backend options behind same WebSocket contract.
- Add GPU path (`PIPER_USE_CUDA`, provider selection) and benchmark CPU vs GPU TTFT + RTF.
- Keep warm pools per voice/model to reduce cold-start variance.

### B) Pipeline latency
- Adaptive chunk gating: trigger synth on punctuation + low char threshold + idle budget.
- Reduce post-model work (`resample`, packaging) with vectorized/streaming path.
- Parallelize aligner/packaging where backend can stream partial chunks.

### C) Alignment quality
- Add explicit spoken-index stream for UI reveal to decouple display from source-index mapping.
- Evaluate WhisperX alignment path on a fixed math-heavy eval set for coverage/tail error tradeoffs.

### D) Experiment rigor
- Increase runs per point (>=10) for stable p95/p99.
- Add fixed-seed text corpus buckets (short/medium/long/math-heavy).
- Persist benchmark metadata: commit hash, machine, CPU/GPU, backend/env vars.

## Recommended Current Default
- Backend: `piper` (current voice)
- Math normalizer: `rule`
- Chunking for low TTFT: small client chunks (`~8-24`) with minimal delay
- Keep profiling enabled for tuning (`TTS_PROFILE=1`)

## Text Sets Covered So Far

Prompt sets exercised in analysis:
- `sample_text.txt` repeated/truncated to target lengths (legacy TTFT length sweeps).
- `benchmark_math_normalizers.py` fixed math cases (product, Pythagorean theorem, derivative).
- `experiments/math_corpus.txt` math-heavy corpus for module-inclusive profiling.
- `benchmark_model_pareto.py` default mixed set (`plain`, `math`, `math2`).
- `experiments/cases_large_template.txt` multi-domain set:
  - `plain_1`, `plain_2`, `math_1`, `math_2`, `math_3`, `units_1`, `mixed_1`.

## Large Corpus Sweeps (7 Cases x 3 Lengths)

Commands:
```bash
# rule
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8041/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner none --math-normalizer rule \
  --cases-file experiments/cases_large_template.txt \
  --lengths 120,240,480 --runs-per-length 3 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --json-out experiments/results/ttft_cases_large_rule_r3.json

# off
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8042/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner none --math-normalizer off \
  --cases-file experiments/cases_large_template.txt \
  --lengths 120,240,480 --runs-per-length 2 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --json-out experiments/results/ttft_cases_large_off_r2.json

# sre
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8043/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner none --math-normalizer sre \
  --cases-file experiments/cases_large_template.txt \
  --lengths 120,240,480 --runs-per-length 2 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --json-out experiments/results/ttft_cases_large_sre_r2.json
```

TTFT p50 by length:
- `rule`: 120 `58.4 ms`, 240 `63.3 ms`, 480 `61.6 ms`
- `off`: 120 `63.4 ms`, 240 `62.4 ms`, 480 `62.1 ms`
- `sre`: 120 `59.8 ms`, 240 `60.5 ms`, 480 `58.1 ms`

Notes:
- `math_norm_p50` remains effectively `0.0 ms` in all three modes; TTFT is still mostly model-side.
- Lower alignment coverage for `rule/sre` on math prompts is expected with transformed text/index mapping.

## Extended Corpus Sweep (25 Cases x 2 Lengths)

Extended case file:
- `experiments/cases_extended_v1.txt` (plain, math, units, money, dates, abbreviations, mixed)

Run:
```bash
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8044/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner none --math-normalizer rule \
  --cases-file experiments/cases_extended_v1.txt \
  --lengths 160,320 --runs-per-length 1 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --json-out experiments/results/ttft_cases_extended_v1_rule_r1.json
```

Result summary:
- len 160: client TTFT p50 `62.4 ms`, server TTFT p50 `61.2 ms`, alignment coverage p50 `98.1%`
- len 320: client TTFT p50 `60.4 ms`, server TTFT p50 `59.1 ms`, alignment coverage p50 `95.6%`

## Aligner-Enabled Sweeps Added

### 1) Large corpus with aligner (`whisperx`)

Rule-normalized run:
```bash
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8047/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner whisperx --math-normalizer rule \
  --cases-file experiments/cases_large_template.txt \
  --lengths 120,240 --runs-per-length 2 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --receiver-idle-timeout 8.0 \
  --json-out experiments/results/ttft_cases_large_rule_whisperx_r2_t8.json
```

Off-normalized comparison:
```bash
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8048/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner whisperx --math-normalizer off \
  --cases-file experiments/cases_large_template.txt \
  --lengths 120,240 --runs-per-length 1 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --receiver-idle-timeout 8.0 \
  --json-out experiments/results/ttft_cases_large_off_whisperx_r1_t8.json
```

Summary:
- `rule + whisperx`: client TTFT p50 `135.3 ms`, server TTFT p50 `132.2 ms`, aligner p50 `77.9 ms`
- `off + whisperx`: client TTFT p50 `134.9 ms`, server TTFT p50 `132.0 ms`, aligner p50 `77.4 ms`
- Compared to no-aligner runs (`~60-63 ms` TTFT), enabling aligner adds about `+70-75 ms` TTFT on this stack.

### 2) Extended corpus with aligner (`whisperx`)

Run:
```bash
.venv/bin/python benchmark_ttft.py --uri ws://127.0.0.1:8049/tts \
  --backend piper --piper-voice en_US-lessac-medium.onnx --piper-config en_US-lessac-medium.onnx.json \
  --aligner whisperx --math-normalizer rule \
  --cases-file experiments/cases_extended_v1.txt \
  --lengths 160 --runs-per-length 1 --warmup-runs 1 \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --receiver-idle-timeout 8.0 \
  --json-out experiments/results/ttft_cases_extended_v1_rule_whisperx_r1_t8.json
```

Summary:
- len 160: client TTFT p50 `140.4 ms`, server TTFT p50 `137.9 ms`
- aligner p50 `79.4 ms`, alignment coverage p50 `98.8%`
- same corpus without aligner was ~`61.6 ms` client TTFT p50.

Operational note:
- `benchmark_ttft.py` now auto-uses a larger receiver idle timeout when aligner is enabled (8.0s default vs 2.0s without aligner), to prevent false timeouts during stricter aligner runs.

## Module-Inclusive Latency Breakdown (Math + Alignment)

To explicitly include the math and alignment modules in TTFT decomposition, these runs were executed on a math-heavy corpus (`experiments/math_corpus.txt`) with ramped chunking (`4,8,32`) and 10 ms inter-chunk delay.

### A) Math module cost (`ALIGNER=none`)
Files:
- `experiments/results/ttft_math_off_r2_port8023.json`
- `experiments/results/ttft_math_rule_r2_port8024.json`
- `experiments/results/ttft_math_sre_r2_port8025.json`

P50 component summary:
- `math=off`: server TTFT `92.1 ms`, `math_normalize` `0.005 ms`, model `80.2 ms`
- `math=rule`: server TTFT `85.0 ms`, `math_normalize` `0.020 ms`, model `72.9 ms`
- `math=sre`: server TTFT `42.2 ms`, `math_normalize` `0.019 ms`, model `30.5 ms`

Interpretation:
- Measured math-normalization compute itself is tiny (`~0.005-0.020 ms`) in this path.
- TTFT differences are dominated by downstream model behavior (different normalized text content changes first-chunk inference cost).

### B) Alignment module cost (`ALIGNER=whisperx`)
Files:
- `experiments/results/ttft_math_rule_whisperx_r2_port8028.json`
- `experiments/results/ttft_math_off_whisperx_r2_port8032.json`

P50 component summary:
- `math=rule + whisperx`: server TTFT `180.7 ms`, model `48.8 ms`, aligner `119.7 ms`
- `math=off + whisperx`: server TTFT `150.2 ms`, model `41.8 ms`, aligner `96.5 ms`

Interpretation:
- With WhisperX enabled, aligner time becomes the largest TTFT contributor (`~61-66%` of server TTFT).
- This validates treating alignment as a separate post-hoc module for low-latency serving.

## Pareto Sweep: TTFT vs Quality (Alignment Separate)

Quality proxy used here is ASR intelligibility (WER/CER from `faster_whisper`) and is intentionally separate from alignment metrics.

Script:
- `benchmark_model_pareto.py`

Run:
```bash
.venv/bin/python benchmark_model_pareto.py \
  --uri ws://127.0.0.1:8033/tts \
  --math-normalizer rule \
  --chunk-mode ramp --chunk-plan 4,8,32 \
  --runs 2 \
  --json-out experiments/results/pareto_models_rule_r2_resampled.json
```

Compared “models” (voices):
- `en_US-lessac-low.onnx`
- `en_US-lessac-medium.onnx`
- `en_US-lessac-high.onnx`

Summary:
- `low`: TTFT p50 `65.5 ms`, WER mean `0.180`, CER mean `0.080`
- `medium`: TTFT p50 `66.6 ms`, WER mean `0.092`, CER mean `0.045`
- `high`: TTFT p50 `391.5 ms`, WER mean `0.095`, CER mean `0.050`

Pareto frontier:
- `low` (best TTFT)
- `medium` (near-best TTFT, substantially better quality)

Practical choice:
- Default to `medium` when quality matters and +1 ms TTFT is acceptable.
- Use `low` for strict latency targets where quality drop is acceptable.

## Local Backend Sweep (Mac Silicon, On-Device Only)

Constraint:
- No cloud TTS APIs. All candidates run locally on the test machine.

Backends evaluated:
- `piper` (voices: low/medium/high)
- `say` (macOS voice: `Eddy (English (US))`)
- `kokoro` (`af_sarah`, local ONNX `kokoro-v1.0.int8.onnx`)

Run:
```bash
.venv/bin/python benchmark_model_pareto.py \
  --uri ws://127.0.0.1:8054/tts \
  --math-normalizer rule \
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \
  --runs 1 \
  --piper-voices en_US-lessac-low.onnx,en_US-lessac-medium.onnx,en_US-lessac-high.onnx \
  --say-voices "Eddy (English (US))" --say-rate 190 \
  --kokoro-voice-list af_sarah \
  --kokoro-model models/kokoro/kokoro-v1.0.int8.onnx \
  --kokoro-voices-bin models/kokoro/voices-v1.0.bin \
  --json-out experiments/results/pareto_backends_rule_r1_full.json
```

Summary:
- `piper::en_US-lessac-low.onnx`: TTFT p50 `70.2 ms`, WER mean `0.243`, CER mean `0.109`
- `piper::en_US-lessac-medium.onnx`: TTFT p50 `69.8 ms`, WER mean `0.089`, CER mean `0.046`
- `piper::en_US-lessac-high.onnx`: TTFT p50 `302.8 ms`, WER mean `0.099`, CER mean `0.062`
- `say::Eddy (English (US))`: TTFT p50 `413.9 ms`, WER mean `0.283`, CER mean `0.212`
- `kokoro::af_sarah` (int8): TTFT p50 `1436.2 ms`, WER mean `0.075`, CER mean `0.043`

Pareto frontier in this run:
- `piper::en_US-lessac-medium.onnx` (best low-latency point)
- `kokoro::af_sarah` (best WER/CER, but TTFT too high for low-latency target)

Current recommendation for “maximum quality at same TTFT”:
- For sub-100 ms TTFT target: keep `piper medium`.
- `piper high` was not Pareto-dominant in this run (higher TTFT with slightly worse WER than medium).
- Do not use current local `kokoro` path for low-latency serving unless TTFT >1 s is acceptable.
