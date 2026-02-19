# Latency Playbook

This project should be optimized with a strict decomposition mindset, not by tweaking random knobs.

## 1) Decompose TTFT First

For each run, break TTFT into:

- `gap_buffer_to_synth_ms` (pre-model wait)
- `gap_math_normalize_ms` (text normalization)
- `model_compute_ms` (backend inference)
- `gap_aligner_ms` (forced alignment, optional)
- `packaging_only_ms` (resample/chunk/encode payload prep)
- `gap_audio_out_to_send_ms` (websocket send overhead)

Rule: optimize the largest stable term first.

## 2) Keep Two Profiles

Always maintain both:

1. **Latency profile**: `ALIGNER=none`, ramp chunks (`4,8,32`), low delay.
2. **Quality/alignment profile**: `ALIGNER=whisperx`, same text set.

Never compare runs with different chunking/text and conclude backend wins.

## 3) Fix Inputs Before Comparing Models

Use fixed:

- same corpus (`experiments/cases_*.txt`)
- same chunk plan + delay
- same voice/model config
- warmed runs counted separately from cold runs

Track p50/p95, not single-run anecdotes.

## 4) Interpret Aligner Cost Correctly

If aligner is on, post-model time includes aligner. Use:

- `post_model_total_incl_aligner_ms` for end-to-end post-model
- `gap_aligner_ms` for aligner
- `packaging_only_ms` for non-aligner overhead

If `packaging_only_ms` is small and `gap_aligner_ms` is large, do not chase serialization optimizations.

## 5) Optimization Priority Order

1. Reduce pre-model wait (`TTS_MIN_CHARS_TO_SYNTH`, chunk plan, delay).
2. Improve model TTFT (backend/voice/model settings).
3. Decide aligner policy (off, asynchronous, or on-demand) based on product need.
4. Only then tune packaging/network overhead.

## 6) Quality Strategy

Quality is not a single scalar in this repo. Evaluate:

- subjective MOS-style listening on a fixed prompt set
- alignment coverage and average misalignment
- intelligibility on math-heavy text

Use a Pareto view: TTFT vs quality, then choose operating points by product mode (realtime vs accurate captions).
