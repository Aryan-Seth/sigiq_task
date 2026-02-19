from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RESULTS = EXP / "results"
FIGS = EXP / "figures"
REPORT_PATH = EXP / "POLISHED_REPORT.md"


MODE_FILES = {
    "off": RESULTS / "ttft_cases_large_off_r2.json",
    "rule": RESULTS / "ttft_cases_large_rule_r3.json",
    "sre": RESULTS / "ttft_cases_large_sre_r2.json",
}

ALIGNER_BASELINE_FILE = RESULTS / "ttft_cases_large_rule_r3.json"
ALIGNER_ON_FILE = RESULTS / "ttft_cases_large_rule_whisperx_r2_t8.json"
EXTENDED_BASELINE_FILE = RESULTS / "ttft_cases_extended_v1_rule_r1.json"
EXTENDED_ALIGNER_FILE = RESULTS / "ttft_cases_extended_v1_rule_whisperx_r1_t8.json"
PARETO_FILE = RESULTS / "pareto_backends_rule_r1_full.json"


@dataclass
class RowSet:
    rows: list[dict[str, Any]]
    valid_rows: list[dict[str, Any]]


def _as_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    if math.isnan(out) or math.isinf(out):
        return float("nan")
    return out


def _p50(values: list[float]) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return float("nan")
    n = len(clean)
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / float(len(clean))


def _fmt_ms(v: float) -> str:
    if math.isnan(v):
        return "-"
    return f"{v:.1f} ms"


def _fmt_pct(v: float) -> str:
    if math.isnan(v):
        return "-"
    return f"{v:.1f}%"


def _fmt_num(v: float, digits: int = 3) -> str:
    if math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def _load_rows(path: Path) -> RowSet:
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid: list[dict[str, Any]] = []
    for row in rows:
        ttft = _as_float(row.get("client_ttft_recv_ms"))
        timed_out = bool(row.get("receiver_timed_out", False))
        if not timed_out and not math.isnan(ttft):
            valid.append(row)
    return RowSet(rows=rows, valid_rows=valid)


def _length_p50_ttft(rs: RowSet) -> dict[int, float]:
    bucket: dict[int, list[float]] = {}
    for row in rs.valid_rows:
        length = int(row.get("chars", 0))
        bucket.setdefault(length, []).append(_as_float(row.get("client_ttft_recv_ms")))
    return {k: _p50(v) for k, v in sorted(bucket.items())}


def _component_summary(rs: RowSet) -> dict[str, float]:
    return {
        "client_ttft_p50": _p50([_as_float(r.get("client_ttft_recv_ms")) for r in rs.valid_rows]),
        "server_ttft_p50": _p50(
            [_as_float(r.get("server_ttft_audio_out_ms")) for r in rs.valid_rows]
        ),
        "model_p50": _p50([_as_float(r.get("server_model_compute_ms")) for r in rs.valid_rows]),
        "post_model_p50": _p50(
            [_as_float(r.get("server_post_model_to_audio_out_ms")) for r in rs.valid_rows]
        ),
        "aligner_p50": _p50([_as_float(r.get("server_gap_aligner_ms")) for r in rs.valid_rows]),
    }


def _category(case_name: str) -> str:
    if "_" not in case_name:
        return case_name
    return case_name.split("_", 1)[0]


def _coverage_by_category(rs: RowSet) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for row in rs.valid_rows:
        name = str(row.get("case", "unknown"))
        cov = 100.0 * _as_float(row.get("alignment_coverage"))
        bucket.setdefault(_category(name), []).append(cov)
    return {k: _p50(v) for k, v in sorted(bucket.items())}


def _plot_backend_pareto(
    summary: list[dict[str, Any]], frontier: list[dict[str, Any]], out: Path
) -> None:
    colors = {"piper": "#147AD6", "say": "#E86A33", "kokoro": "#1FA187", "dummy": "#9A9A9A"}
    plt.figure(figsize=(9, 5.5))

    for row in summary:
        backend = str(row.get("backend", "unknown"))
        variant = str(row.get("variant", "unknown"))
        x = _as_float(row.get("ttft_p50_ms"))
        y = _as_float(row.get("wer_mean"))
        if math.isnan(x) or math.isnan(y):
            continue
        plt.scatter(x, y, s=120, color=colors.get(backend, "#555555"), alpha=0.9)
        label = f"{backend}:{variant}"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    fpts = sorted(
        [
            (_as_float(r.get("ttft_p50_ms")), _as_float(r.get("wer_mean")))
            for r in frontier
            if not math.isnan(_as_float(r.get("ttft_p50_ms")))
            and not math.isnan(_as_float(r.get("wer_mean")))
        ],
        key=lambda t: t[0],
    )
    if len(fpts) >= 2:
        xs, ys = zip(*fpts)
        plt.plot(xs, ys, linestyle="--", linewidth=1.7, color="#111111", alpha=0.8, label="Pareto frontier")

    plt.title("TTFT vs ASR WER (Lower Is Better)")
    plt.xlabel("TTFT p50 (ms)")
    plt.ylabel("WER mean")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_mode_lengths(mode_length_ttft: dict[str, dict[int, float]], out: Path) -> None:
    colors = {"off": "#999999", "rule": "#147AD6", "sre": "#1FA187"}
    markers = {"off": "o", "rule": "s", "sre": "^"}
    plt.figure(figsize=(8.5, 5.2))
    for mode, vals in mode_length_ttft.items():
        xs = sorted(vals.keys())
        ys = [vals[x] for x in xs]
        if not xs:
            continue
        plt.plot(
            xs,
            ys,
            marker=markers.get(mode, "o"),
            linewidth=2,
            markersize=6,
            color=colors.get(mode, "#444444"),
            label=mode,
        )
    plt.title("TTFT p50 by Text Length and Math Normalization Mode")
    plt.xlabel("Text length (characters)")
    plt.ylabel("Client TTFT p50 (ms)")
    plt.grid(alpha=0.25)
    plt.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_aligner_breakdown(
    baseline: dict[str, float], aligner: dict[str, float], out: Path
) -> None:
    labels = ["rule (no aligner)", "rule + whisperx"]
    model_vals = np.array([baseline["model_p50"], aligner["model_p50"]], dtype=float)
    post_vals = np.array([baseline["post_model_p50"], aligner["post_model_p50"]], dtype=float)
    align_vals = np.array([0.0, aligner["aligner_p50"] if not math.isnan(aligner["aligner_p50"]) else 0.0], dtype=float)

    x = np.arange(len(labels))
    width = 0.55

    plt.figure(figsize=(8.6, 5.2))
    plt.bar(x, model_vals, width, label="Model compute", color="#147AD6")
    plt.bar(x, post_vals, width, bottom=model_vals, label="Post-model packaging", color="#4FA3E3")
    plt.bar(
        x,
        align_vals,
        width,
        bottom=(model_vals + post_vals),
        label="Aligner",
        color="#1FA187",
    )
    plt.xticks(x, labels)
    plt.ylabel("Server component latency p50 (ms)")
    plt.title("Server Latency Breakdown: Aligner Contribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_coverage_categories(
    cov_base: dict[str, float], cov_align: dict[str, float], out: Path
) -> None:
    cats = sorted(set(cov_base.keys()) | set(cov_align.keys()))
    base_vals = [_as_float(cov_base.get(c, float("nan"))) for c in cats]
    align_vals = [_as_float(cov_align.get(c, float("nan"))) for c in cats]

    x = np.arange(len(cats))
    width = 0.35
    plt.figure(figsize=(10.5, 5.4))
    plt.bar(x - width / 2, base_vals, width, label="rule (no aligner)", color="#147AD6")
    plt.bar(x + width / 2, align_vals, width, label="rule + whisperx", color="#1FA187")
    plt.ylim(0, 105)
    plt.xticks(x, cats)
    plt.ylabel("Alignment coverage p50 (%)")
    plt.title("Coverage by Prompt Category")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def build() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    mode_sets = {mode: _load_rows(path) for mode, path in MODE_FILES.items()}
    mode_length_ttft = {mode: _length_p50_ttft(rs) for mode, rs in mode_sets.items()}

    baseline = _component_summary(_load_rows(ALIGNER_BASELINE_FILE))
    aligner = _component_summary(_load_rows(ALIGNER_ON_FILE))

    cov_base = _coverage_by_category(_load_rows(EXTENDED_BASELINE_FILE))
    cov_align = _coverage_by_category(_load_rows(EXTENDED_ALIGNER_FILE))

    pareto = json.loads(PARETO_FILE.read_text(encoding="utf-8"))
    backend_summary = list(pareto.get("summary", []))
    frontier = list(pareto.get("pareto_frontier", []))

    fig1 = FIGS / "fig_01_backend_pareto.png"
    fig2 = FIGS / "fig_02_ttft_by_mode_length.png"
    fig3 = FIGS / "fig_03_aligner_breakdown.png"
    fig4 = FIGS / "fig_04_coverage_by_category.png"

    _plot_backend_pareto(backend_summary, frontier, fig1)
    _plot_mode_lengths(mode_length_ttft, fig2)
    _plot_aligner_breakdown(baseline, aligner, fig3)
    _plot_coverage_categories(cov_base, cov_align, fig4)

    best_frontier = frontier[0] if frontier else {}
    best_backend = f"{best_frontier.get('backend', '-')}/{best_frontier.get('variant', '-')}"
    best_ttft = _as_float(best_frontier.get("ttft_p50_ms"))
    best_wer = _as_float(best_frontier.get("wer_mean"))

    report = f"""# Polished Evaluation Report

_Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_

## Executive Summary
- Goal: maximize speech quality under strict TTFT constraints for a fully local (on-device) TTS websocket pipeline.
- Best low-latency operating point from current sweeps: **{best_backend}**.
- Best frontier point metrics: **TTFT p50 {_fmt_ms(best_ttft)}**, **WER mean {_fmt_num(best_wer)}**.
- Aligner (`whisperx`) materially increases TTFT: baseline server TTFT p50 {_fmt_ms(baseline["server_ttft_p50"])} vs aligner {_fmt_ms(aligner["server_ttft_p50"])}.

## Methodology
- Websocket benchmark driver: `benchmark_ttft.py`
- Quality proxy benchmark: `benchmark_model_pareto.py` (ASR WER/CER via `faster_whisper`)
- Input corpus: large mixed set (`experiments/cases_large_template.txt`) and extended category set (`experiments/cases_extended_v1.txt`)
- Chunking policy: ramp plan `4,8,32` with low inter-chunk delay
- Metrics:
  - `client_ttft_recv_ms` for end-user TTFT
  - `server_ttft_audio_out_ms` and component timings (`model`, `aligner`, `post-model`)
  - `alignment_coverage` (analyzed separately from quality proxy)

## Key Figures
### 1) Backend Pareto Frontier
![Backend Pareto](figures/fig_01_backend_pareto.png)

### 2) TTFT by Math Normalization Mode and Length
![TTFT by mode](figures/fig_02_ttft_by_mode_length.png)

### 3) Aligner Latency Component Breakdown
![Aligner breakdown](figures/fig_03_aligner_breakdown.png)

### 4) Alignment Coverage by Prompt Category
![Coverage categories](figures/fig_04_coverage_by_category.png)

## Findings
- `piper medium` is currently the strongest low-latency quality point.
- `kokoro` (local int8 ONNX path) improves WER/CER on tested prompts but with much higher TTFT on this hardware/runtime path.
- `say` backend is simple and fully local but underperforms in measured intelligibility and latency.
- `whisperx` alignment is useful for timing fidelity but should be treated as a separate, optional latency tier.

## Product Recommendations
- Default production profile:
  - backend: `piper` + `en_US-lessac-medium.onnx`
  - math mode: `rule`
  - aligner: off by default; optional `whisperx` when subtitle timing precision is required.
- Performance profile variants:
  - **Low-latency mode**: no aligner, small/ramped chunks.
  - **Rich-caption mode**: aligner on, higher receiver idle timeout budget.

## Reproducibility
Generate this report and figures:
```bash
.venv/bin/python experiments/build_polished_report.py
```

Re-run local backend Pareto sweep:
```bash
.venv/bin/python benchmark_model_pareto.py \\
  --uri ws://127.0.0.1:8057/tts \\
  --math-normalizer rule \\
  --chunk-mode ramp --chunk-plan 4,8,32 --delay 0.01 \\
  --runs 1 \\
  --piper-voices en_US-lessac-low.onnx,en_US-lessac-medium.onnx,en_US-lessac-high.onnx \\
  --say-voices "Eddy (English (US))" --say-rate 190 \\
  --kokoro-voice-list af_sarah \\
  --kokoro-model models/kokoro/kokoro-v1.0.int8.onnx \\
  --kokoro-voices-bin models/kokoro/voices-v1.0.bin \\
  --json-out experiments/results/pareto_backends_rule_r1_full.json
```
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote figures: {fig1}, {fig2}, {fig3}, {fig4}")


if __name__ == "__main__":
    build()
