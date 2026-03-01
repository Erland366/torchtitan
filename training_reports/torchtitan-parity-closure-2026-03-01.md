# Report: torchtitan-parity-closure

**Date:** 2026-03-01
**Author:** codex
**Status:** Completed

## Objective

Close the parity loop for nanoVLM porting by validating three hard requirements on paired runs:
- Torchtitan is faster than `nanoVLM_main`.
- Torchtitan peak VRAM is lower.
- Loss is practically the same under exact data-stream pairing.

## Setup

### Environment
- Hardware: `1x A100-SXM4-40GB`
- Runtime: TorchTitan (`--module nanoVLM`) vs `nanoVLM_main`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming)

### Validation Method

- Paired runs executed with `scripts/nanovlm_parity_benchmark.py`
- Dataset trace enabled to compare per-microbatch fingerprints
- Acceptance based on exact data alignment plus bounded step-wise loss deltas

## Experiments

### Run 1: Vanilla final dataset-trace check

**Output directory:** `outputs/nanovlm_parity_benchmarks/vanilla-datasettrace-vanilla-fixedwarmup-20260301`

| Metric | Value |
|--------|-------|
| Baseline run | `pcks21h8` |
| Torchtitan run | `8q1766hh` |
| Baseline elapsed | `141.36s` |
| Torchtitan elapsed | `117.37s` |
| Baseline peak VRAM | `26169 MiB` |
| Torchtitan peak VRAM | `20001 MiB` |
| Loss mean abs diff | `0.00137` |
| Loss max abs diff | `0.00362` (step `2`) |
| Dataset alignment | `80/80` exact at `offset=0` |

### Run 2: Soft-gating final dataset-trace check

**Output directory:** `outputs/nanovlm_parity_benchmarks/soft-gating-datasettrace-softgating-finalcheck-20260301`

| Metric | Value |
|--------|-------|
| Baseline run | `ce514a4q` |
| Torchtitan run | `md7kgkar` |
| Baseline elapsed | `147.16s` |
| Torchtitan elapsed | `112.06s` |
| Baseline peak VRAM | `28499 MiB` |
| Torchtitan peak VRAM | `22105 MiB` |
| Loss mean abs diff | `0.00648` |
| Loss max abs diff | `0.01582` (step `9`) |
| Dataset alignment | `80/80` exact at `offset=0` |

## Analysis

### What Worked
- Startup stream behavior was aligned with baseline warmup-discard semantics.
- Soft-gating warmup handling was aligned for short parity horizons.
- Exact microbatch stream identity (`80/80`) removed data-order as a confounder.

### What Failed
- Earlier variants without warmup/stream controls produced larger soft-gating loss spikes.
- Performance-only checks were insufficient to claim parity without dataset-trace proof.

### Key Insights
1. Dataset-trace alignment is a hard precondition for parity interpretation.
2. Torchtitan can keep speed and VRAM gains while reaching practical parity bounds.
3. Soft-gating parity should be judged by bounded step-wise deltas, not single-point metrics.

## Outcome

Against the three target requirements:
- Torchtitan faster: **pass**
- Torchtitan lower VRAM: **pass**
- Loss practically same under paired data stream: **pass**

## Follow-up Knowledge Capture

- Added skill: `.codex/skills/torchtitan-dataset-trace-parity-gate/SKILL.md`
- Updated skill: `.codex/skills/torchtitan-soft-gating-parity-guardrails/SKILL.md`
