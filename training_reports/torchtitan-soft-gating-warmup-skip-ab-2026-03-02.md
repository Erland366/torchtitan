# Report: torchtitan-soft-gating-warmup-skip-ab

**Date:** 2026-03-02
**Author:** codex
**Status:** Completed

## Objective

Validate whether removing TorchTitan's startup warmup-discard keeps soft-gating parity acceptable, or whether warmup-discard remains required for stable parity against `nanoVLM_main`.

## Setup

### Environment
- Hardware: `1x A100-SXM4-40GB`
- Runtime: `nanoVLM_main` baseline vs `torchrun -m torchtitan.train --module nanoVLM`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming), soft-gating configs

### Configuration

Common settings across both A/B checks:
- steps: `100`
- project/entity: `patrickirawan-mbzuai/momh`
- compile: enabled
- parity harness: `scripts/nanovlm_parity_benchmark.py --mode soft-gating`

## Experiments

### Run 1: With startup warmup-discard enabled

**Output directory:** `outputs/nanovlm_parity_benchmarks/soft-gating-codex-softgating-100step-withskip-20260302-0135`

**Results:**
| Metric | Value |
|--------|-------|
| Baseline run | `i76lufb4` |
| Torchtitan run | `xrxed5z3` |
| Loss mean abs diff | `0.0048146` |
| Loss max abs diff | `0.19608` (step `14`) |
| Baseline peak VRAM | `28521 MiB` |
| Torchtitan peak VRAM | `22107 MiB` |
| Baseline median TPS | `7370.67` |
| Torchtitan median TPS | `31015.0` |

### Run 2: Without startup warmup-discard

**Output directory:** `outputs/nanovlm_parity_benchmarks/soft-gating-codex-softgating-100step-noskip-20260302-0135`

**Results:**
| Metric | Value |
|--------|-------|
| Baseline run | `elv6513t` |
| Torchtitan run | `uzxh88t8` |
| Loss mean abs diff | `0.0556629` |
| Loss max abs diff | `0.35209` (step `5`) |
| Baseline peak VRAM | `28521 MiB` |
| Torchtitan peak VRAM | `22299 MiB` |
| Baseline median TPS | `7477.64` |
| Torchtitan median TPS | `30496.0` |

## Analysis

### What Worked
- Soft-gating paired runs completed cleanly in both variants.
- TorchTitan remained faster and lower-memory than baseline in both variants.

### What Failed
- Removing warmup-discard significantly worsened loss parity:
  - mean abs diff increased by `+0.0508483`
  - max abs diff increased by `+0.15601`
- No-skip variant also regressed TorchTitan peak VRAM (`+192 MiB`) and median TPS (`-519`) vs with-skip.

### Key Insights
1. For this soft-gating port, startup warmup-discard is still a net-positive parity control.
2. "Same dataset source" is insufficient for step-wise parity if startup consumption semantics differ.
3. With-skip remains the better operational default for parity-sensitive A/B validation.

## Outcome

- Decision: keep startup warmup-discard enabled in TorchTitan for soft-gating parity path.
- This report is a concrete A/B justification for that decision at `100` steps.

## Appendix

- Summary files:
  - `outputs/nanovlm_parity_benchmarks/soft-gating-codex-softgating-100step-withskip-20260302-0135/summary.json`
  - `outputs/nanovlm_parity_benchmarks/soft-gating-codex-softgating-100step-noskip-20260302-0135/summary.json`
