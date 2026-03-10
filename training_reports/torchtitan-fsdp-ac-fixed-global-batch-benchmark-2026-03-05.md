# Report: torchtitan-fsdp-ac-fixed-global-batch-benchmark

**Date:** 2026-03-05
**Author:** codex
**Status:** Completed

## Objective

Test whether enabling activation checkpointing (AC) under 2-GPU FSDP improves wall-clock speed while reducing VRAM, with effective batch held constant.

## Setup

### Environment
- Hardware: `2x A100-SXM4-40GB`
- Runtime: `torchrun --nproc_per_node=2 -m torchtitan.train --module nanoVLM`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming)

### Configuration

- Harness: `scripts/nanovlm_ac_batchsize_benchmark.py`
- Modes compared: `none` vs `full` activation checkpointing
- Search policy: highest valid local batch per mode with `search_steps=20`
- Final comparison: `steps=100`
- Fixed effective batch: `global_batch_size=64`
- Output directory: `outputs/ac_benchmarks/max-batch-ac-compare-2gpu-20260305-fullplan`

## Experiments

### Run 1: `nanovlm_230m_vanilla_finevisionmax_nopack`

| Mode | Max local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB, any GPU) | Median TPS excl step1 | Final loss |
|------|-----------------:|-----------:|------------:|--------------------------:|----------------------:|-----------:|
| `none` | 16 | 2 | 499.62 | 29291 | 15656 | 5.81250 |
| `full` | 32 | 1 | 517.90 | 24807 | 17083 | 6.18750 |

Delta (`full - none`):
- elapsed: `+18.28s` (slower wall clock)
- peak VRAM: `-4484 MiB` (`~15.31%` lower)
- median TPS: `+1427` (`~9.11%` higher)

### Run 2: `nanovlm_230m_momh_soft_gating_b5_tttv_nopack`

| Mode | Max local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB, any GPU) | Median TPS excl step1 | Final loss |
|------|-----------------:|-----------:|------------:|--------------------------:|----------------------:|-----------:|
| `none` | 16 | 2 | 384.79 | 29435 | 21606 | 5.43750 |
| `full` | 32 | 1 | 367.41 | 25071 | 32530 | 5.68750 |

Delta (`full - none`):
- elapsed: `-17.38s` (faster wall clock)
- peak VRAM: `-4364 MiB` (`~14.83%` lower)
- median TPS: `+10924` (`~50.56%` higher)

## Analysis

### What Worked
- AC reliably increased feasible local batch from `16` to `32` at fixed global batch.
- AC reduced peak VRAM by about `4.3-4.5 GiB` per GPU on both vanilla and soft-gating.
- Soft-gating gained both speed and memory improvements with AC.

### What Failed
- Vanilla did not improve wall-clock under AC in this 100-step run despite better median TPS.
- Non-AC local batch `32` OOMs for both configs, so no-AC cannot use GA `1` at this global batch.

### Key Insights
1. For fixed-effective-batch tuning, AC is a reliable memory lever and a model-dependent speed lever.
2. Soft-gating benefits from AC for both speed and VRAM in this environment.
3. Vanilla requires explicit wall-clock A/B confirmation; higher TPS alone is not sufficient.

## Next Steps

- [ ] Add per-step latency decomposition (compute vs wait/collective) for vanilla `none` vs `full`.
- [ ] Benchmark an intermediate AC strategy if available (not only `none` and `full`).
- [ ] Keep fixed-global-batch A/B as a standard preflight before long training runs.

## Appendix

- Benchmark summary:
  - `outputs/ac_benchmarks/max-batch-ac-compare-2gpu-20260305-fullplan/summary.md`
  - `outputs/ac_benchmarks/max-batch-ac-compare-2gpu-20260305-fullplan/summary.json`
