# Report: torchtitan-ddp-vs-fsdp-fixed-effective-batch

**Date:** 2026-03-12
**Author:** codex
**Status:** Completed

## Objective

Test whether `FSDP` is always better than `DDP` when both are allowed to use
their best-performing settings at the same effective batch size on `2` GPUs.

## Setup

### Environment
- Hardware: `2x A100-SXM4-40GB`
- Runtime: `torchrun --nproc_per_node=2 -m torchtitan.train --module nanoVLM`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming)
- W&B entity/project: `patrickirawan-mbzuai/momh`

### Harness
- Script: `scripts/nanovlm_ddp_vs_fsdp_benchmark.py`
- Search stage: `20` steps per candidate, W&B disabled
- Final stage: `100` steps for the winning DDP and FSDP arm, W&B enabled
- Fixed effective batch: `global_batch_size=64`
- Candidate local batches: `32, 16, 8, 4, 2, 1`
- Families: `ddp`, `fsdp`
- AC modes: `none`, `full`
- Dataloader workers: `0`
- Compile: enabled
- Output directory: `outputs/ddp_vs_fsdp_benchmarks/full-20260312`

### Winner Selection Policy

Within each family, select the fastest successful search candidate using:
1. lowest elapsed wall clock
2. lower peak per-GPU VRAM
3. higher median TPS excluding step `1`
4. larger local batch size

This benchmark compares the best DDP arm against the best FSDP arm. It does not
assume that the largest feasible local batch is automatically the fastest.

## Experiments

### Run 1: `nanovlm_230m_vanilla_finevisionmax_nopack`

| Family | AC | Local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB, any GPU) | Median TPS excl step1 | Final loss | W&B |
|--------|----|------------:|-----------:|------------:|--------------------------:|----------------------:|-----------:|-----|
| `ddp` | `none` | 16 | 2 | 601.99 | 31841 | 12545 | 6.56621 | `0ssc9oxh` |
| `fsdp` | `none` | 8 | 4 | 610.50 | 15627 | 12174 | 6.50000 | `1fxmbuta` |

Delta (`fsdp - ddp`):
- elapsed: `+8.52s` (FSDP slower)
- peak VRAM: `-16214 MiB` (FSDP lower)
- median TPS: `-371`

Verdict: `tradeoff`

Interpretation:
- FSDP reduced memory materially at the same effective batch.
- DDP was still the faster family winner for vanilla in this environment.
- The fastest FSDP arm was `lb8`, even though FSDP could also complete larger local-batch arms. This is a measured result, not a memory-limit statement.

### Run 2: `nanovlm_230m_momh_soft_gating_b5_tttv_nopack`

| Family | AC | Local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB, any GPU) | Median TPS excl step1 | Final loss | W&B |
|--------|----|------------:|-----------:|------------:|--------------------------:|----------------------:|-----------:|-----|
| `ddp` | `none` | 8 | 4 | 552.72 | 21141 | 13278 | 5.66392 | `9jhl8ou8` |
| `fsdp` | `none` | 16 | 2 | 532.83 | 29487 | 14238 | 5.56250 | `2am53tou` |

Delta (`fsdp - ddp`):
- elapsed: `-19.89s` (FSDP faster)
- peak VRAM: `+8346 MiB` (FSDP higher)
- median TPS: `+960`

Verdict: `tradeoff`

Interpretation:
- FSDP was the faster family winner for soft-gating.
- DDP was the lower-VRAM family winner for soft-gating.
- The family choice is therefore model-dependent and objective-dependent.

## Analysis

### What Worked
- The harness successfully compared best-of-family DDP vs FSDP instead of comparing arbitrary fixed micro-batch choices.
- The final result is clear for both configs: neither family dominated both speed and memory.
- In both configs, the winning family arms used `activation-checkpoint.mode=none`.

### What Failed
- The hypothesis "FSDP is always better if it can reduce gradient accumulation" was not supported.
- Local-batch fit alone did not predict the fastest family winner.

### Key Insights
1. FSDP and DDP should be benchmarked per model/config instead of chosen from a single rule of thumb.
2. Activation checkpointing did not produce the fastest winner in this best-of-family comparison, even though it can improve feasible local batch size.
3. Same effective batch does not imply same best local batch across distributed families.
4. For vanilla, FSDP saved VRAM at the same local batch but not enough to move the no-AC fit boundary beyond DDP.

## Decision

- Do not treat FSDP as categorically better than DDP for nanoVLM training.
- Use this benchmark harness when selecting the distributed family for a new config.
- Separate three questions explicitly:
  - maximum feasible local batch
  - fastest local batch
  - lowest-VRAM family winner

## Appendix

- Benchmark summary:
  - `outputs/ddp_vs_fsdp_benchmarks/full-20260312/summary.json`
  - `outputs/ddp_vs_fsdp_benchmarks/full-20260312/summary.md`
- W&B runs:
  - vanilla DDP: `https://wandb.ai/patrickirawan-mbzuai/momh/runs/0ssc9oxh`
  - vanilla FSDP: `https://wandb.ai/patrickirawan-mbzuai/momh/runs/1fxmbuta`
  - soft-gating DDP: `https://wandb.ai/patrickirawan-mbzuai/momh/runs/9jhl8ou8`
  - soft-gating FSDP: `https://wandb.ai/patrickirawan-mbzuai/momh/runs/2am53tou`
