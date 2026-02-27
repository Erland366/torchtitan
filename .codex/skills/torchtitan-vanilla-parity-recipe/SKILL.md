---
name: torchtitan-vanilla-parity-recipe
description: >
  Keep nanoVLM_main and Torchtitan vanilla training aligned while preserving throughput.
  Use when: porting vanilla finevisionmax training and you need step-level loss parity checks.
metadata:
  short-description: "Vanilla parity checklist for Torchtitan porting"
  tags:
    - research
    - torchtitan
    - nanovlm
    - parity
  domain: research
  created: 2026-02-27
  author: codex
---

# Torchtitan Vanilla Parity Recipe

## General Description

This skill captures a proven parity workflow for nanoVLM vanilla training after moving to Torchtitan.  
It focuses on preserving training numerics first, then validating throughput and VRAM with reproducible 100-step A/B runs.

## When to Apply

Use this knowledge when:
- You are porting `nanoVLM_main` vanilla training to Torchtitan.
- You need close step-by-step loss alignment for at least the first 100 steps.
- You must compare speed and peak VRAM under identical run conditions.

Do NOT use when:
- The target run is MoMH soft-gating and not vanilla.
- Dataset, batch size, gradient accumulation, or logging cadence differ between A/B runs.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| 100-step paired mean abs loss diff | 0.0002273 | Across 100 matched steps (baseline `3epjh0uu` vs Torchtitan `ea33lgdw`) |
| 100-step paired max abs loss diff | 0.00104 | Maximum observed step gap at step 71 |
| Baseline 100-step elapsed | 356.65s | `nanoVLM_main` run `3epjh0uu` |
| Torchtitan 100-step elapsed | 324.78s | Torchtitan run `ea33lgdw` |
| Baseline peak VRAM | 26249 MiB | external `nvidia-smi` sampler |
| Torchtitan peak VRAM | 19995 MiB | external `nvidia-smi` sampler |

## Recommended Practice

Keep the parity-critical path stable and change one axis at a time.

### Step 1: Lock core run parity inputs

- Match dataset path, dataset config name, sample filtering thresholds, batch size, gradient accumulation, LR schedule shape, and training steps.
- Keep compile enabled for both model and loss.
- Use batch-level collation and flatten images to tensor in the dataloader path.

### Step 2: Run strict 100-step A/B verification

- Run baseline and Torchtitan under identical conditions.
- Record:
  - step-100 loss
  - step-wise loss deltas over all 100 steps
  - end-to-end elapsed seconds
  - external peak VRAM
- Treat any failed startup run (network/init failure) as invalid evidence.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| List-image collation + in-forward image flattening | Changed runtime behavior and increased peak memory while drifting loss | Keep image flattening in dataloader tensor collation for vanilla parity |
| Disabling compile for short 100-step window | Lower throughput and higher peak memory in this setup | Keep compile enabled and optimize around data/step pipeline |
| Comparing runs with mismatched log/checkpoint settings | Distorted wall-clock and memory comparisons | Keep A/B settings identical except the variable under test |

## Configuration

```yaml
# Parity-critical defaults (conceptual)
training:
  steps: 100
  local_batch_size: 8
  global_batch_size: 64
  seq_len: 2048

metrics:
  log_freq: 1
  enable_wandb: true

compile:
  enable: true
  components: ["model", "loss"]

checkpoint:
  enable: true
  last_save_model_only: true
  last_save_in_hf: false
```

## References

- Related log: `references/experiment-log.md` (2026-02-27 retrospective entry)
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/vanilla-vanilla100-vramfix-clean1/summary.json`
