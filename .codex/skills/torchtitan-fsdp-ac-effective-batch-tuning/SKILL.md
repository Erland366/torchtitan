---
name: torchtitan-fsdp-ac-effective-batch-tuning
description: >
  Tune activation checkpointing under FSDP with fixed effective batch and evidence-based speed/VRAM decisions.
  Use when: selecting AC mode and local batch size for nanoVLM Torchtitan runs on limited VRAM.
metadata:
  short-description: "Fixed-global-batch AC tuning protocol for FSDP"
  tags:
    - research
    - torchtitan
    - fsdp
    - activation-checkpointing
    - vram
    - performance
  domain: research
  created: 2026-03-05
  author: codex
---

# Torchtitan FSDP AC Effective Batch Tuning

## General Description

This skill defines a practical protocol to choose activation checkpointing mode under FSDP while keeping effective batch fixed.
It avoids invalid comparisons by first finding each mode's max feasible local batch, then comparing 100-step wall clock and peak VRAM at the same global batch.

## When to Apply

Use this knowledge when:
- You train nanoVLM with FSDP and care about both speed and memory.
- You need to compare `activation_checkpoint.mode=none` vs `full` fairly.
- You can keep `global_batch_size` fixed and let gradient accumulation adapt.

Do NOT use when:
- Different modes are compared with different effective batch sizes.
- One side is evaluated only on partial/failed runs.

## Results Summary

Evidence source:
- `outputs/ac_benchmarks/max-batch-ac-compare-2gpu-20260305-fullplan/summary.json`

| Config | no-AC (local, GA) | full-AC (local, GA) | Wall Clock Delta (full-none) | VRAM Delta (full-none) | Recommendation |
|--------|--------------------|---------------------|-------------------------------|------------------------|----------------|
| `nanovlm_230m_vanilla_finevisionmax_nopack` | `16, 2` | `32, 1` | `+18.28s` (slower) | `-4484 MiB` | choose by priority: speed=`none`, VRAM=headroom=`full` |
| `nanovlm_230m_momh_soft_gating_b5_tttv_nopack` | `16, 2` | `32, 1` | `-17.38s` (faster) | `-4364 MiB` | prefer `full` |

## Recommended Practice

### Step 1: Fix effective batch and search max local batch per mode

- Keep `global_batch_size` identical across modes.
- For each AC mode, increase local batch until OOM, then back off to highest successful value.

### Step 2: Run matched final comparisons

- Use the selected local batch for each mode.
- Run the same step count (at least `100` steps for quick decisions).
- Compare:
  - elapsed wall clock
  - peak VRAM (any GPU and total)
  - median TPS excluding step 1

### Step 3: Decide per model, not globally

- If full AC is both faster and lower VRAM, standardize on full AC for that config.
- If full AC is lower VRAM but slower, choose based on the bottleneck:
  - throughput bottleneck: keep no AC
  - memory bottleneck / higher micro-batch need: use full AC

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Comparing AC modes with different global batch | Effective optimization target changed | Keep `global_batch_size` fixed and only vary local batch/GA |
| Treating OOM-at-search as a failed benchmark | OOM is expected during max-batch discovery | Only final selected runs determine mode quality |
| Using TPS alone as final decision | Can disagree with end-to-end elapsed | Use elapsed wall clock as primary speed metric |

## Configuration

```bash
source /home/coder/edd/nanoVLM_root/nanoVLM_main/.venv/bin/activate && \
python /home/coder/edd/nanoVLM_root/torchtitan/scripts/nanovlm_ac_batchsize_benchmark.py \
  --configs nanovlm_230m_vanilla_finevisionmax_nopack nanovlm_230m_momh_soft_gating_b5_tttv_nopack \
  --modes none full \
  --nproc-per-node 2 \
  --global-batch-size 64 \
  --search-steps 20 \
  --steps 100
```

## References

- Report: `training_reports/torchtitan-fsdp-ac-fixed-global-batch-benchmark-2026-03-05.md`
- Related skill: `.codex/skills/torchtitan-vram-gap-triage/SKILL.md`
- Related skill: `.codex/skills/torchtitan-soft-gating-parity-guardrails/SKILL.md`
