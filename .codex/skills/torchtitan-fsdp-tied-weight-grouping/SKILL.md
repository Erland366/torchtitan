---
name: torchtitan-fsdp-tied-weight-grouping
description: >
  Keep tied nanoVLM LM weights inside one FSDP unit and evaluate the change as a
  correctness fix first, not as automatic parity evidence.
metadata:
  short-description: "Group tied LM weights into one FSDP unit"
  tags:
    - research
    - torchtitan
    - nanovlm
    - fsdp
    - tied-weights
  domain: research
  created: 2026-03-12
  author: codex
---

# TorchTitan FSDP Tied-Weight Grouping

## General Description

This skill captures the correct FSDP handling for tied nanoVLM LM weights.
When `output.weight` is tied to `tok_embeddings.weight`, both sides of that
shared parameter should remain inside the same FSDP unit.

This is primarily a correctness and maintainability rule. It can reduce VRAM
slightly, but it should not be treated as automatic evidence of exact parity or
as a primary speed fix for dataloader-bound runs.

## When to Apply

Use this knowledge when:
- `lm_tie_weights=True` in nanoVLM.
- You are modifying `parallelize.py` or other FSDP wrapping logic.
- You need to verify that a shared embedding/head parameter is sharded safely.

Do NOT use when:
- The model does not tie input embeddings and output head weights.
- You are evaluating baseline-vs-Torchtitan parity and have not yet run a
  step-level comparison.

## Results Summary

| Config | Baseline | Tied-Grouped | Final Loss | Median TPS Delta | Peak VRAM Delta | Exact Step Parity |
|--------|----------|--------------|-----------:|-----------------:|----------------:|------------------|
| `nanovlm_230m_vanilla_finevisionmax_nopack` | `utriabh7` | `g3rn91jj` | same (`5.81250`) | `+350` | `-1222 MiB` | no |
| `nanovlm_230m_momh_soft_gating_b5_tttv_nopack` | `fws2j2m1` | `xqbxfkgh` | same (`5.43750`) | `-264` | `-916 MiB` | no |

Observed step-level drift in the 100-step checks:
- vanilla: `3` mismatches, max abs diff `0.03125`
- soft-gating: `1` mismatch, max abs diff `0.03125`

## Recommended Practice

### Step 1: Group tied weights structurally

If `lm_tie_weights=True`, wrap these together:
- `tok_embeddings`
- `norm`
- `output`

Do not shard `tok_embeddings` separately from `output` in tied mode.

### Step 2: Validate at two levels

Run:
- focused unit tests that assert grouping behavior
- a short multi-GPU smoke test

Then, if parity matters, run a paired benchmark and compare:
- full step-level loss series
- median TPS excluding warm startup
- peak VRAM

### Step 3: Interpret results correctly

Treat this as:
- a correctness fix for shared parameters under FSDP

Do not treat it as:
- proof of exact numerical parity
- proof that end-to-end training will speed up if the workload is input-bound

Final-loss equality is not enough after FSDP wrapping changes.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Tied embedding and head wrapped by different FSDP units | Shared parameter semantics cross wrapper boundaries | Keep both sides of the tie in one FSDP unit |
| Final loss matched, but parity was declared exact | A few step-level values drifted by `0.03125` | Require full paired loss comparison |
| Speed claim generalized across models | Vanilla and soft-gating reacted differently | Benchmark each model family explicitly |

## Configuration

```yaml
fsdp_tied_weight_policy:
  if_lm_tie_weights: shard_together
  tied_group:
    - tok_embeddings
    - norm
    - output
  validation:
    require_unit_test: true
    require_multigpu_smoke: true
    require_step_level_loss_compare_for_parity_claims: true
```

## References

- `training_reports/torchtitan-fsdp-tied-weight-grouping-benchmark-2026-03-12.md`
- `outputs/fsdp_tie_group_bench_wandb/baseline_vanilla_100/train.log`
- `outputs/fsdp_tie_group_bench_wandb/patched_vanilla_100/train.log`
- `outputs/fsdp_tie_group_bench_wandb/baseline_softgating_100/train.log`
- `outputs/fsdp_tie_group_bench_wandb/patched_softgating_100/train.log`
