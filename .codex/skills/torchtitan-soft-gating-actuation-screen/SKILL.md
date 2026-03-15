---
name: torchtitan-soft-gating-actuation-screen
description: >
  Screen soft-gating actuation strength before long TorchTitan nanoVLM runs.
  Use when: soft-gating balance metrics look neutral or new balance logic fails
  to improve downstream eval despite stable training.
metadata:
  short-description: "Check whether soft-gating is strong enough to matter"
  tags:
    - research
    - torchtitan
    - nanovlm
    - soft-gating
    - evaluation
  domain: research
  created: 2026-03-15
  author: codex
---

# TorchTitan Soft-Gating Actuation Screen

## General Description

This skill captures a failure mode where TorchTitan nanoVLM soft-gating trains
stably and logs balanced gate probabilities, but the actual `tt` vs `tv` bias
separation remains too small to materially affect downstream quality. It turns
that lesson into a short-run screening protocol so long-run budget is spent
only on variants that clearly move the gate.

## When to Apply

Use this knowledge when:
- controller or aux-loss soft-gating variants do not beat the plain control run
- `tv_prob_mean` stays near `0.5` across layers
- gate-effect diagnostics remain small
- training loss and VRAM look healthy but downstream eval does not improve

Do NOT use when:
- the run is failing operationally before steady-state training
- downstream eval was performed on mismatched checkpoints instead of a shared WSM selection rule

## Results Summary

Reference run family:
- hardware: `2x A100 40GB`
- distributed family: `FSDP`
- activation checkpointing: `full`
- `steps=3000`
- no packing
- WSM merge: last `4` checkpoints, `mean`
- downstream eval: full `mmstar`

| Metric | Value | Notes |
|--------|-------|-------|
| control `mmstar average,none` | `0.3624798405` | Best result in the first `3000`-step WSM comparison |
| controller `mmstar average,none` | `0.3560083661` | Worse than control |
| aux-loss `mmstar average,none` | `0.3538701372` | Worse than control |
| controller abs `tv_error_mean` avg | `0.00012` | Gate stayed essentially neutral |
| controller abs `tv_error_mean` max | `0.00026` | No meaningful layer-level deviation |
| controller implied pair-logit diff avg | `0.00048` | Inferred from logged `tv_prob_mean`; too small to matter |
| aux abs `tv_error_mean` avg | `0.00171` | Gate moved more than controller |
| aux abs `tv_error_mean` max | `0.00461` | Still modest |
| aux `tt_tv_abs_mean` avg | `0.0128` | Direct gate-effect metric across layers |
| aux `tt_tv_abs_max` overall | `0.0573` | Strongest observed per-head bias gap |

## Recommended Practice

### Step 1: Keep the long-run contract fixed

For any actuation screen family, keep these fixed unless the screen is explicitly
about one of them:
- same hardware
- same distributed family
- same activation checkpoint mode
- same global/local batch
- same checkpoint cadence
- same WSM merge rule for promoted comparisons

### Step 2: Run short actuation screens first

Before another `3000`-step balance run, run `100-300` step screens for:
- control
- `momh_soft_gating_scale in {2, 4}`
- controller update rate above the neutral reference, e.g. `{0.002, 0.004}`
- stronger warm init if that path is available

Promote only variants that show clearly larger gate-effect metrics without early-step instability.

### Step 3: Use direct actuation metrics, not only balance proxies

At minimum, inspect:
- `momh_balance/layer_*/tv_prob_mean`
- `momh_balance/layer_*/tv_error_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_max`

Treat near-zero `tv_error_mean` as evidence that the controller is not steering
the gate strongly enough, even if training remains stable.

### Step 4: Choose winners by merged-checkpoint downstream eval

Do not promote a balance variant on final-step training loss alone.
Keep WSM selection and full downstream eval as the decision gate.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Controller balance run | Gate stayed too close to `0.5/0.5` | A stable controller can still be too weak to matter |
| Aux-loss balance run | Gate moved modestly but downstream `mmstar` got worse | More movement alone is not enough |
| Judging from balance proxy only | `tt/tv` gate probability does not guarantee meaningful realized attention change | Log direct bias-gap metrics and, if possible, realized-effect metrics |
| Spending long-run budget before actuation screening | Weak variants can consume full `3000`-step runs without any downstream gain | Require `100-300` step promotion screens first |

## Configuration

```yaml
soft_gating_actuation_screen:
  fixed_long_run_contract:
    activation_checkpoint_mode: full
    global_batch_size: 64
    local_batch_size: 32
    checkpoint_interval: 250
    wsm_last_n: 4
    wsm_merge_method: mean
  short_screen:
    steps: [100, 300]
    gate_scale: [1, 2, 4]
    controller_update_rate: [0.001, 0.002, 0.004]
  promote_if:
    gate_effect_increases: true
    no_early_step_instability: true
    downstream_eval_on_merged_checkpoint: true
```

## References

- `eval_results/torchtitan/wsm-s0-control-mmstar-full-20260315-v1/per_task.json`
- `eval_results/torchtitan/wsm-s1-controller-mmstar-full-20260315-v1/per_task.json`
- `eval_results/torchtitan/wsm-s2-aux-mmstar-full-20260315-v1/per_task.json`
- `outputs/wsm_softgating_3000_20260314_wandb/s1_controller/tb/20260314-2334/wandb/run-20260314_233404-4f7iu2qc/files/wandb-summary.json`
- `outputs/wsm_softgating_3000_20260314_wandb/s2_aux/tb/20260315-1122/wandb/run-20260315_112232-vfp59yr1/files/wandb-summary.json`
