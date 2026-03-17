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

This skill captures two distinct soft-gating actuation failure modes in
TorchTitan nanoVLM:
- genuinely weak controller/aux actuation
- invalid warm-start experiments caused by `momh_gate` initialization being lost
  during the `to_empty(...)+init_weights()` model construction path

It turns those lessons into a short-run screening protocol so long-run budget is
spent only on variants that clearly move the gate and are known to survive the
real runtime.

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
| `C1` avg `tt_tv_abs_mean` | `0.001040` | `scale=1`, `u=1e-3`, `100` steps |
| `C2` avg `tt_tv_abs_mean` | `0.002017` | `scale=2`, `u=1e-3`, `100` steps |
| `C3` avg `tt_tv_abs_mean` | `0.004183` | `scale=4`, `u=1e-3`, best stage-1 screen |
| `C4` avg `tt_tv_abs_mean` | `0.003876` | `scale=2`, `u=2e-3`, `100` steps |
| `C3` confirm300 avg `tt_tv_abs_mean` | `0.004078` | Stable at `300` steps, still below promotion bar |
| promotion bar | `0.02` | Minimum average layerwise actuation required before another `3000`-step run |
| fixed `split_warm` abs mean | `~4.0` | Corrected `10`-step unfrozen screen after reapplying init through `init_weights()` |
| fixed `split_warm_frozen_gate` abs mean | `4.0` | Corrected `10`-step frozen screen; all `30` layers nonzero |
| `R1` split warm `100`-step abs mean | `3.9990` | Strong actuation survives through step `100` |
| `R2` split warm + low gate LR abs mean | `3.9999` | Best current trainable retention recipe |
| `R3` split warm + low gate LR + layer-mean controller abs mean | `3.9826` | Controller preserved most specialization, but did not improve balance enough to justify added complexity |
| `R4` split warm + frozen gate abs mean | `4.0` | Exact retention ceiling with major throughput gain |
| `R4` throughput | `74660.1` | Upper bound showing early frozen gates are cheap operationally |
| `R2` full `mmstar average,none` | `0.3494459090` | Strong retained actuation and stable training, but worse than control, controller, and aux-loss |
| `R2` final 3000-step loss | `0.48438` | Stable long-run optimization did not translate into better downstream score |
| `R2` final-step TPS | `~35093` | Good systems result, but not a quality win |
| `R5` freeze-thaw `mmstar average,none` | `0.3475482593` | Scheduled retention still underperformed all earlier long-run variants |
| `FZ1` pure frozen gate `mmstar average,none` | `0.3434869139` | Best throughput, worst downstream score |
| `W1 init1` weak split warm `100`-step abs mean | `2.000032` | `strength=1.0`, `lr_momh_gate=1e-5` |
| `W1 init05` weak split warm `100`-step abs mean | `1.000081` | `strength=0.5`, `lr_momh_gate=1e-5` |
| `W2 init1` `300`-step confirm abs mean | `1.999163` | Stable nontrivial gate with lower rigidity |
| `W1 init1` `mmstar average,none` | `0.3531268939` | Better than `R2`, `R5`, and `FZ1`, but still below control/controller/aux |

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

### Step 2: Verify warm-start initialization survives the real runtime

Before interpreting any warm-start result, verify that the configured gate init
survives the real `to_empty(...)+init_weights()+FSDP` path.

Use a `10`-step screen and require:
- all layers log nonzero `momh_gate_effect/*/tt_tv_abs_mean`
- the measured magnitude matches the requested init within tolerance

If this check fails, do not trust any warm-start experiment result until the
initialization path is fixed.

### Step 3: Once warm-start is valid, screen stability rather than raw strength

After the init fix, `tt_tv_split_warm` already produces very strong actuation
(`tt_tv_abs_mean ~ 4.0`) at `10` steps. The next problem is not "make it
stronger" but "keep that specialization useful."

Promote only variants that remain strong without early-step instability.
A practical promotion rule is:
- require average layerwise `momh_gate_effect/*/tt_tv_abs_mean >= 0.02`
- if warm-start is already in the `~4.0` range, stop sweeping `gate_scale`
  and instead test how long the specialization survives
- first try reduced `lr_momh_gate` on top of a strong warm start
- use frozen-gate runs as a ceiling / upper-bound check
- if frozen gates look best, move to a freeze-thaw schedule rather than more
  controller-gain sweeps
- keep layer-mean controller secondary unless it changes balance without
  noticeably eroding retained specialization

### Step 4: Use direct actuation metrics, not only balance proxies

At minimum, inspect:
- `momh_balance/layer_*/tv_prob_mean`
- `momh_balance/layer_*/tv_error_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_max`
- `momh_gate_effect/layer_*/tt_tv_signed_mean`
- `momh_gate_effect/layer_*/tt_tv_signed_std`

Treat near-zero `tv_error_mean` as evidence that the controller is not steering
the gate strongly enough, even if training remains stable. Conversely, if
`tt_tv_abs_mean` is already large, the problem is no longer weak actuation.

The current evidence says:
- increasing `momh_soft_gating_scale` does increase the proxy effect size
- but scale alone was not enough to clear the promotion threshold
- the original warm-start screens were invalid because the init was not surviving `init_weights()`
- after fixing that bug, `tt_tv_split_warm` already gives strong actuation at runtime
- `tt_tv_split_warm` with `lr_momh_gate=1e-5` preserves that strong actuation
  almost exactly through `100` steps
- frozen gates preserve the warm start exactly and sharply improve throughput,
  so the next design cycle should prioritize preserving or scheduling that
  strong specialization, not making the raw score bigger

### Step 5: Choose winners by merged-checkpoint downstream eval

Do not promote a balance variant on final-step training loss alone.
Keep WSM selection and full downstream eval as the decision gate.

### Step 6: Promote retention winners before inventing new balancing logic

The current best trainable short-run recipe is:
- `tt_tv_split_warm`
- `lr_momh_gate=1e-5`
- no controller

Before adding more balancing mechanisms, promote that recipe to:
- a `300`-step confirmation run
- then a `3000`-step WSM run if the `300`-step check still shows strong
  retained actuation and stable loss

Treat freeze-thaw as the next new mechanism if the low-gate-LR variant either:
- loses useful specialization over longer horizons, or
- keeps strong specialization but still underperforms on merged downstream eval

### Step 7: Do not confuse retained actuation with downstream usefulness

`R2` established that a recipe can:
- preserve `tt_tv_abs_mean ~ 4.0` in short screens
- train stably for `3000` steps
- keep strong throughput and VRAM

and still lose on full merged `mmstar`.

That means:
- strong gate retention is necessary evidence that the mechanism is active
- it is not sufficient evidence that the mechanism is helpful
- merged downstream eval remains the only promotion gate for long-run recipes

### Step 8: If strong gates lose, weaken the gate before inventing a new controller

The weaker retained split-warm runs established that reducing the initialization
strength from `2.0` to `1.0`:
- preserves a clearly nonzero gate (`tt_tv_abs_mean ~ 2.0`)
- remains operationally stable through `3000` steps
- improves full `mmstar` relative to stronger static-gate recipes

But the weaker gate still did not beat the plain control or the earlier
controller/aux runs. That means:
- weakening the gate is a sensible direction when strong retained gates lose
- but the current soft-gating bias family still needs a better usefulness target
  than “stronger or weaker static `tt/tv` bias”
- if a weaker gate still loses after a real `3000`-step WSM eval, treat that as
  evidence that this bias family is near exhaustion for the current benchmark

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Controller balance run | Gate stayed too close to `0.5/0.5` | A stable controller can still be too weak to matter |
| Aux-loss balance run | Gate moved modestly but downstream `mmstar` got worse | More movement alone is not enough |
| Scale-only controller sweep | `momh_soft_gating_scale` improved proxy metrics but topped out at `0.004078` average `tt_tv_abs_mean` after `300` steps | Stronger gain alone does not justify another long run |
| Warm-start screen before init fix | `momh_gate` init was dropped by `to_empty(...)+init_weights()` | Never trust warm-start results until the real runtime preserves them |
| Adding controller on top of corrected warm start | Layer-mean controller did not materially improve the `100`-step retained balance signal and slightly reduced actuation versus low gate LR alone | Preserve specialization first; add balancing only if it changes the downstream target materially |
| Strong retained warm-start (`R2`) | Stable training and strong retained gate still scored `0.34945` on full `mmstar` | Static strong specialization can still be the wrong recipe; downstream eval outranks gate metrics |
| Freeze-thaw (`R5`) | Scheduling the strong gate still scored `0.34755` on full `mmstar` | Scheduling alone does not rescue a misaligned static-bias target |
| Pure frozen gate (`FZ1`) | Exact gate retention plus very high throughput still scored `0.34349` | Strong constant gating can be a systems win and a quality loss at the same time |
| Weaker retained gate (`W1 init1`) | Reducing init strength improved over the stronger static-gate runs but still scored `0.35313` | Lower rigidity helps, but this bias family still does not surpass the simpler control recipe |
| Judging from balance proxy only | `tt/tv` gate probability does not guarantee meaningful realized attention change | Log direct bias-gap metrics and, if possible, realized-effect metrics |
| Spending long-run budget before actuation screening | Weak variants can consume full `3000`-step runs without any downstream gain | Require `100-300` step promotion screens first |
| Chasing stronger scores after the init fix | Warm-start actuation was already `~4.0`; the real issue became retention and usefulness | Once the gate is strong, change the training recipe, not the raw score scale |

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
  warm_start_gate:
    verify_at_10_steps: true
    expected_abs_mean: 4.0
  retention_screen:
    steps: 100
    low_gate_lr: 1e-5
    compare_against:
      - split_warm
      - split_warm_low_gate_lr
      - split_warm_layer_mean_low_gate_lr
      - split_warm_frozen_gate
  long_run_validation:
    steps: 3000
    accept_only_if:
      - merged_full_mmstar_beats_or_matches_control
      - strong_retained_actuation_does_not_trade_off_quality
  promote_if:
    average_tt_tv_abs_mean_gte: 0.02
    no_early_step_instability: true
    downstream_eval_on_merged_checkpoint: true
```

## References

- `eval_results/torchtitan/wsm-s0-control-mmstar-full-20260315-v1/per_task.json`
- `eval_results/torchtitan/wsm-s1-controller-mmstar-full-20260315-v1/per_task.json`
- `eval_results/torchtitan/wsm-s2-aux-mmstar-full-20260315-v1/per_task.json`
- `outputs/wsm_softgating_3000_20260314_wandb/s1_controller/tb/20260314-2334/wandb/run-20260314_233404-4f7iu2qc/files/wandb-summary.json`
- `outputs/wsm_softgating_3000_20260314_wandb/s2_aux/tb/20260315-1122/wandb/run-20260315_112232-vfp59yr1/files/wandb-summary.json`
- `outputs/soft_gating_actuation_20260315/c1/tb/20260315-1729/wandb/run-20260315_172908-8zy5vdmm/files/wandb-summary.json`
- `outputs/soft_gating_actuation_20260315/c2/tb/20260315-1733/wandb/run-20260315_173334-us9da8tb/files/wandb-summary.json`
- `outputs/soft_gating_actuation_20260315/c3/tb/20260315-1738/wandb/run-20260315_173806-v8iyz3c5/files/wandb-summary.json`
- `outputs/soft_gating_actuation_20260315/c4/tb/20260315-1742/wandb/run-20260315_174234-mj7k0u83/files/wandb-summary.json`
- `outputs/soft_gating_actuation_20260315/c3_confirm300/tb/20260315-1754/wandb/run-20260315_175455-iswp6247/files/wandb-summary.json`
- `outputs/soft_gating_actuation_fixcheck_20260315_online_r3/split_warm_frozen_gate/tb/20260315-2036/wandb/run-20260315_203620-0foz4aah/files/wandb-summary.json`
- `outputs/soft_gating_actuation_fixcheck_20260315_online_r3/split_warm/tb/20260315-2038/wandb/run-20260315_203847-s6t2he56/files/wandb-summary.json`
- `outputs/soft_gating_retention_20260315/split_warm/tb/20260315-2210/wandb/run-20260315_221012-1piu5o9q/files/wandb-summary.json`
- `outputs/soft_gating_retention_20260315/split_warm_low_gate_lr/tb/20260315-2215/wandb/run-20260315_221522-cvlgcd9q/files/wandb-summary.json`
- `outputs/soft_gating_retention_20260315/split_warm_layer_mean_low_gate_lr/tb/20260315-2220/wandb/run-20260315_222014-085jqs1n/files/wandb-summary.json`
- `outputs/soft_gating_retention_20260315/split_warm_frozen_gate/tb/20260315-2225/wandb/run-20260315_222519-4x73elww/files/wandb-summary.json`
- `outputs/soft_gating_retention_20260316/r2_3000_prod/tb/20260316-0049/wandb/run-20260316_004922-vusfmvt0/files/wandb-summary.json`
- `eval_results/torchtitan/r2-mmstar-full-validate-20260316/per_task.json`
