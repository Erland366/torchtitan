---
name: torchtitan-soft-gating-parity-guardrails
description: >
  Guardrails for soft-gating parity when speed and VRAM improve but loss diverges.
  Use when: 100-step soft-gating A/B shows good performance but unacceptable loss drift,
  especially in early training steps.
metadata:
  short-description: "Soft-gating parity triage checklist"
  tags:
    - research
    - torchtitan
    - nanovlm
    - soft-gating
    - parity
  domain: research
  created: 2026-02-27
  author: codex
---

# Torchtitan Soft-Gating Parity Guardrails

## General Description

This skill captures a defensive workflow for soft-gating ports where Torchtitan is faster and lower-memory but not yet numerically aligned with nanoVLM_main.  
It prioritizes step-level loss parity checks before performance claims.

## When to Apply

Use this knowledge when:
- You are comparing soft-gating runs between `nanoVLM_main` and Torchtitan.
- Speed and VRAM improve but step-wise loss differences remain large.
- You need a strict pass/fail gate for parity before merging changes.

Do NOT use when:
- The run pair does not have the same target steps.
- Either side failed to log complete step-wise losses.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Speed delta | +14.06% | Baseline `8k9vwq4u` (440.46s) vs Torchtitan `nz6mijs6` (378.54s) |
| VRAM delta | -6416 MiB | Baseline `28521 MiB` vs Torchtitan `22105 MiB` |
| Mean abs loss diff | 0.005749 | Better than earlier candidates, still not exact parity |
| Max abs loss diff | 0.16721 | Largest divergence at step 9 |
| Baseline-vs-baseline mean abs diff | 0.004764 | Early-step instability exists even without framework change |
| Baseline-vs-baseline max abs diff | 0.14690 | Also peaks at step 9 |

## Current Best Reference State

Use this as the default soft-gating reference until a new run beats it on all three goals:
- faster than baseline,
- lower peak VRAM than baseline,
- lower (or equal) mean/max loss diff than current best.

Reference run pair:
- benchmark dir: `outputs/nanovlm_parity_benchmarks/soft-gating-soft100-flexwarm-structonly-20260301-2`
- baseline wandb: `8k9vwq4u`
- torchtitan wandb: `nz6mijs6`
- parity: mean abs diff `0.0057487`, max abs diff `0.16721` at step `9`

## Negative Results Ledger

Do not repeat these changes without a new hypothesis, because they regressed soft-gating parity:

| Change | Run | Outcome |
|--------|-----|---------|
| Removed pre-step microbatch consume | `soft-gating-soft20-nowarmconsume-20260301-1` | mean abs diff `0.04177`, max `0.24875` (worse) |
| Passed `None` samples through dataset iterator | `soft-gating-soft20-nonepassthrough-20260301-1` | mean abs diff `0.02915`, max `0.16999` (worse) |
| Added strict raw-image validity sync in trainer path | `soft-gating-soft20-validitysync-20260301-1` | mean abs diff `0.02386`, max `0.14260` (worse) |

## Recommended Practice

- Treat soft-gating parity as failed until mean and max step-wise loss deltas are within your accepted tolerance.
- Lock data order, effective batch, LR schedule, and optimizer grouping before any additional performance tuning.
- Use the same 100-step harness and compare all paired steps, not only step-100.
- Reject incomplete runs and network-failed starts from parity evidence.
- Always compute a baseline-vs-baseline reference band before attributing all drift to Torchtitan.
- Prioritize step `1-15` investigations first; this is where large soft-gating divergence is most likely.
- Prefer model+loss compile over model-only compile for this port until a model-only run proves equal or better loss parity and VRAM.
- Use `soft-gating-soft100-flexwarm-structonly-20260301-2` as the control arm for all new soft-gating trials.

## Early-Step Drift Triage

1. Confirm paired run validity:
   - both sides logged all 100 steps,
   - same dataset/config/seed controls,
   - no startup failures.
2. Compute three comparisons on identical parsing logic:
   - baseline A vs baseline B,
   - baseline A vs Torchtitan,
   - baseline B vs Torchtitan.
3. If all large errors cluster in early steps (for example step 9-11), isolate numeric/runtime differences first:
   - grad clipping path,
   - optimizer grouping and per-group LR schedule,
   - compile boundaries and AMP behavior.
4. Only after early-step drift is reduced, continue with throughput/memory tuning.

## Early-Step Parity Gate

Treat parity as failed when either of these checks fails on steps `1-15`:
- early-step max absolute diff exceeds `0.02`,
- early-step mean absolute diff exceeds `0.005`.

These thresholds are stricter than whole-window averages and are intended to catch the soft-gating instability zone around steps `9-14`.

## Minimum A/B Matrix

Use this matrix before any additional architectural changes:

| Axis | Values |
|------|--------|
| `workers` | `0`, `1` |
| compile scope | `model+loss`, `model-only` |
| optimizer backend | `fused`, `foreach` |
| seed | fixed single seed, then one alternate seed |
| dataloader controls | fixed shuffle order, identical filtering thresholds |

For each cell:
- run paired baseline vs Torchtitan with identical step target,
- compute early-step (`1-15`) and full-window (`1-100`) loss deltas,
- include baseline-vs-baseline reference in the same report.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Declaring parity from speed and VRAM only | Numeric behavior still diverged | Parity is a loss-alignment decision, not a throughput decision |
| Mixing valid and invalid benchmark runs | Baseline with zero logged steps polluted conclusions | Require complete logs on both frameworks |
| Tuning multiple axes at once | Root cause of divergence became unclear | Use one-axis changes with full reruns |
| Ignoring baseline instability | Torchtitan looked worse than it really was | Always compare against baseline-vs-baseline drift band |

## Configuration

```yaml
parity_gate:
  steps: 100
  require_complete_steps: true
  compare_metric: step_loss
  early_window:
    start_step: 1
    end_step: 15
    max_abs_diff_threshold: 0.02
    mean_abs_diff_threshold: 0.005
  report:
    - early_window_mean_abs_diff
    - early_window_max_abs_diff
    - mean_abs_diff
    - max_abs_diff
    - step_of_max_abs_diff
```

## References

- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft100-vramfix-clean1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft100-flexwarm-structonly-20260301-2/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-optgroup-fix/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-clipfix-final2-20260228-022309/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-revertlosswarmup-20260228-1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-revertwarmup-modelonly-20260228-1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nowarmconsume-20260301-1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nonepassthrough-20260301-1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-validitysync-20260301-1/summary.json`
- Related troubleshooting: `references/troubleshooting.md`
