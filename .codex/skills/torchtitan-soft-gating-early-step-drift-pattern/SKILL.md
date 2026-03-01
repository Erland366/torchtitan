---
name: torchtitan-soft-gating-early-step-drift-pattern
description: >
  Detect and triage recurring step-9..11 soft-gating loss divergence in paired
  nanoVLM_main vs Torchtitan runs while preserving speed and VRAM gains.
  Use when: overall loss looks close but exact parity fails due to early-step spikes.
metadata:
  short-description: "Early-step soft-gating drift triage"
  tags:
    - research
    - torchtitan
    - nanovlm
    - soft-gating
    - parity
    - diagnostics
  domain: research
  created: 2026-03-01
  author: codex
---

# Torchtitan Soft-Gating Early-Step Drift Pattern

## General Description

This skill captures a repeated soft-gating parity failure mode where Torchtitan remains faster and lower-memory but diverges in early steps.  
The pattern is most visible around steps `9-11`, and can persist even after multiple data/compile adjustments.

## When to Apply

Use this knowledge when:
- You have paired baseline vs Torchtitan soft-gating runs with complete step logs.
- Full-run mean loss difference is moderate, but max step-wise difference is still high.
- Speed and VRAM goals pass, yet strict exact parity fails.

Do NOT use when:
- The run pair is incomplete (missing steps on either side).
- The run has startup/network failures that invalidate comparison.

## Results Summary

| Run Pair | Steps | Mean Abs Diff | Max Abs Diff | Peak Step |
|----------|-------|---------------|--------------|-----------|
| `soft-gating-soft100-flexwarm-structonly-20260301-2` | 100 | `0.0057487` | `0.16721` | `9` |
| `soft-gating-soft20-nowarmconsume-20260301-1` | 20 | `0.04177` | `0.24875` | `2` |
| `soft-gating-soft20-nonepassthrough-20260301-1` | 20 | `0.02915` | `0.16999` | `9` |
| `soft-gating-soft20-validitysync-20260301-1` | 20 | `0.023855` | `0.14260` | `11` |

## Drift Signature

Common signature in this project:
- largest mismatches cluster early (usually around steps `9-11`);
- later steps often re-converge to small absolute differences;
- speed and VRAM improvements remain stable despite numeric mismatch.

## Recommended Practice

1. Keep a fixed control arm:
   `soft-gating-soft100-flexwarm-structonly-20260301-2`.
2. Evaluate each new hypothesis on 20 steps first, but only accept it if:
   - mean diff improves vs control-derived short-run baseline, and
   - max diff in early window (`1-15`) improves.
3. Promote only winning hypotheses to 100-step validation.
4. Maintain three-way interpretation:
   - baseline vs torchtitan,
   - baseline vs baseline (instability band),
   - candidate vs current best control.

## Failure Modes

| What Failed | Why It Hurt | Lesson |
|-------------|-------------|--------|
| Removing startup pre-consume | Shifted effective data stream at first training step | Keep warmup consume behavior for parity |
| Passing `None` samples through iterator | Changed collate/batch composition behavior | Keep pre-collate filtering path unchanged |
| Over-constraining raw-image validity in trainer | Introduced extra branch behavior and no parity gain | Favor minimal parity-preserving checks |

## Configuration

```yaml
soft_gating_parity_protocol:
  control_run: soft-gating-soft100-flexwarm-structonly-20260301-2
  stage_1:
    steps: 20
    gate:
      early_window: [1, 15]
      compare: [mean_abs_diff, max_abs_diff]
      must_improve_vs_control: true
  stage_2:
    steps: 100
    require:
      faster_than_baseline: true
      lower_peak_vram_than_baseline: true
      no_regression_vs_stage_1: true
```

## References

- `outputs/nanovlm_parity_benchmarks/soft-gating-soft100-flexwarm-structonly-20260301-2/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nowarmconsume-20260301-1/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nonepassthrough-20260301-1/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-validitysync-20260301-1/summary.json`
