---
name: torchtitan-soft-gating-parity-guardrails
description: >
  Guardrails for soft-gating parity when speed and VRAM improve but loss diverges.
  Use when: 100-step soft-gating A/B shows good performance but unacceptable loss drift.
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
| Speed delta | +21.04% | Baseline `co8wprbr` (497.13s) vs Torchtitan `g0ci82mg` (392.51s) |
| VRAM delta | -6416 MiB | Baseline `28521 MiB` vs Torchtitan `22105 MiB` |
| Mean abs loss diff | 0.005222 | Too high for strict parity |
| Max abs loss diff | 0.16211 | Largest divergence at step 9 |

## Recommended Practice

- Treat soft-gating parity as failed until mean and max step-wise loss deltas are within your accepted tolerance.
- Lock data order, effective batch, LR schedule, and optimizer grouping before any additional performance tuning.
- Use the same 100-step harness and compare all paired steps, not only step-100.
- Reject incomplete runs and network-failed starts from parity evidence.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Declaring parity from speed and VRAM only | Numeric behavior still diverged | Parity is a loss-alignment decision, not a throughput decision |
| Mixing valid and invalid benchmark runs | Baseline with zero logged steps polluted conclusions | Require complete logs on both frameworks |
| Tuning multiple axes at once | Root cause of divergence became unclear | Use one-axis changes with full reruns |

## Configuration

```yaml
parity_gate:
  steps: 100
  require_complete_steps: true
  compare_metric: step_loss
  report:
    - mean_abs_diff
    - max_abs_diff
    - step_of_max_abs_diff
```

## References

- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft100-vramfix-clean1/summary.json`
- Related benchmark summary: `torchtitan/outputs/nanovlm_parity_benchmarks/soft-gating-soft20-optgroup-fix/summary.json`
- Related troubleshooting: `references/troubleshooting.md`
