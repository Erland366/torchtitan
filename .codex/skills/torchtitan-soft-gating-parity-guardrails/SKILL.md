---
name: torchtitan-soft-gating-parity-guardrails
description: >
  Guardrails for soft-gating parity when speed and VRAM improve but loss still needs
  bounded step-level agreement against nanoVLM_main.
  Use when: soft-gating ports are operationally faster/lower-memory and need stable parity acceptance criteria.
metadata:
  short-description: "Soft-gating parity acceptance guardrails"
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

This skill defines a conservative acceptance protocol for soft-gating parity.
It treats parity as a combination of exact data-stream alignment and bounded step-level loss deltas,
while preserving the speed and VRAM improvements that motivated the port.

## When to Apply

Use this knowledge when:
- You are validating soft-gating in paired `nanoVLM_main` vs Torchtitan runs.
- Speed and VRAM are better in Torchtitan, but you need reliable parity gates.
- You are deciding whether a run is "same enough" to unblock integration.

Do NOT use when:
- Either side has incomplete step logs.
- Dataset-trace alignment is not yet exact.

## Results Summary

| Run Pair | Steps | Loss Mean Abs Diff | Loss Max Abs Diff | Baseline | Torchtitan |
|----------|-------|--------------------|-------------------|----------|------------|
| `soft-gating-datasettrace-softgating-finalcheck-20260301` | 10 | `0.00648` | `0.01582` (step 9) | `ce514a4q`, `147.16s`, `28499 MiB` | `md7kgkar`, `112.06s`, `22105 MiB` |

Derived deltas from this accepted check:
- speed: ~`23.86%` faster (`1.313x`)
- peak VRAM: `-6394 MiB` (~`22.44%` lower)
- dataset trace: `80/80` exact microbatch matches at `offset=0`

## Recommended Practice

### Step 1: Enforce data-stream gate first

Require all of these from the paired `summary.json`:
- `best_alignment.offset == 0`
- `exact_matches == compared_pairs`
- `match_ratio == 1.0`

### Step 2: Enforce soft-gating loss acceptance window

For short closure checks (10 steps), accept only when:
- `max_abs_diff <= 0.02`
- `mean_abs_diff <= 0.01`

For extended checks (100 steps), report both:
- early window (`steps 1-15`) max/mean deltas,
- full window (`steps 1-100`) max/mean deltas.

When comparing runtime configurations (for example AC `none` vs `full`), do not use those runs as parity evidence by default.
Treat AC mode comparisons as performance/memory studies unless they are paired against baseline with full dataset-trace controls.

### Step 3: Keep performance guardrails in the same report

Never accept a parity fix that regresses the primary goals:
- Torchtitan must remain faster than baseline.
- Torchtitan peak VRAM must remain lower than baseline.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Accepting parity without dataset-trace evidence | Loss comparisons were confounded by possible stream skew | Require exact microbatch alignment first |
| Optimizing only for speed/VRAM | Numeric drift remained hidden | Keep loss acceptance gate mandatory |
| Changing many axes at once | Could not isolate cause of drift/regression | Run one hypothesis at a time with paired reruns |
| Using AC-on/off loss deltas as parity verdict | AC-mode A/B is not a baseline-paired parity experiment | Use baseline-vs-Torchtitan paired runs with dataset-trace gate for parity claims |

## Configuration

```yaml
soft_gating_parity_gate:
  dataset_trace:
    require_offset_zero: true
    require_exact_match_ratio: 1.0
  short_check:
    steps: 10
    mean_abs_diff_max: 0.01
    max_abs_diff_max: 0.02
  long_check:
    steps: 100
    report_windows:
      - [1, 15]
      - [1, 100]
  performance:
    require_torchtitan_faster: true
    require_torchtitan_lower_peak_vram: true
```

## References

- `outputs/nanovlm_parity_benchmarks/soft-gating-datasettrace-softgating-finalcheck-20260301/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft100-flexwarm-structonly-20260301-2/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nowarmconsume-20260301-1/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-nonepassthrough-20260301-1/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-soft20-validitysync-20260301-1/summary.json`
