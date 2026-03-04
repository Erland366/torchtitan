---
name: torchtitan-dataset-trace-parity-gate
description: >
  Verify that baseline and Torchtitan consume the exact same microbatch stream
  before making loss-parity conclusions.
  Use when: parity work is blocked and you need hard evidence that data order is aligned.
metadata:
  short-description: "Dataset-stream parity gate for nanoVLM ports"
  tags:
    - research
    - torchtitan
    - nanovlm
    - parity
    - dataset
  domain: research
  created: 2026-03-01
  author: codex
---

# Torchtitan Dataset-Trace Parity Gate

## General Description

This skill formalizes a hard gate for parity work: validate microbatch identity first, then evaluate loss drift.
It prevents false debugging on optimizer/compile paths when the true issue is data-stream misalignment.

## When to Apply

Use this knowledge when:
- Baseline and Torchtitan loss differ and root cause is unclear.
- You changed startup/warmup behavior or dataloader plumbing.
- You need to prove that per-step microbatches are truly paired.

Do NOT use when:
- Runs are intentionally comparing different datasets, filters, or sequence limits.
- Either side failed to complete the target step count.

## Results Summary

| Benchmark | Data Alignment | Loss Mean Abs Diff | Loss Max Abs Diff | Speed | Peak VRAM |
|-----------|----------------|--------------------|-------------------|-------|-----------|
| `vanilla-datasettrace-vanilla-fixedwarmup-20260301` | `80/80` exact (`offset=0`) | `0.00137` | `0.00362` (step 2) | TT `117.37s` vs base `141.36s` | TT `20001 MiB` vs base `26169 MiB` |
| `soft-gating-datasettrace-softgating-finalcheck-20260301` | `80/80` exact (`offset=0`) | `0.00648` | `0.01582` (step 9) | TT `112.06s` vs base `147.16s` | TT `22105 MiB` vs base `28499 MiB` |

## Recommended Practice

### Step 1: Run preflight controls before any parity claim

- Always run with a fresh `--dump_folder` to avoid accidental checkpoint resume or
  state-key mismatch.
- Log startup consume mode explicitly (`with-skip` or `no-skip`) and compare only
  runs that use the same mode.
- Require both baseline and Torchtitan to complete the same step horizon.

### Step 2: Run parity harness with dataset trace enabled

```bash
source .venv/bin/activate && \
python scripts/nanovlm_parity_benchmark.py \
  --mode soft-gating \
  --steps 10 \
  --trace-dataset \
  --trace-max-updates 10 \
  --run-suffix trace-check
```

### Step 3: Check alignment first

Read `summary.json` and require all of the following:
- `dataset_trace.best_alignment.offset == 0`
- `dataset_trace.best_alignment.exact_matches == dataset_trace.best_alignment.compared_pairs`
- `dataset_trace.best_alignment.match_ratio == 1.0`

### Step 4: Only then interpret loss differences

If alignment is exact, investigate numeric/runtime causes (compile boundaries, optimizer grouping, clipping semantics).
If alignment is not exact, prioritize startup/warmup/dataloader alignment before touching model math.

### Step 5: Apply both early-window and full-window gates

- Early-window gate: evaluate strict drift on steps `1-15` to catch startup-sensitive behavior.
- Full-window gate: evaluate a longer horizon (for example `100` steps) for stability.
- Treat a run as parity-ready only if both gates pass or both remain within agreed tolerances.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Chasing compile/optimizer hypotheses before checking data stream | Loss drift was confounded by stream skew | Always run dataset trace gate first |
| Comparing runs without full paired steps | Partial logs produced misleading parity summaries | Require complete step windows on both stacks |
| Accepting near-match offsets | Shifted streams can look plausible but still break strict parity | Require `offset=0` and exact microbatch identity |

## Configuration

```yaml
dataset_trace_gate:
  enabled: true
  trace_max_updates: 10
  preflight:
    fresh_dump_folder: true
    require_same_startup_consume_mode: true
    require_equal_completed_steps: true
  acceptance:
    offset: 0
    exact_match_ratio: 1.0
    require_full_step_pairing: true
  parity_windows:
    - early_steps: [1, 15]
    - full_steps: [1, 100]
```

## References

- `outputs/nanovlm_parity_benchmarks/vanilla-datasettrace-vanilla-fixedwarmup-20260301/summary.json`
- `outputs/nanovlm_parity_benchmarks/soft-gating-datasettrace-softgating-finalcheck-20260301/summary.json`
- `scripts/nanovlm_parity_benchmark.py`
