---
name: torchtitan-packed-buffer-tuning
description: >
  Recover packed TorchTitan throughput by tuning packing buffer depth before
  blaming compile startup or queue depth. Use when packed Cauldron or
  pretraining-style runs fit memory but are still slower than expected.
metadata:
  short-description: "Tune packed buffer depth first"
  tags:
    - research
    - torchtitan
    - packing
    - dataloader
    - throughput
  domain: research
  created: 2026-03-13
  author: codex
---

# TorchTitan Packed Buffer Tuning

## General Description

Packed single-GPU slowdown in this repo was not primarily caused by compile
startup or producer queue depth. The main lever was the packing buffer depth:
reducing `packing_num_sequences` from the old effective `local_batch_size * 4`
policy to `8` improved the `100`-step packed TorchTitan wall clock from
`2602s` to `2330.697s`.

## When to Apply

Use this knowledge when:
- packed TorchTitan runs fit memory but still lose wall clock badly
- non-packed runs are healthy, so the regression is isolated to the packed path
- logs suggest the slowdown is inside the steady-state train loop rather than startup

Do NOT use when:
- the run is clearly OOM-limited before step 1
- the bottleneck is already known to be outside the dataloader/packing path

## Results Summary

| Packed Run | Wall Clock | Notes |
|-----------|-----------:|------|
| original TorchTitan packed baseline | `2602s` | effective `packing_num_sequences=32` at `local_batch_size=8` |
| tuned TorchTitan packed run | `2330.697s` | `packing_num_sequences=8`, `packing_queue_size=4` |
| packed `nanoVLM_main` baseline | `2097s` | still faster than tuned TorchTitan |

Interpretation:
- `packing_num_sequences` was the meaningful packed-speed lever
- `packing_queue_size=4` is the correct parity cleanup, but it was low-impact by itself

## Recommended Practice

### Step 1: Prove the problem is steady-state, not startup

Compare shell wall clock against in-log training time. If both stacks pay
similar startup tax, do not spend more time on compile-startup hypotheses.

### Step 2: Reduce packed buffer depth before reducing local batch

Try:
- `packing_num_sequences=8`

before spending more time on:
- lowering local batch
- producer queue depth experiments
- blaming compile startup

### Step 3: Keep queue-depth parity, but treat it as secondary

Use:
- `packing_queue_size=4`

to match `nanoVLM_main` parity expectations, but do not expect it to recover
most of the packed throughput gap by itself.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Lowering local batch alone | Throughput was still poor | Packed slowdown was not simply a local-batch-size problem |
| Matching queue depth alone | Step-time barely changed | Queue depth parity was correct but secondary |
| Blaming compile startup | Startup tax was similar across stacks | The packed gap was inside the train loop |

## Configuration

```yaml
packed_throughput_policy:
  primary_lever:
    packing_num_sequences: 8
  parity_cleanup:
    packing_queue_size: 4
  do_not_assume:
    - compile_startup_is_the_main_gap
    - lower_local_batch_is_enough
```

## References

- `training_reports/packed-torchtitan-buffer-diagnosis-2026-03-13.md`
- `training_reports/packed-vanilla-compile-ac-benchmark-2026-03-13.md`
