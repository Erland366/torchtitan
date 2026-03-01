---
name: vlm-loss-parity-triangulation
description: >
  Triangulate loss mismatch between TorchTitan and baseline VLM training stacks
  using controlled checks before changing core runtime behavior.
  Use when: speed/VRAM look good but exact loss parity still fails.
metadata:
  short-description: "Loss parity debugging protocol for VLM ports"
  tags:
    - parity
    - loss
    - diagnostics
    - vlm
  domain: research
  created: 2026-03-01
  author: codex
---

# VLM Loss Parity Triangulation

## General Description

This skill defines a repeatable protocol to isolate why losses differ between
TorchTitan and a baseline VLM stack. It avoids premature runtime edits by first
locking down architecture, LR semantics, data ordering, and token normalization.

## When to Apply

Use this knowledge when:
- Throughput and memory targets are met, but loss curves are still different.
- Short debug runs produce contradictory parity conclusions.
- Soft-gating or attention-path variants show early-step drift.

Do NOT use when:
- The two runs are intentionally using different models, checkpoints, or schedules.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Speed in latest A/B | TorchTitan ~2.40x faster | Vanilla short run |
| VRAM in latest A/B | TorchTitan ~20.68% lower | Vanilla short run |
| Exact loss parity | Not yet satisfied | Requires deeper controlled triangulation |

## Recommended Practice

Apply a staged parity protocol from strictest to broadest.

### Step 1: Lock run semantics

- Use same checkpoint source and confirm active loaded architecture.
- Align LR schedule behavior for the exact debug horizon.
- Keep compile scope and mixed-precision settings explicit.

### Step 2: Fixed-batch replay before streaming

- Run a deterministic fixed-batch replay and compare:
  - logits checksum
  - loss
  - grad norm
- Only then move to streaming dataloader comparisons.

### Step 3: Streaming parity with instrumented traces

- Compare step-wise loss, effective tokens, grad norm, and LR per group.
- Record peak VRAM via external sampler with same interval/tooling on both runs.
- Gate acceptance on both early-step window and full-window metrics.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Declaring parity from short-window loss only | Warmup/schedule semantics can dominate tiny runs | Always include LR trace and early-step gate |
| Forcing requested YAML architecture during resume | Baseline may preserve checkpoint architecture on load | Verify active loaded config, not only requested config |

## Configuration

```yaml
parity_protocol:
  stages:
    - fixed_batch_replay
    - short_streaming_ab
    - long_streaming_ab
  required_traces:
    - step_loss
    - lr_by_group
    - grad_norm
    - effective_tokens
    - peak_vram_external
  acceptance:
    early_window_steps: [1, 15]
```

## References

- Related reports: `training_reports/torchtitan-upstream-alignment-retrospective-2026-03-01.md`
- Related skills: `torchtitan-soft-gating-early-step-drift-pattern`, `torchtitan-vanilla-parity-recipe`
- External docs: parity harness and TorchTitan model extension patterns
