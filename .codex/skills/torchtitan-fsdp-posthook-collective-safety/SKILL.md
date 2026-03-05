---
name: torchtitan-fsdp-posthook-collective-safety
description: >
  Prevent FSDP hangs caused by rank-asymmetric post-step metric hooks.
  Use when: model hooks touch DTensor or distributed collectives during/after optimizer step.
metadata:
  short-description: "FSDP-safe post-hook metric collection"
  tags:
    - torchtitan
    - fsdp
    - distributed
    - diagnostics
    - nanovlm
  domain: research
  created: 2026-03-05
  author: codex
---

# TorchTitan FSDP Post-Hook Collective Safety

## General Description

This skill captures a strict safety protocol for post-optimizer hooks in FSDP runs.
It prevents NCCL watchdog hangs by enforcing identical collective ordering across all ranks.

## When to Apply

Use this knowledge when:
- A post-step hook reads model parameters under FSDP or DTensor.
- Hook metrics require optional cross-rank aggregation.
- Training stalls near step 1 with collective timeout signatures.

Do NOT use when:
- The hook is guaranteed local-only and never touches DTensor/distributed collectives.

## Results Summary

| Scenario | Outcome |
|----------|---------|
| Rank-0-only `full_tensor()` in post hook | NCCL timeout and collective mismatch |
| All-rank hook + `to_local()` + symmetric `all_reduce` | Stable 2-GPU FSDP run |
| Per-step global sync for diagnostics | Throughput regression versus sparse/disabled metrics |

## Recommended Practice

- Execute post-step hooks on all ranks.
- Avoid implicit gathers (`full_tensor()`) in hook hot path.
- Use `to_local()` for DTensor-safe local stats.
- If global stats are required, run one symmetric `all_reduce` that every rank enters.
- Make communication-heavy diagnostics optional and sparse by default.
- Publish aggregated diagnostics on rank 0 only after collective completion.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Rank 0 enters gather while other ranks skip hook | Collective ordering diverges | Never gate hook execution by rank before collectives |
| Mixing hook collectives with asymmetric control flow | Some ranks block in NCCL while others continue | Keep hook branching rank-symmetric until collectives complete |
| Per-step global diagnostic reductions | Extra communication in optimizer hot path | Use sparse intervals or disable during throughput runs |

## Configuration

```yaml
momh_gate_metrics:
  enabled: false
  mode: local
  interval: 50
```

## References

- `references/experiment-log.md` (2026-03-05 observation/retrospective)
- `references/troubleshooting.md` (`Soft-gating FSDP stalls before first step on 2 GPUs`)
- `torchtitan/models/nanoVLM/hooks.py`
