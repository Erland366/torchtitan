---
name: torchtitan-upstream-alignment-guardrail
description: >
  Keep shared TorchTitan runtime code aligned with upstream while preserving
  nanoVLM-specific behavior through model-local extension points.
  Use when: porting model behavior into TorchTitan and preventing long-term drift.
metadata:
  short-description: "Prefer upstream shared runtime, isolate model quirks"
  tags:
    - torchtitan
    - parity
    - maintainability
    - checkpoint
  domain: research
  created: 2026-03-01
  author: codex
---

# TorchTitan Upstream Alignment Guardrail

## General Description

This skill captures a strict rule for TorchTitan ports: shared runtime code
(`trainer`, `checkpoint`, `metrics`) should stay close to upstream, and
model-specific behavior should live in adapters/hooks/protocol extensions.
Use this when parity work starts adding many special cases into shared runtime.

## When to Apply

Use this knowledge when:
- A custom model port starts modifying shared TorchTitan runtime files.
- Upstream mergeability is becoming difficult because of local runtime forks.
- Required behavior can be expressed in model-local APIs instead.

Do NOT use when:
- Upstream runtime itself must change for all models and there is no model-local surface.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Shared runtime drift reduction | High | `trainer.py` reverted to upstream behavior in this cycle |
| Compatibility retained | Yes | HF `_orig_mod` key remap preserved via nanoVLM adapter hook |

## Recommended Practice

Keep shared runtime close to upstream and introduce explicit extension points for model-only behavior.

### Step 1: Revert shared runtime first

- Diff against upstream commit for:
  - `torchtitan/trainer.py`
  - `torchtitan/components/checkpoint.py`
  - `torchtitan/components/metrics.py`
- Remove local logic unless it is truly framework-wide.

### Step 2: Re-home model-specific behavior

- Add adapter or hook methods in protocol/model modules.
- Keep custom checkpoint remapping in model adapter, not checkpoint manager core.
- Keep model-specific optimizer/LR nuances in model optimizer container/hooks.
- Keep model-specific metrics/statistics helpers out of `trainer.py` core
  by moving them to model-local helper modules or mixins under
  `torchtitan/models/<model_name>/`.

### Step 3: Enforce explicit acceptance checks

- Check that shared trainer changes remain mostly orchestration-level and do not
  introduce model-specific branches in common runtime loops.
- Require that model-specific behavior can be disabled by removing the model-local
  adapter/hook/helper module, without touching shared runtime files.
- Validate one short run after refactor and confirm no new compile-recompile
  regressions were introduced.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Runtime patched with model-specific key filtering | Shared code became brittle and diverged | Move key adaptation into model adapter hook |
| Attempted architecture-shape forcing in TorchTitan config | Checkpoint tensor shape mismatches on load | Treat active loaded checkpoint shape as source of truth unless explicitly migrating weights |

## Configuration

```yaml
port_policy:
  shared_runtime: upstream_first
  model_specific_behavior: adapter_or_hook_only
  acceptance:
    - no_new_model_specific_branches_in_trainer
    - no_new_model_specific_branches_in_checkpoint_manager
    - trainer_diff_is_orchestration_only
    - model_metrics_logic_lives_under_models_tree
```

## References

- Related reports: `training_reports/torchtitan-upstream-alignment-retrospective-2026-03-01.md`
- Related skills: `torchtitan-vanilla-parity-recipe`, `torchtitan-soft-gating-parity-guardrails`
- External docs: upstream TorchTitan patterns and model extension points
