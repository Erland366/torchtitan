---
name: torchtitan-checkpoint-isolation-parity-runs
description: >
  Prevent false parity failures caused by checkpoint-state reuse across changed
  trainer or dataloader state schemas.
  Use when: parity runs fail at startup with missing checkpoint keys or resume unexpectedly.
metadata:
  short-description: "Isolate checkpoint folders for clean parity A/B runs"
  tags:
    - research
    - torchtitan
    - nanovlm
    - parity
    - checkpoint
  domain: research
  created: 2026-03-04
  author: codex
---

# Torchtitan Checkpoint Isolation For Parity Runs

## General Description

This skill prevents invalid parity conclusions caused by stale checkpoint folders.
When trainer/dataloader state keys change between code revisions, reusing an old
`dump_folder` can fail before training starts or silently resume from the wrong state.

## When to Apply

Use this knowledge when:
- A parity run fails on startup with checkpoint-key errors.
- Recent refactors changed trainer, dataloader, or state dict fields.
- You need deterministic A/B comparison from step 1.

Do NOT use when:
- You are intentionally validating checkpoint backward compatibility.

## Results Summary

| Symptom | Root Cause | Resolution |
|---------|------------|------------|
| `Missing key in checkpoint state_dict: dataloader.dp_rank_0.` | Reused checkpoint folder from a run with different saved-state schema | Launch with fresh `--dump_folder` per run pair |
| Startup load ambiguity in parity A/B | Folder already contains prior checkpoints | Isolate output roots by run id and compare only clean step-1 starts |

## Recommended Practice

### Step 1: Isolate every parity run output root

- Use a unique `--dump_folder` for each baseline and Torchtitan run pair.
- Do not reuse folders across commits or config changes.

### Step 2: Enforce clean-start semantics

- Verify logs show training starts at step `1`.
- Verify logs do not load from an existing local checkpoint unless explicitly intended.

### Step 3: Keep resume tests separate

- If compatibility testing is needed, run it as a dedicated experiment.
- Do not mix resume-compatibility checks with parity-speed-VRAM checks.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Reused old dump folder after trainer-state refactor | State keys no longer matched checkpoint payload | Fresh output folders are mandatory after state-schema edits |
| Compared resumed run against clean-start baseline | Step trajectories were not equivalent from step 1 | Parity A/B must share identical start semantics |

## Configuration

```yaml
parity_run_hygiene:
  checkpoint_isolation: true
  require_fresh_dump_folder: true
  require_training_start_step: 1
  disallow_resume_in_parity_ab: true
```

## References

- `outputs/manual_loss_checks/simplify4_1gpu_steps10.log`
- `outputs/manual_loss_checks/simplify4_2gpu_steps10.log`
- `references/troubleshooting.md`
