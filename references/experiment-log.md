# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

<!-- New entries go above this line -->

## 2026-02-27

- **Date**: 2026-02-27
- **Type**: Retrospective
- **General description**: Consolidated parity benchmark evidence for vanilla and soft-gating port tracks.
- **Details**:
  - Vanilla benchmark (`vanilla-vanilla100-vramfix-clean1`, 100 steps):
    - Baseline run `3epjh0uu`: elapsed `356.65s`, peak VRAM `26249 MiB`.
    - Torchtitan run `ea33lgdw`: elapsed `324.78s`, peak VRAM `19995 MiB`.
    - Step-wise loss diff: mean absolute difference `0.0002273`, max absolute difference `0.00104` at step `71`.
  - Soft-gating benchmark (`soft-gating-soft100-vramfix-clean1`, 100 steps):
    - Baseline run `co8wprbr`: elapsed `497.13s`, peak VRAM `28521 MiB`.
    - Torchtitan run `g0ci82mg`: elapsed `392.51s`, peak VRAM `22105 MiB`.
    - Step-wise loss diff: mean absolute difference `0.005222`, max absolute difference `0.16211` at step `9`.
  - Key implementation choices in this cycle:
    - compile both model and loss in paper configs;
    - preserve Torchtitan native compile path;
    - keep external `nvidia-smi` sampling and strict 100-step paired comparison.
  - Evidence quality guardrails:
    - treat runs with zero logged steps as invalid for parity claims (`vanilla-vanilla100-vramfix`);
    - keep speed/VRAM claims separate from loss-parity claims.
  - Outcome vs goals:
    - vanilla: all three goals satisfied (loss parity, faster runtime, lower VRAM);
    - soft-gating: speed and VRAM goals satisfied, loss parity still not satisfied.
