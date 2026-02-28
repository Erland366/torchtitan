---
name: torchtitan-parity-harness-resilience
description: >
  Keep paired parity benchmarks valid when baseline exits non-zero during teardown
  after completing all target steps.
  Use when: nanoVLM_main finishes training but the wrapper reports failure at process shutdown.
metadata:
  short-description: "Resilient paired benchmark harness rules"
  tags:
    - research
    - torchtitan
    - nanovlm
    - benchmark
    - parity
    - reliability
  domain: research
  created: 2026-02-28
  author: codex
---

# Torchtitan Parity Harness Resilience

## General Description

This skill captures guardrails for paired nanoVLM-main vs Torchtitan benchmark scripts so valid runs are not discarded because of known baseline teardown crashes.
It keeps strict correctness by allowing continuation only when baseline has already logged the full target steps.

## When to Apply

Use this knowledge when:
- The baseline process exits non-zero after training appears complete.
- Logs contain known teardown signatures after step target is reached.
- You need `summary.json` from both frameworks in one paired run.

Do NOT use when:
- Baseline did not log all target steps.
- Failure signature indicates a real training failure (OOM, NaN, dataloader crash before completion).

## Validated Signatures

Treat as teardown-only (ignorable) only if parsed baseline steps are complete:
- `Fatal Python error: PyGILState_Release`
- `Python runtime state: finalizing`
- `terminate called without an active exception`

## Recommended Practice

1. Parse baseline log immediately after baseline subprocess exits.
2. If return code is non-zero:
   - continue only when:
     - parsed baseline max step >= target steps, and
     - log contains a known teardown-only signature.
   - otherwise stop and report failure.
3. Run Torchtitan leg as usual and produce a single paired summary.
4. Record run validity flags in `summary.json`:
   - baseline full steps
   - torchtitan full steps
   - paired full steps

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Blindly accepting all non-zero baseline exits | Real failures got hidden | Gate on complete-step parse and signature whitelist |
| Stopping on any non-zero baseline exit | Valid paired runs were dropped | Allow teardown-only continuation after full completion |
| Missing structured validity flags | Hard to trust benchmark outputs | Keep explicit validity fields in summary |

## Implementation Anchor

- Script: `scripts/nanovlm_parity_benchmark.py`
- Predicate: `_baseline_can_ignore_return_code(...)`
- Tests: `tests/unit_tests/test_nanovlm_parity_benchmark.py`

