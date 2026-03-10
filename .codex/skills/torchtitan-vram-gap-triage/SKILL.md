---
name: torchtitan-vram-gap-triage
description: >
  Triage VRAM deltas after Torchtitan parity and speed are achieved.
  Use when: you need to separate true regressions from run-harness artifacts in memory measurements.
metadata:
  short-description: "Methodical VRAM-gap triage for Torchtitan ports"
  tags:
    - research
    - torchtitan
    - vram
    - profiling
  domain: research
  created: 2026-02-27
  author: codex
---

# Torchtitan VRAM Gap Triage

## General Description

This skill captures a repeatable method to investigate VRAM differences between baseline nanoVLM and Torchtitan after loss parity is already validated.  
It avoids false conclusions by separating startup failures and harness drift from true training-phase memory behavior.

## When to Apply

Use this knowledge when:
- You already have stable 100-step loss parity.
- Torchtitan is equal or faster in throughput and memory conclusions disagree across runs.
- You need evidence-driven memory triage before changing architecture-level behavior.

Do NOT use when:
- Baseline and Torchtitan runs are not matched on dataset, steps, or batch settings.
- You only have partial logs from failed startup runs.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Earlier observed vanilla gap | +668 MiB | `vanilla-vanilla100-tmux`: Torchtitan `26917 MiB` vs baseline `26249 MiB` |
| Latest validated vanilla delta | -6254 MiB | `vanilla-vanilla100-vramfix-clean1`: Torchtitan `19995 MiB` vs baseline `26249 MiB` |
| 2-GPU FSDP AC delta (vanilla) | -4484 MiB | `full` AC vs `none` at fixed global batch `64` |
| 2-GPU FSDP AC delta (soft-gating) | -4364 MiB | `full` AC vs `none` at fixed global batch `64` |
| Interpretation | configuration-sensitive | Enforce run-validity checks before drawing VRAM conclusions |

## Recommended Practice

Use layered ablations and keep one change per run.

### Step 1: Establish reliable measurement

- Use an external `nvidia-smi` sampler at 1s intervals during the whole run.
- Persist elapsed time, peak memory, and return status in a dedicated summary file.
- Always keep run logs for correlation.

### Step 2: Separate memory phases

- Identify when peak occurs (`AT_SEC`) relative to total run time.
- Determine if peak is:
  - training-phase persistent plateau, or
  - tail behavior near checkpoint/export.

### Step 3: Run focused ablations

- `checkpoint.on` vs `checkpoint.off`
- `activation_checkpoint.mode=none` vs `full` at fixed global batch
- `metrics.log_freq` changes
- compile-path changes that keep numerics intact
- optimizer backend (`foreach` vs `fused`) only if parity remains acceptable
- reject runs where either side does not log all target steps

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Treating network-failed runs as memory evidence | Run never entered stable training path | Exclude failed/partial runs from memory conclusions |
| Changing multiple knobs in one experiment | Impossible attribution | Use one-axis ablations only |
| Using only framework-reported memory | Misses transient/allocator behavior | Keep external sampler as source of truth for peak memory |
| Comparing AC modes with different effective batch | Memory/speed conclusions become confounded | Keep global batch fixed and let local batch/GA be the only mode-dependent values |

## Configuration

```bash
# Example external sampler wrapper pattern
START=$(date +%s)
(while true; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/run.mem
  sleep 1
done) & MON=$!

# run training command here ...

END=$(date +%s)
kill "$MON" || true
echo "ELAPSED_SEC=$((END-START))" > /tmp/run.time
awk 'BEGIN{m=0}{if($1>m)m=$1}END{print "MAX_MEM_MIB=" m}' /tmp/run.mem >> /tmp/run.time
```

## References

- Related log: `references/experiment-log.md` (2026-02-27 retrospective entry)
- Related troubleshooting: `references/troubleshooting.md`
- Related benchmark summaries:
  - `torchtitan/outputs/nanovlm_parity_benchmarks/vanilla-vanilla100-tmux/summary.json`
  - `torchtitan/outputs/nanovlm_parity_benchmarks/vanilla-vanilla100-vramfix-clean1/summary.json`
  - `torchtitan/outputs/ac_benchmarks/max-batch-ac-compare-2gpu-20260305-fullplan/summary.json`
