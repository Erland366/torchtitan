---
name: torchtitan-dataloader-bound-throughput-diagnosis
description: >
  Diagnose unexpectedly slow TorchTitan training runs by separating dataloader
  bottlenecks from distributed-family bottlenecks before tuning DDP/FSDP further.
metadata:
  short-description: "Check dataloader-bound slowdown first"
  tags:
    - research
    - torchtitan
    - dataloader
    - throughput
    - diagnosis
  domain: research
  created: 2026-03-12
  author: codex
---

# TorchTitan Dataloader-Bound Throughput Diagnosis

## General Description

This skill captures a simple rule: if multi-GPU nanoVLM training feels too
slow, do not assume the main problem is FSDP, DDP, or logging. First check
whether the run is dominated by the dataloader and input pipeline.

For the vanilla `230m` FineVisionMax setup in this repository, the decisive
signal was that `2` GPUs had nearly the same wall clock as `1` GPU, while
TorchTitan's own timing metrics reported `~75%` of step time in data loading on
`2` GPUs.

## When to Apply

Use this knowledge when:
- `2` GPUs feel barely faster than `1` GPU.
- DDP vs FSDP A/B results are weak or inconsistent.
- The model is relatively small and the workload is multimodal or streaming.

Do NOT use when:
- You already know the run is compute-bound from synthetic-data or kernel-only benchmarks.
- The workload does not include expensive input processing.

## Results Summary

| Run | Elapsed (30 steps) | Median TPS excl step 1 | Data Loading % |
|-----|-------------------:|-----------------------:|---------------:|
| `2` GPU DDP, `log_freq=10`, TB on | `220.546s` | `12222` | `74.52%-77.23%` |
| `1` GPU, `log_freq=10`, TB on | `222.983s` | `23612` | `63.60%-64.47%` |

Additional controls:
- `2` GPU DDP, `log_freq=1`, no TB: `209.536s`, median TPS `12838`
- `2` GPU DDP, `log_freq=10`, no TB: `212.726s`, median TPS `12375`

Interpretation:
- logging overhead exists, but it is not the primary bottleneck
- the main bottleneck is input/data loading
- packed-path regressions can survive after worker sharding is fixed; in that
  case, inspect packing-buffer depth (`packing_num_sequences`) separately from
  general worker/prefetch tuning

## Recommended Practice

### Step 1: Run a 1-GPU vs 2-GPU wall-clock control

Keep as much as possible fixed:
- same config
- same local batch size if memory allows
- same global batch size
- same compile setting
- same dataloader worker count

If `2` GPUs barely improve wall clock over `1` GPU, suspect the input path first.

### Step 2: Read TorchTitan's built-in timing metrics

Inspect:
- `time_metrics/end_to_end(s)`
- `time_metrics/data_loading(s)`
- `time_metrics/data_loading(%)`

If `time_metrics/data_loading(%)` is around `>60%`, the run is likely input-bound.
If it is around `>70%`, distributed-family tuning is probably not the main speed lever.

### Step 3: Demote secondary suspects

Before spending time on:
- DDP vs FSDP family tuning
- FSDP reshard policy
- step-wise logging
- TensorBoard or W&B overhead

first prove they are material. In this repo's vanilla case, they were secondary.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Treating DDP vs FSDP as the main speed question | Family tuning changed little while data loading dominated the step | Check input bottlenecks first |
| Assuming `log_freq=1` caused the slow run | `log_freq=1` vs `10` changed little in the 30-step control | Logging overhead was not the main issue |
| Assuming 2 GPUs should automatically speed up training | The input path scaled poorly and erased most of the expected gain | Multi-GPU speedup must be validated end to end |

## Configuration

```yaml
throughput_diagnosis:
  first_checks:
    - compare_1gpu_vs_2gpu_wall_clock
    - inspect_time_metrics_data_loading_pct
  thresholds:
    likely_input_bound_pct: 60
    strongly_input_bound_pct: 70
  defer_until_after_input_check:
    - fsdp_vs_ddp_tuning
    - reshard_policy_tuning
    - logging_overhead_optimization
```

## References

- `training_reports/torchtitan-dataloader-bound-throughput-diagnosis-2026-03-12.md`
- `outputs/diag_speed_20260312/ddp2_vanilla_logfreq10_tb/run.log`
- `outputs/diag_speed_20260312/singlegpu_vanilla_logfreq10_tb/run.log`
