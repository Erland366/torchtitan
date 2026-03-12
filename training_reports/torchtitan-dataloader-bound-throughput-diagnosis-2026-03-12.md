# Report: torchtitan-dataloader-bound-throughput-diagnosis

**Date:** 2026-03-12
**Author:** codex
**Status:** Completed

## Objective

Explain why current nanoVLM TorchTitan distributed runs feel unexpectedly slow,
even after the DDP vs FSDP family benchmark established that the training math
and distributed configuration are broadly correct.

## Setup

### Environment
- Hardware: `2x A100-SXM4-40GB`
- Runtime: `torchrun -m torchtitan.train --module nanoVLM`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming)

### Diagnostic Configuration

- Config: `nanovlm_230m_vanilla_finevisionmax_nopack`
- `training.steps=30`
- `training.local_batch_size=16`
- `training.global_batch_size=64`
- `activation-checkpoint.mode=none`
- `dataloader.num_workers=0`
- compile: enabled

### Runs Compared

1. `2` GPUs, DDP, `metrics.log_freq=1`, no TensorBoard/W&B
2. `2` GPUs, DDP, `metrics.log_freq=10`, no TensorBoard/W&B
3. `2` GPUs, DDP, `metrics.log_freq=10`, TensorBoard enabled
4. `1` GPU, no data parallelism, `metrics.log_freq=10`, TensorBoard enabled

## Experiments

### Run 1: 2-GPU DDP, `log_freq=1`, no TensorBoard

| Metric | Value |
|--------|------:|
| Elapsed (30 steps) | `209.536s` |
| Median TPS excl step 1 | `12838` |
| Final loss | `8.90194` |

### Run 2: 2-GPU DDP, `log_freq=10`, no TensorBoard

| Metric | Value |
|--------|------:|
| Elapsed (30 steps) | `212.726s` |
| Median TPS excl step 1 | `12375` |
| Final loss | `8.90194` |

### Run 3: 2-GPU DDP, `log_freq=10`, TensorBoard enabled

| Metric | Value |
|--------|------:|
| Elapsed (30 steps) | `220.546s` |
| Median TPS excl step 1 | `12222` |
| Final loss | `8.90193` |

Built-in TorchTitan timing metrics:

| Step | `time_metrics/end_to_end(s)` | `time_metrics/data_loading(s)` | `time_metrics/data_loading(%)` |
|------|-----------------------------:|-------------------------------:|-------------------------------:|
| `10` | `4.8258` | `2.0706` | `77.23%` |
| `20` | `5.3400` | `2.0357` | `76.24%` |
| `30` | `5.3814` | `2.0051` | `74.52%` |

### Run 4: 1 GPU, `log_freq=10`, TensorBoard enabled

| Metric | Value |
|--------|------:|
| Elapsed (30 steps) | `222.983s` |
| Median TPS excl step 1 | `23612` |
| Final loss | `8.90195` |

Built-in TorchTitan timing metrics:

| Step | `time_metrics/end_to_end(s)` | `time_metrics/data_loading(s)` | `time_metrics/data_loading(%)` |
|------|-----------------------------:|-------------------------------:|-------------------------------:|
| `10` | `4.9376` | `0.8733` | `63.67%` |
| `20` | `5.5510` | `0.8826` | `63.60%` |
| `30` | `5.5513` | `0.8947` | `64.47%` |

## Analysis

### What Worked
- The built-in TorchTitan timing metrics were sufficient to diagnose the
  bottleneck without adding new instrumentation.
- The 1-GPU vs 2-GPU control cleanly separated input-pipeline limitations from
  distributed-family questions.

### Root Cause Identified
- Raw dataset sharding was happening too late.
- In streaming mode, rank and worker assignment occurred after expensive sample
  processing, so multiple ranks/workers repeated image decode, resize/split, and
  tokenizer work before discarding most samples.
- In packed mode, worker-aware sharding was bypassed entirely and every worker
  could pack from the full upstream stream.
- The packing producer thread also failed silently on uncaught exceptions, which
  could leave a worker blocked on `queue.get()` forever.

### What Failed
- `2` GPUs did not provide meaningful end-to-end wall-clock improvement over
  `1` GPU in this configuration.
- Adjusting `metrics.log_freq` did not materially improve throughput.

### Key Insights
1. The current vanilla run is input-bound, not primarily distributed-bound.
2. On `2` GPUs, roughly `75%` of the measured step window is spent in data
   loading under `num_workers=0`.
3. The same run on `1` GPU still spends about `64%` of its time in data
   loading, so the bottleneck already exists before distributed overhead is
   added.
4. DDP vs FSDP tuning cannot recover the missing speed while the dataloader
   dominates the step time.
5. `metrics.log_freq=1` and TensorBoard logging add some overhead, but they are
   not the main explanation for the slow end-to-end wall clock.

## Fix Validation

Implemented changes:
- shard streaming datasets by DP rank before VQA processing using
  `datasets.distributed.split_dataset_by_node(...)`
- shard worker inputs before VQA processing for both map-style and streaming
  paths
- route packed-mode iteration through worker-aware `iter_for_worker(...)`
- surface packing producer thread failures back to the main iterator instead of
  hanging silently

Validation run:
- config: `nanovlm_230m_vanilla_finevisionmax_nopack`
- `training.steps=20`
- `2` GPUs, DDP, `metrics.log_freq=10`, no TensorBoard/W&B

| Setting | Elapsed | Median TPS excl step 1 | Final loss | Status |
|---------|--------:|-----------------------:|-----------:|--------|
| `num_workers=0` | `133.383s` | `18306.5` | `9.01889` | completed |
| `num_workers=2` | `109.381s` | `46916.0` | `9.41222` | completed |

Observed outcome:
- the `num_workers=2` run completed cleanly with no worker stall or NCCL timeout
- elapsed improved by `24.002s` over `20` steps (`~18.0%` faster)
- this validates that the previous worker path was structurally wrong, not just
  poorly tuned

## Decision

- Treat the dataloader/input path as the main optimization target for current
  nanoVLM TorchTitan training speed.
- Do not use further DDP vs FSDP benchmarking as the primary path to recover
  vanilla throughput until the input bottleneck is addressed.

## Next Steps

- [ ] Make `num_workers > 0` stable again for nanoVLM TorchTitan.
- [ ] Isolate whether the dominant cost is streaming fetch, image decoding,
      image transforms, or collation/packing.
- [ ] Re-run the same 1-GPU vs 2-GPU control after dataloader changes.

## Appendix

- Output directories:
  - `outputs/diag_speed_20260312/ddp2_vanilla_logfreq1`
  - `outputs/diag_speed_20260312/ddp2_vanilla_logfreq10_notb`
  - `outputs/diag_speed_20260312/ddp2_vanilla_logfreq10_tb`
  - `outputs/diag_speed_20260312/singlegpu_vanilla_logfreq10_tb`
