# Non-Packing Vanilla Compile+AC Benchmark (2026-03-13)

## Summary

This benchmark answers a narrower question than the packed-path report:

- dataset path: `HuggingFaceM4/FineVisionMax`
- packing: disabled
- hardware: `1x A100 40GB`
- effective batch size: `64`
- performance policy:
  - `compile=true`
  - activation checkpointing enabled
  - best safe local batch per stack

On this non-packing single-GPU vanilla path, TorchTitan was faster than
`nanoVLM_main`.

## Setup

### nanoVLM_main

- config base: `nanoVLM_main/configs/train.paper.vanilla-finevisionmax.nopack.yaml`
- benchmark config copies:
  - `outputs/nopack_benchmark_compile_ac_20260313/nanovlm_vanilla_nopack_compile_ac_bs32.yaml`
  - `outputs/nopack_benchmark_compile_ac_20260313/nanovlm_vanilla_nopack_compile_ac_bs16.yaml`
  - `outputs/nopack_benchmark_compile_ac_20260313/nanovlm_vanilla_nopack_compile_ac_bs8.yaml`
- overrides:
  - `compile=true`
  - `activation_checkpointing=true`
  - `activation_checkpointing_mode=regular`
  - `max_training_steps=100`
  - `log_wandb=false`
  - checkpoint push disabled

### TorchTitan

- config base: `nanovlm_230m_vanilla_finevisionmax_nopack`
- CLI overrides:
  - `--training.steps 100`
  - `--training.global-batch-size 64`
  - `--activation-checkpoint.mode full`
  - `--metrics.no-enable-wandb`
  - `--metrics.no-enable-tensorboard`
  - `--checkpoint.no-enable`

## Batch Search

### nanoVLM_main

- `bs32, ga2`: failed on first forward
  - error: `torch.OutOfMemoryError`
  - attempted allocation: `186.00 MiB`
  - process memory at failure: `39.22 GiB`
- `bs16, ga4`: failed on first backward
  - error: `torch.OutOfMemoryError`
  - attempted allocation: `6.01 GiB`
- `bs8, ga8`: survived the `300s` validation window and was chosen as the
  best safe setting

### TorchTitan

- `lb32, ga2`: completed `100` validation steps within the `300s` validation
  window and was chosen as the best safe setting
- lower local-batch probes were unnecessary once `lb32` completed cleanly

## Final Timed Runs

### nanoVLM_main

- output dir: `outputs/nopack_benchmark_compile_ac_20260313/nanovlm_bs8_final`
- elapsed wall clock: `364.498s`
- final log summary:
  - `Time: 263.21s`
  - `T/s: 12559.32`

### TorchTitan

- output dir: `outputs/nopack_benchmark_compile_ac_20260313/torchtitan_lb32_final`
- elapsed wall clock: `314.013s`
- final log summary:
  - `step: 100`
  - `loss: 6.88797`
  - `memory: 25.81 GiB`
  - `tps: 53,824`

## Result

Wall-clock comparison:

- `nanoVLM_main`: `364.498s`
- TorchTitan: `314.013s`

Derived speedup:

- `TorchTitan / nanoVLM_main = 1.161x`
- TorchTitan is about `16.1%` faster in wall clock on this benchmark

## Interpretation

- The packed-path slowdown does not generalize to the non-packing FineVisionMax
  vanilla path.
- Under the current compile+AC benchmark policy, TorchTitan gets a much larger
  best safe local batch on this path (`32` vs `8`), and that is enough to edge
  out `nanoVLM_main` in wall clock.
- The win is real but modest. This is not a dramatic speedup.
