# Packed Vanilla Compile+AC Benchmark (2026-03-13)

## Summary

This report captures the corrected packed vanilla benchmark after fixing two issues in the earlier run:

1. performance benchmarking should not inherit `compile=false` from source configs unless compile-off is the explicit subject
2. validators must not overlap on the same GPU, because that invalidates OOM conclusions

It also includes the follow-up TorchTitan config fix that compiles the packed loss by default.

On the corrected setup, TorchTitan remained slower than `nanoVLM_main` on the single-GPU packed vanilla path.

## Setup

Common intent:
- dataset: `patrickamadeus/the_cauldron`
- packing: enabled
- effective batch size: `64`
- hardware: `1x A100 40GB`
- W&B: disabled for timing cleanliness

`nanoVLM_main`:
- config base: `nanoVLM_main/configs/train.current.yaml`
- benchmark config copy: `outputs/packing_benchmark_compile_ac_20260313/nanovlm_train_current_compile_ac_bs{1,2,4}.yaml`
- compile: `true`
- activation checkpointing: `true`
- activation checkpointing mode: `regular`

`TorchTitan`:
- config base: `nanovlm_230m_vanilla_pretrain_cauldron_pack`
- compile: enabled by config for `model` and `loss`
- activation checkpointing: `full`

## Batch Search

### nanoVLM_main
- `bs4, ga16`: failed on first backward
  - error: `torch.OutOfMemoryError`
  - attempted allocation: `6.01 GiB`
- `bs2, ga32`: stable in validation and chosen as best safe setting

### TorchTitan
- before packed loss-compile fix:
  - `bs8, ga8`: failed on first backward
  - `bs4, ga16`: stable
- after packed loss-compile fix:
  - `bs16, ga4`: failed on first backward
    - attempted allocation: `12.02 GiB`
  - `bs8, ga8`: stable in validation and chosen as the best safe setting

## Packed TorchTitan Config Fix

Files changed:
- `torchtitan/models/nanoVLM/configs/paper.py`
- `torchtitan/models/nanoVLM/README.md`

Change:
- packed vanilla configs now default to:
  - `CompileConfig(enable=True, components=["model", "loss"])`

Observed effect during validation:
- `bs4 + AC full` step-1 memory dropped from about `29.45 GiB` to `20.42 GiB`
- best safe packed TorchTitan batch increased from `bs4` to `bs8`

## Final Timed Runs

### nanoVLM_main
- output dir: `outputs/packing_benchmark_compile_ac_20260313/nanovlm_main_bs2_final`
- elapsed file: `2097`
- final training summary from log:
  - `Time: 1999.35s`
  - `T/s: 15258.53`

### TorchTitan
- original output dir: `outputs/packing_benchmark_compile_ac_20260313/torchtitan_bs8_final`
- original elapsed file: `2602`
- original final step 100 from log:
  - `tps: 20537`
  - `memory: 35.34 GiB`
- tuned config-backed output dir:
  `outputs/packed_config_default_final_20260313`
- tuned elapsed file: `2375`
- tuned final step 100 from log:
  - `memory: 37.66 GiB`

## Result

Wall-clock comparison:
- `nanoVLM_main`: `2097s`
- TorchTitan original: `2602s`
- TorchTitan tuned: `2375s`

Derived speedup:
- original TorchTitan vs `nanoVLM_main`: `0.806x`
- tuned TorchTitan vs `nanoVLM_main`: `0.883x`
- tuned TorchTitan is still `13.3%` slower in wall clock on this benchmark
- tuned TorchTitan improved by about `8.7%` over the original packed baseline

## Retrospective

What worked:
- compile+AC improved the nanoVLM packed baseline materially compared with the earlier compile-off run
- serial validation removed the earlier false OOM caused by overlapping GPU jobs
- compiling the packed TorchTitan loss recovered significant memory headroom and improved the best safe packed batch size
- lowering packed `packing_num_sequences` from the old `local_batch_size * 4`
  behavior to `8` recovered a meaningful packed-speed win on TorchTitan
- promoting the tuned packed buffer policy into the paper config preserved most
  of that win in a clean config-backed rerun

What failed:
- the corrected packed single-GPU benchmark still does not show a TorchTitan speed advantage
- inheriting compile-off from source configs gave an answer that was not aligned with the performance question being asked
- even after the packed loss-compile fix and the packing-buffer tune, TorchTitan
  still loses wall clock on this path

Decision:
- keep `compile=true` as the default performance-benchmark policy in this repo unless the benchmark is explicitly compile-off
- keep activation checkpointing enabled in packed-path comparisons unless the experiment is explicitly about AC ablation
- keep packed TorchTitan configs on `components=["model", "loss"]`
- keep packed Cauldron TorchTitan configs on `packing_num_sequences=8` and
  `packing_queue_size=4`
- avoid using packed single-GPU vanilla as evidence that TorchTitan is faster than `nanoVLM_main`
