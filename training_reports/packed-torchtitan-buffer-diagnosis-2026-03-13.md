# Packed TorchTitan Buffer Diagnosis (2026-03-13)

## Summary

This diagnosis follows the packed single-GPU vanilla benchmark where
TorchTitan was slower than `nanoVLM_main` despite fitting a larger local batch.

The main findings are:

- the packed slowdown is in the steady-state training loop, not compile startup
- lowering `packing_num_sequences` helps packed throughput materially
- matching `packing_queue_size=4` to `nanoVLM_main` is the correct parity fix,
  but it has little effect by itself

## Evidence From Existing Final Runs

### nanoVLM_main

- output dir: `outputs/packing_benchmark_compile_ac_20260313/nanovlm_main_bs2_final`
- shell wall clock: `2097s`
- in-log epoch time: `1999.35s`
- startup + shutdown overhead: about `98s`

### TorchTitan

- output dir: `outputs/packing_benchmark_compile_ac_20260313/torchtitan_bs8_final`
- shell wall clock: `2602s`
- log timestamps show a similar startup/shutdown envelope of roughly `~100s`

Interpretation:

- both stacks pay a similar compile/startup tax
- the packed wall-clock gap comes from the training loop itself

## Short Diagnostics

Common setup:

- `1x A100 40GB`
- `nanovlm_230m_vanilla_pretrain_cauldron_pack`
- `compile=true`
- `activation-checkpoint.mode=full`
- `training.steps=10`
- `global batch = 64`
- `W&B` and TensorBoard disabled

### Baseline packed TorchTitan

- `local_batch_size=8`
- default `packing_num_sequences = local_batch_size * 4 = 32`
- default `packing_queue_size = 2`
- step `10` reached at about `318s` from process start
- memory in the 100-step baseline run later settled around `35.34 GiB`

### Smaller packing buffer

- `local_batch_size=8`
- `--dataloader.packing_num_sequences 8`
- queue size still `2`
- step `10` reached at about `291s` from process start
- memory rose to about `37.66 GiB`

Result:

- about `27s` faster to reach step `10`
- enough to complete the `10`-step run inside the `300s` foreground window

### Queue-depth parity fix

- code change: expose `packing_queue_size` and default it to `4`
- rerun:
  - `local_batch_size=8`
  - `--dataloader.packing_num_sequences 8`
  - default `packing_queue_size=4`
- step `10` reached at about `289s` from process start

Result:

- essentially unchanged from queue size `2`
- queue depth parity is correct, but not the main packed-speed lever

### Smaller local batch

- `local_batch_size=4`
- default packed settings
- did not reach step `10` within the `300s` foreground window

Result:

- lower local batch did not rescue packed throughput

## Conclusion

The packed TorchTitan slowdown is not mainly:

- compile startup
- queue depth
- or simply choosing `lb8` instead of `lb4`

The strongest current explanation is:

- the packed steady-state path is sensitive to `packing_num_sequences`
- the default `local_batch_size * 4` policy is too expensive for this packed
  single-GPU benchmark once `local_batch_size` grows

## Follow-up Result

Follow-up run:

- config-backed `nanovlm_230m_vanilla_pretrain_cauldron_pack`
- `compile=true`
- `activation-checkpoint.mode=full`
- packed defaults after the config change:
  - `packing_num_sequences=8`
  - `packing_queue_size=4`
- best safe batch: `local_batch_size=8`

Result:

- old packed TorchTitan baseline (`packing_num_sequences=32` effective):
  `2602s`
- tuned TorchTitan with CLI override:
  `2330.697s`
- tuned TorchTitan after promoting the setting into the packed paper config:
  `2375s`
- config-backed improvement vs old baseline: about `8.7%`

Updated conclusion:

- `packing_num_sequences` was the right packed-speed lever to promote into the
  paper config
- queue depth parity remains correct but secondary
- even after the fix, packed single-GPU TorchTitan is still slower than the
  `nanoVLM_main` packed baseline (`2097s`)
