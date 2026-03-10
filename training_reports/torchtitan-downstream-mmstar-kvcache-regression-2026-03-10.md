# Report: torchtitan-downstream-mmstar-kvcache-regression

**Date:** 2026-03-10  
**Author:** codex  
**Status:** Completed

## Objective

Restore downstream eval fallback performance after making `torchtitan_nanovlm`
standalone (no runtime import from `nanoVLM_main`) while preserving exact
`mmstar` scores.

## Code Changes

- Added eval-only cached runtime:
  - `torchtitan/eval/nanovlm_cached_runtime.py`
- Wired `torchtitan_nanovlm` backend to use cached generation:
  - `torchtitan/eval/lmms_torchtitan_nanovlm.py`
- Kept orchestration unchanged (`raw-first`, fallback to plugin):
  - `scripts/nanovlm_downstream_eval.py`

Key runtime changes:
- prefill once on full prompt
- decode with per-layer KV cache
- reuse one shared block mask per step across layers

## Commands

### Full regression run

```bash
source /home/coder/edd/nanoVLM_root/nanoVLM_main/.venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path /home/coder/edd/nanoVLM_root/nanoVLM_main/checkpoints/momh-gqa-uptrain-paper-a05-pack_nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-135M-Instruct_2xGPU_bs128_1000_lr_vision_0.0-language_1e-05-5e-05_0217-185054/uptraining-result \
  --checkpoint_format hf \
  --tasks mmstar \
  --batch_size 16 \
  --device cuda \
  --model_backend huggingface \
  --fallback_backend torchtitan_plugin \
  --output_dir eval_results/torchtitan \
  --run_name mmstar-full-kvcache-regression-20260310
```

### Baseline reference

- `eval_results/torchtitan/mmstar-full-20260310`

## Results

### Score parity (baseline vs kvcache regression)

Both runs used `backend_used = torchtitan_nanovlm`.

| Metric | Baseline (`mmstar-full-20260310`) | New (`mmstar-full-kvcache-regression-20260310`) | Delta |
|---|---:|---:|---:|
| average,none | 0.3221761081839732 | 0.3221761081839732 | 0.0 |
| coarse perception,none | 0.49521320592013796 | 0.49521320592013796 | 0.0 |
| fine-grained perception,none | 0.30835789732252517 | 0.30835789732252517 | 0.0 |
| instance reasoning,none | 0.3145030100448731 | 0.3145030100448731 | 0.0 |
| logical reasoning,none | 0.2743012762814743 | 0.2743012762814743 | 0.0 |
| math,none | 0.29092600017820547 | 0.29092600017820547 | 0.0 |
| science & technology,none | 0.24975525935662304 | 0.24975525935662304 | 0.0 |

### Runtime

Fallback duration from metadata:

- Baseline: `2230.8699s`
- New cached runtime: `1205.9187s`

Relative improvement:

- `~45.9%` faster fallback duration (`(2230.8699 - 1205.9187) / 2230.8699`)

## Notes

- Raw `huggingface` backend still fails for this checkpoint type due to missing
  processor assets in local folder; fallback remains required.
- Standalone requirement remains satisfied: no runtime import from
  `nanoVLM_main` wrapper path.
