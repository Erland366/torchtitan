# Report: torchtitan-downstream-mmstar-eval

**Date:** 2026-03-10
**Author:** codex
**Status:** Completed

## Objective

Validate the new TorchTitan manual downstream evaluation path and capture the
`mmstar` score for the current nanoVLM checkpoint.

Primary checks:
- downstream eval command path works end-to-end from TorchTitan repo
- output artifacts are deterministic (`summary.json`, `per_task.json`, `metadata.json`)
- backend provenance is explicit (raw attempt vs fallback)

## Setup

### Checkpoint

- `/home/coder/edd/nanoVLM_root/nanoVLM_main/checkpoints/momh-gqa-uptrain-paper-a05-pack_nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-135M-Instruct_2xGPU_bs128_1000_lr_vision_0.0-language_1e-05-5e-05_0217-185054/uptraining-result`

### Command

```bash
source /home/coder/edd/nanoVLM_root/nanoVLM_main/.venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path /home/coder/edd/nanoVLM_root/nanoVLM_main/checkpoints/momh-gqa-uptrain-paper-a05-pack_nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-135M-Instruct_2xGPU_bs128_1000_lr_vision_0.0-language_1e-05-5e-05_0217-185054/uptraining-result \
  --checkpoint_format hf \
  --tasks mmstar \
  --batch_size 16 \
  --device cuda \
  --output_dir eval_results/torchtitan \
  --run_name mmstar-full-20260310
```

### Runtime behavior

- Raw backend configured: `huggingface`
- Fallback configured: `torchtitan_plugin`
- Final backend used: `torchtitan_nanovlm`

## Results

Artifact directory:
- `eval_results/torchtitan/mmstar-full-20260310`

Main metric (`per_task.json`):
- `mmstar average,none = 0.3221761082`

Subscores:
- coarse perception: `0.4952132059`
- fine-grained perception: `0.3083578973`
- instance reasoning: `0.3145030100`
- logical reasoning: `0.2743012763`
- math: `0.2909260002`
- science & technology: `0.2497552594`

Timing (`metadata.json`):
- raw attempt duration: `21.682s` (failed)
- fallback attempt duration: `2230.870s` (succeeded)

## Analysis

### What worked

- New TorchTitan eval entrypoint completed full `mmstar` and emitted all expected JSON artifacts.
- Backend provenance was preserved in metadata, enabling clear interpretation.

### What failed

- Raw `huggingface` backend failed for this checkpoint because the local folder
  does not include processor assets required by `AutoProcessor.from_pretrained`.

### Practical takeaway

For this checkpoint class, use raw-first with fallback enabled as default.
Do not interpret scores without checking `metadata.backend_used`.

## Follow-up

- Added result skill:
  - `.codex/skills/torchtitan-downstream-eval-raw-first-fallback/SKILL.md`
- Updated guardrail skill:
  - `.codex/skills/torchtitan-upstream-alignment-guardrail/SKILL.md`
- Added troubleshooting pattern for raw backend processor-missing failure.
