# nanoVLM Torchtitan Notes

This module contains the torchtitan port of `nanoVLM_main`.

## Parity Notes For `nanovlm_230m_vanilla_finevisionmax_nopack`

The paper config is set to mirror `nanoVLM_main/configs/train.paper.vanilla-finevisionmax.nopack.yaml` for the most important training dynamics:

- Optimizer defaults aligned with `torch.optim.AdamW(...)` from `nanoVLM_main`:
  - `beta1=0.9`
  - `beta2=0.999`
  - `eps=1e-8`
  - `weight_decay=0.01`
  - nanoVLM optimizer container defaults to `implementation="foreach"` to avoid
    accidental fused-backend drift in parity runs when implementation is not
    explicitly pinned in config
- LR schedule aligned with `nanoVLM_main/train.py:get_lr(...)` behavior:
  - per-update LR assignment before optimizer step
  - warmup computed as `max_steps * 0.005`
  - cosine decay to `0.1 * max_lr`
  - manual nanoVLM LR path is selected using optimizer-group `max_lr` metadata
    (not strict group-name matching), so scheduler step ordering remains robust
    if group names evolve
- Compile scope aligned with original behavior:
  - compile model blocks
  - compile loss (fuses logits->CE path and reduces peak reserved VRAM)
  - use Torchtitan native per-block compile path (`apply_compile`)
    rather than a custom nanoVLM forward-wrapper compile
  - warm flex-attention compile paths (prefill/decode, structural/soft-gating)
    before model block compile to stabilize first training steps
- Vision encoder init parity:
  - ViT patch position embedding init uses `uniform_(0, 1)` to match
    `nanoVLM_main` constructor semantics
- Data filtering aligned with original dataset processing:
  - `relevance_ratings`
  - `image_correspondence_ratings`
  - `visual_dependency_ratings`
  - `formatting_ratings`
- Tokenizer/runtime alignment for `resume_from_vlm_checkpoint` parity:
  - tokenizer: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - decoder max positions: `2048`
  - tokenizer `model_max_length` follows `max_sample_length`
- Dataloader memory path aligned:
  - `pin_memory=True`
- Dataloader collation semantics aligned:
  - batch-level `VQACollator` execution (not per-sample pre-collation)
  - preserves the same long-sample filtering behavior as `nanoVLM_main`
    and keeps token/image stream order in sync
- Packing queue sizing aligned:
  - `packing_num_sequences` defaults to `local_batch_size * 4` to mirror
    `nanoVLM_main/ConstantLengthDataset(num_of_sequences=batch_size * 4)`
  - avoids oversized prefetch buffers that previously caused large startup
    latency in packing/pretraining runs

## Throughput Logging Caveat

Using `--metrics.log_freq 1` prints metrics every step and can reduce measured throughput versus the default periodic logging (`log_freq=50`).

## Checkpointing Caveat

For parity runs, avoid reusing a stale checkpoint folder from a previous experiment.
The paper configs now use config-specific checkpoint folders and keep final saves in
native DCP format (`last_save_in_hf=False`) so training resume behavior stays
consistent.

## W&B Env Vars

Torchtitan reads W&B settings from environment variables:

- Team/entity: `WANDB_ENTITY` (or backward-compatible `WANDB_TEAM`)
- Project: `WANDB_PROJECT`
- Run name: `WANDB_RUN_NAME`

## 100-Step A/B Parity Benchmark

Use the parity benchmark runner to execute `nanoVLM_main` and Torchtitan
back-to-back with:

- synchronized 100-step setup
- external `nvidia-smi` VRAM sampling
- parsed per-step loss/tps summaries
- JSON + Markdown output artifacts

Optional benchmark overrides:

- `--torchtitan-config <config_fn>`: choose a specific Torchtitan config
  function instead of the mode default.
- `--torchtitan-extra-args "...args..."`: append extra CLI args to the
  Torchtitan command (useful for sections that are CLI-overridable).

Example (vanilla):

```bash
source ../nanoVLM_main/.venv/bin/activate && \
python scripts/nanovlm_parity_benchmark.py \
  --mode vanilla \
  --steps 100 \
  --wandb-entity patrickirawan-mbzuai \
  --wandb-project momh
```

Example (soft-gating):

```bash
source ../nanoVLM_main/.venv/bin/activate && \
python scripts/nanovlm_parity_benchmark.py \
  --mode soft-gating \
  --steps 100 \
  --wandb-entity patrickirawan-mbzuai \
  --wandb-project momh
```
