# nanoVLM Torchtitan Notes

This module contains the torchtitan port of `nanoVLM_main`.

## Parity Notes For `nanovlm_230m_vanilla_finevisionmax_nopack`

The paper config is set to mirror `nanoVLM_main/configs/train.paper.vanilla-finevisionmax.nopack.yaml` for the most important training dynamics:

- Optimizer defaults aligned with `torch.optim.AdamW(...)` from `nanoVLM_main`:
  - `beta1=0.9`
  - `beta2=0.999`
  - `eps=1e-8`
  - `weight_decay=0.01`
- LR schedule aligned to the original cosine schedule shape:
  - warmup `50` steps (0.5% of 10k)
  - cosine decay
  - minimum LR factor `0.1`
- Compile scope aligned with original behavior:
  - compile model blocks
  - keep loss uncompiled
- Data filtering aligned with original dataset processing:
  - `relevance_ratings`
  - `image_correspondence_ratings`
  - `visual_dependency_ratings`
  - `formatting_ratings`
- Dataloader memory path aligned:
  - `pin_memory=True`

## Throughput Logging Caveat

Using `--metrics.log_freq 1` prints metrics every step and can reduce measured throughput versus the default periodic logging (`log_freq=50`).
