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
  - soft-gating paper config uses `warmup_steps=10` for the 2000-step recipe to
    match `nanoVLM_main` warmup ratio
- Compile scope aligned with original behavior:
  - compile model blocks
  - compile loss (fuses logits->CE path and reduces peak reserved VRAM)
- use Torchtitan native per-block compile path (`apply_compile`)
  rather than a custom nanoVLM forward-wrapper compile
- warm flex-attention compile paths (prefill/decode, structural/soft-gating)
  before model block compile to stabilize first training steps
- keep packed vanilla pretraining configs on the same `["model", "loss"]`
  compile recipe as the faster no-pack paths
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
  - the collator now returns the final flat image tensor directly, removing an
    extra post-collate Python flatten pass in the training hot path
- Dataloader sharding moved ahead of expensive preprocessing:
  - streaming datasets are split by DP rank before image/tokenizer work
  - worker sharding happens before VQA processing for both streaming and
    map-style paths
  - packed-mode workers now use the same worker-aware source iterator
  - avoids duplicated decode/resize/tokenization across ranks and workers and
    fixes the main throughput bottleneck observed in multi-GPU vanilla runs
- FineVisionMax nopack queue tuning updated:
  - the default vanilla FineVisionMax nopack config now uses
    `prefetch_factor=4` with `num_workers=2`
  - on the repaired worker path this improved a 15-step 2-GPU vanilla DDP
    control from `96.85s` to `85.51s` without changing loss or peak VRAM
- Startup dataloader warmup aligned:
  - Torchtitan discards one initial nanoVLM microbatch before optimization to
    mirror `nanoVLM_main` worker warmup behavior
  - avoids one-microbatch stream skew that previously caused loss drift
- Packing queue sizing aligned:
  - packed Cauldron paper configs now pin `packing_num_sequences=8`
    instead of following `local_batch_size * 4`
  - this keeps the packed prefetch buffer small enough to recover steady-state
    throughput on the single-GPU packed vanilla path
  - on the config-backed `100`-step packed benchmark, this reduced TorchTitan
    wall clock from `2602s` to `2375s`
- Packed producer queue depth aligned:
  - `packing_queue_size` now defaults to `4` to match
    `nanoVLM_main/ConstantLengthDataset(queue_size=4)`
  - a short packed diagnostic showed this parity fix is low-impact by itself;
    the stronger throughput lever was lowering `packing_num_sequences`
- Packing producer errors are surfaced:
  - uncaught packing-thread failures now raise immediately in the worker
    iterator instead of leaving the DataLoader blocked on an empty queue

## Throughput Logging Caveat

Using `--metrics.log_freq 1` prints metrics every step and can reduce measured throughput versus the default periodic logging (`log_freq=50`).

## Checkpointing Caveat

For parity runs, avoid reusing a stale checkpoint folder from a previous experiment.
The paper configs now use config-specific checkpoint folders and keep final saves in
native DCP format (`last_save_in_hf=False`) so training resume behavior stays
consistent.

## Offline WSM Notes

The repo now has an offline `WSM` entry point for nanoVLM. In v1 this is
explicitly **not** a trainer rewrite:

- training still uses the existing scheduler stack
- WSM configs implement "warmup then stable LR" by setting `decay_ratio=0.0`
- checkpoint merging happens offline from saved TorchTitan checkpoints

Relevant nanoVLM config variants:

- `nanovlm_230m_vanilla_finevisionmax_nopack_wsm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm`
- `nanovlm_230m_vanilla_finevisionmax_nopack_wsm_debug`

Operational guidance:

- treat merge duration / latest-`N` checkpoint selection as the primary WSM knob
- keep checkpoint folders separate from non-WSM recipes
- use the checkpoint-conversion merge script to emit a merged HF-style checkpoint
  for downstream evaluation
- the merge script writes `wsm_merge_metadata.json` alongside the merged model so
  checkpoint provenance and weights stay reproducible

## FSDP Tied-Weight Note

When `lm_tie_weights=True`, nanoVLM keeps `tok_embeddings`, `norm`, and `output`
inside the same FSDP wrapper. Splitting the tied embedding and LM head across
different FSDP units can break shared-weight assumptions under FSDP.

## HF Export Compatibility Notes

When exporting nanoVLM checkpoints from TorchTitan to HF/nanoVLM safetensors:

- RoPE `inv_freq` is exported as `decoder.rotary_embd.inv_freq` for compatibility
  with nanoVLM strict local loading.
- Tied LM embeddings are exported using a single canonical key
  (`decoder.head.weight`). `decoder.token_embedding.weight` is intentionally not
  duplicated in tied mode, because nanoVLM strict loading can treat that alias as
  an unexpected key.
- Soft-gating checkpoints keep per-layer `decoder.blocks.*.attn.momh_gate`
  tensors in the exported safetensors bundle so TorchTitan downstream fallback
  eval can round-trip those weights back into the native model.
- `scripts/checkpoint_conversion/convert_to_hf.py --model_name nanoVLM ...` now
  emits `config.json` and `model.safetensors` directly in the output folder, so
  `nanoVLM_main/evaluation.py --mode nanovlm --model <output_dir>` works without
  manual file patching.

## W&B Env Vars

Torchtitan reads W&B settings from environment variables:

- Team/entity: `WANDB_ENTITY` (or backward-compatible `WANDB_TEAM`)
- Project: `WANDB_PROJECT`
- Run name: `WANDB_RUN_NAME`

## Logging Parity Notes

TorchTitan now emits nanoVLM-style train metrics for `nanoVLM` runs in addition
to the default TorchTitan metrics:

- `train/consumed_tokens` (baseline-style effective non-pad token count)
- `n_tokens_seen` (TorchTitan raw token-capacity counter)
- `train/batch_loss` (last microbatch loss at update step, reduced across ranks)
- `train/batch_effective_tokens`
- `train/batch_token_capacity`
- `train/batch_effective_token_ratio`
- `train/step_effective_tokens`
- `train/step_token_capacity`
- `train/step_effective_token_ratio`
- `training_stats/*` aggregates:
  - `avg_*` for tokens/s, load/fw-bw/post times, effective token ratio, images/sample
  - `max_*` and `min_images_per_sample`
  - optimizer LRs (`lr_mp`, `lr_vision_backbone`, `lr_language_backbone`, etc.)
  - `training_stats/grad_norm`
- soft-gating `momh_gate/layer_*/{tt,tv,vt,vv}_mean` metrics from the optimizer hook

`train/step_loss` is intentionally not logged.

### Optional MoMH gate metric communication

For soft-gating, MoMH gate metrics now support communication-aware modes:

- `momh_gate_metrics_enabled` (default `False`)
- `momh_gate_metrics_mode`:
  - `"local"`: rank-local gate statistics only (no cross-rank collectives)
  - `"global"`: synchronized gate statistics across ranks
- `momh_gate_metrics_interval` (default `50` optimizer steps)

These fields live in `NanoVLMModel.Config` and are exposed by paper config
variants:

- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack` (default: disabled)
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_global_step`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_local_sparse`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_global_sparse`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux_wsm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c1`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c2`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c3`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c4`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm_screen_split_warm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_tvwarm`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_frozen_gate`
- `nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm_screen_split_warm_frozen_gate`

Operational guidance:
- keep gate metrics disabled for throughput-focused FSDP runs
- use local sparse mode for low-overhead diagnostics
- use global mode only when exact all-rank gate means are required
- the `_wsm` balance variants keep sparse local gate metrics and a checkpoint
  interval tuned for offline merge-window comparisons
- the `_screen_c*` controller variants are `100`-step actuation screens for
  `2x A100` FSDP runs with `activation-checkpoint.mode=full`, `global_batch_size=64`,
  `local_batch_size=32`, checkpointing disabled, and `momh_gate_metrics_interval=10`
- the `split_warm*` screen variants are the next-cycle actuation experiments:
  they test `tt_tv`-specific warm-starts, optional layer-mean controller updates,
  and an optional frozen-gate mode that removes the gate from LM-gradient updates

### Optional soft-gating balance controls

The `tt_tv` soft-gating path now has opt-in balance controls that operate on the
learnable gate proxy instead of instrumenting realized attention mass:

- `momh_soft_gating_init`: `"zero"`, `"warm"`, `"tt_tv_split_warm"`, or `"tt_tv_tvwarm"`
- `momh_soft_gating_init_strength`: magnitude used by the `tt_tv`-specific warm
  starts (`±strength` on the `tt`/`tv` columns)
- `momh_soft_gating_scale`: multiplies the gate bias before it perturbs the
  attention score, so the same learned/controller gate can have a stronger or
  weaker effect on the realized attention logits
- `momh_balance_mode`: `"off"`, `"aux_loss"`, or `"controller"`
- `momh_balance_signal`: `"gate_prob"` for per-head updates or
  `"layer_mean_gate_prob"` for one shared layer-level shift that preserves
  head-to-head specialization
- `momh_balance_target_tv`: target `tv` probability for the active `tt`/`tv` pair
- `momh_balance_aux_weight`: auxiliary-loss weight when mode is `"aux_loss"`
- `momh_balance_update_rate`: non-gradient post-step update size when mode is `"controller"`

Current limitations:
- balance mode requires `momh_soft_gating=True`
- balance mode requires `momh_soft_gating_pairs="tt_tv"`
- balance mode is not supported with pipeline parallelism
- the `tt_tv`-specific warm starts require `momh_soft_gating_pairs="tt_tv"`
- v1 balances the gate proxy only; it does not recover runtime attention-pair usage
- `momh_soft_gating_scale` changes how strongly the gate affects attention, but
  the balance controller still reads the raw `tt`/`tv` gate proxy, so controller
  tuning and attention-strength tuning remain separate knobs
- under FSDP, aux-loss uses the local `momh_gate` shard and normalizes by the
  total layer head count so it avoids mixing DTensor scalars into the trainer's
  main loss path

Logged diagnostics include:
- `train/momh_balance_aux_loss` for auxiliary-loss runs
- `momh_balance/layer_*/tt_prob_mean`
- `momh_balance/layer_*/tv_prob_mean`
- `momh_balance/layer_*/tv_error_mean`
- `momh_balance/layer_*/aux_loss_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_mean`
- `momh_gate_effect/layer_*/tt_tv_abs_max`
- `momh_gate_effect/layer_*/tt_tv_signed_mean`
- `momh_gate_effect/layer_*/tt_tv_signed_std`
- `momh_gate_effect/layer_*/scale`

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

Benchmark warmup behavior:

- By default, the benchmark auto-sets Torchtitan warmup to
  `int(steps * 0.005)` to match `nanoVLM_main` short-run semantics.
- You can still override explicitly with
  `--torchtitan-extra-args "--lr-scheduler.warmup_steps <N>"`.

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

## Manual Downstream Evaluation (TorchTitan)

TorchTitan now has a manual downstream eval entrypoint mirroring the
`nanoVLM_main/evaluation.sh` ergonomics while keeping checkpoint handling in
TorchTitan.

### Quickstart

```bash
source .venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

CKPT_PATH=/abs/path/to/checkpoint_folder ./evaluation_torchtitan.sh
```

Defaults:
- tasks: `coco2017_cap_val,vqav2_val,ocrbench,scienceqa,docvqa_val`
- limit: `2000`
- batch size: `16`
- primary backend: `torchtitan_nanovlm`
- fallback backend: `none`
- outputs: `eval_results/torchtitan/<run_name>/`

Produced artifacts:
- `summary.json`
- `per_task.json`
- `metadata.json`

### Default local backend

Main script:

```bash
source .venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path /abs/path/to/checkpoint \
  --checkpoint_format auto \
  --tasks mmstar \
  --limit 100 \
  --batch_size 8 \
  --output_dir eval_results/torchtitan
```

Behavior:
- runs the TorchTitan-local lmms-eval registration (`torchtitan_nanovlm`) by
  default
- passes `model=/abs/path/to/checkpoint` to the backend instead of relying on
  generic HF auto-loading
- records the primary attempt and chosen backend in `metadata.json`

### Explicit raw-HF compatibility check

If you specifically want to test the generic lmms-eval Hugging Face path, opt
into it explicitly:

```bash
source .venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path /abs/path/to/checkpoint \
  --checkpoint_format auto \
  --tasks mmstar \
  --limit 100 \
  --batch_size 8 \
  --model_backend huggingface \
  --fallback_backend none \
  --output_dir eval_results/torchtitan
```

Use `--fallback_backend torchtitan_plugin` only when you intentionally want a
secondary retry path after a raw-HF failure.

### DCP Checkpoints

If checkpoint format resolves to DCP, the script converts to HF format first
using `scripts/checkpoint_conversion/convert_to_hf.py` and then evaluates.
Use `--keep_converted` to retain the intermediate converted folder.
