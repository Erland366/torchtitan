"""Paper training configurations for nanoVLM."""

import dataclasses
import os

from huggingface_hub import snapshot_download

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.trainer import Trainer

from .. import model_registry
from ..dataloader import NanoVLMDataLoader
from ..optimizer import NanoVLMOptimizersContainer

_NANOVLM_230M_REPO = "lusxvr/nanoVLM-230M-8k"
_MOMH_SOFT_GATING_B5_TTTV_CKPT_ENV = "NANOVLM_SOFT_GATING_INIT_CKPT"
_PACKED_CAULDRON_NUM_SEQUENCES = 8
_PACKED_CAULDRON_QUEUE_SIZE = 4


def _compile_model_and_loss() -> CompileConfig:
    """Use the standard nanoVLM compile recipe for performance runs."""
    return CompileConfig(enable=True, components=["model", "loss"])


def _packed_cauldron_dataloader(max_sample_length: int) -> NanoVLMDataLoader.Config:
    """Use the tuned packed-dataloader recipe for Cauldron pretraining."""
    return NanoVLMDataLoader.Config(
        dataset_path="patrickamadeus/the_cauldron",
        dataset_name=["all"],
        use_packing=True,
        streaming=True,
        max_images_per_example=1,
        max_images_per_knapsack=18,
        max_sample_length=max_sample_length,
        vit_img_size=512,
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=64,
        tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_img_size=2048,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        packing_num_sequences=_PACKED_CAULDRON_NUM_SEQUENCES,
        packing_queue_size=_PACKED_CAULDRON_QUEUE_SIZE,
        relevance_min_rating=1,
        image_correspondence_min_rating=1,
        visual_dependency_min_rating=1,
        formatting_min_rating=1,
    )


def _with_model_overrides(
    cfg: Trainer.Config,
    **model_overrides,
) -> Trainer.Config:
    """Clone model spec with overridden model-config fields."""
    model_cfg = dataclasses.replace(cfg.model_spec.model, **model_overrides)
    cfg.model_spec = dataclasses.replace(cfg.model_spec, model=model_cfg)
    return cfg


def _with_wsm_schedule(
    cfg: Trainer.Config,
    *,
    checkpoint_folder: str,
    checkpoint_interval: int,
) -> Trainer.Config:
    """Convert an existing config into a stable-LR WSM training recipe."""
    cfg.lr_scheduler = dataclasses.replace(
        cfg.lr_scheduler,
        decay_ratio=0.0,
    )
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        folder=checkpoint_folder,
        interval=checkpoint_interval,
    )
    return cfg


def _with_frozen_momh_gate(cfg: Trainer.Config) -> Trainer.Config:
    """Freeze soft-gating parameters out of LM-gradient updates."""
    cfg.optimizer = dataclasses.replace(
        cfg.optimizer,
        lr_momh_gate=0.0,
    )
    return cfg


def _with_low_momh_gate_lr(
    cfg: Trainer.Config,
    *,
    lr: float = 1e-5,
) -> Trainer.Config:
    """Apply a reduced learning rate for `momh_gate` without changing other groups."""
    cfg.optimizer = dataclasses.replace(cfg.optimizer, lr_momh_gate=lr)
    return cfg


def _with_freeze_thaw_momh_gate(
    cfg: Trainer.Config,
    *,
    freeze_steps: int,
) -> Trainer.Config:
    """Freeze `momh_gate` updates early, then thaw back to its configured LR."""
    cfg.optimizer = dataclasses.replace(
        cfg.optimizer,
        momh_gate_freeze_steps=freeze_steps,
    )
    return cfg


def _with_controller_actuation_screen(
    cfg: Trainer.Config,
    *,
    scale: float,
    checkpoint_folder: str,
    **model_overrides,
) -> Trainer.Config:
    """Convert a controller WSM config into a 100-step actuation screen."""
    cfg = _with_model_overrides(
        cfg,
        momh_gate_metrics_interval=10,
        momh_soft_gating_scale=scale,
        **model_overrides,
    )
    cfg.training = dataclasses.replace(
        cfg.training,
        steps=100,
        local_batch_size=32,
        global_batch_size=64,
    )
    cfg.metrics = dataclasses.replace(
        cfg.metrics,
        log_freq=10,
        enable_wandb=True,
    )
    cfg.parallelism = dataclasses.replace(
        cfg.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
    )
    cfg.activation_checkpoint = dataclasses.replace(
        cfg.activation_checkpoint,
        mode="full",
    )
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        enable=False,
        folder=checkpoint_folder,
        interval=100,
    )
    return cfg


def _controller_actuation_screen_config(
    *,
    variant: str,
    scale: float,
    update_rate: float,
) -> Trainer.Config:
    """Build a named Stage-1 controller actuation screen config."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm()
    checkpoint_folder = (
        "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
        f"balance_controller_wsm_screen_{variant}"
    )
    return _with_controller_actuation_screen(
        cfg,
        scale=scale,
        checkpoint_folder=checkpoint_folder,
        momh_balance_update_rate=update_rate,
    )


def _soft_gating_actuation_screen_config(
    cfg: Trainer.Config,
    *,
    variant: str,
    scale: float = 1.0,
    **model_overrides,
) -> Trainer.Config:
    """Build a named Stage-1 soft-gating actuation screen config."""
    checkpoint_folder = (
        "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
        f"screen_{variant}"
    )
    return _with_controller_actuation_screen(
        cfg,
        scale=scale,
        checkpoint_folder=checkpoint_folder,
        **model_overrides,
    )


def _split_warm_screen_config(
    cfg: Trainer.Config,
    *,
    variant: str,
    **model_overrides,
) -> Trainer.Config:
    """Build a Stage-1 screen config with split `tt`/`tv` warm init."""
    return _soft_gating_actuation_screen_config(
        cfg,
        variant=variant,
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=2.0,
        **model_overrides,
    )


def _resolve_hf_repo(repo_id: str) -> str:
    """Resolve an HF repo ID to a local cache directory."""
    return snapshot_download(repo_id)


def _resolve_soft_gating_base_checkpoint() -> str:
    """Resolve soft-gating init checkpoint path without cross-repo dependency.

    If NANOVLM_SOFT_GATING_INIT_CKPT is set, use it directly. Otherwise,
    default to the public nanoVLM-230M-8k HuggingFace checkpoint.
    """
    override = os.getenv(_MOMH_SOFT_GATING_B5_TTTV_CKPT_ENV, "").strip()
    if override:
        return override
    return _resolve_hf_repo(_NANOVLM_230M_REPO)


def nanovlm_230m_structural_gating_finevisionmax_nopack() -> Trainer.Config:
    """230M MoMH structural-gating on FineVisionMax, no packing.

    Ported from: configs/train.paper.momh.structural-gating-finevisionmax.nopack.yaml

    Per-param-group LRs (matching original YAML):
        lr (language backbone) = 1e-4
        lr_vision              = 0       (frozen)
        lr_projector           = 1e-5
        lr_momh_gate           = 0.1

    W&B is configured via environment variables (see .env):
        WANDB_PROJECT=momh
        WANDB_ENTITY=patrickirawan-mbzuai
        WANDB_RUN_NAME=torchtitan-momh-structural-gating-finevisionmax
    """
    hf_local_path = _resolve_hf_repo(_NANOVLM_230M_REPO)

    return Trainer.Config(
        hf_assets_path=hf_local_path,
        model_spec=model_registry("230m_momh_softgating"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=1e-5,
            lr_momh_gate=0.1,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=64,
            seq_len=2048,
            steps=10000,
            max_norm=1.0,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="HuggingFaceM4/FineVisionMax",
            dataset_name=["default"],
            use_packing=False,
            streaming=True,
            max_images_per_example=1,
            max_images_per_knapsack=18,
            max_sample_length=2048,
            vit_img_size=512,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=64,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=2048,
            num_workers=0,
            pin_memory=True,
            relevance_min_rating=1,
            image_correspondence_min_rating=1,
            visual_dependency_min_rating=1,
            formatting_min_rating=1,
        ),
        metrics=MetricsProcessor.Config(log_freq=50, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_structural_gating_finevisionmax_nopack",
            interval=500,
            initial_load_path=hf_local_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=_compile_model_and_loss(),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_vanilla_finevisionmax_nopack() -> Trainer.Config:
    """230M vanilla (no MoMH) on FineVisionMax, no packing.

    Matches nanoVLM_main/configs/train.paper.vanilla-finevisionmax.nopack.yaml:
      - momh_enabled=False
      - activation_checkpointing=False (no AC, same as nanoVLM_main)
      - compile=True (LM decoder blocks only, ViT stays eager)
    """
    hf_local_path = _resolve_hf_repo(_NANOVLM_230M_REPO)

    return Trainer.Config(
        hf_assets_path=hf_local_path,
        model_spec=model_registry("230m_vanilla"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=1e-5,
            lr_momh_gate=0.0,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=64,
            seq_len=2048,
            steps=10000,
            max_norm=1.0,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="HuggingFaceM4/FineVisionMax",
            dataset_name=["default"],
            use_packing=False,
            streaming=True,
            max_images_per_example=1,
            max_images_per_knapsack=18,
            max_sample_length=2048,
            vit_img_size=512,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=64,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=2048,
            num_workers=2,
            prefetch_factor=4,
            pin_memory=True,
            relevance_min_rating=1,
            image_correspondence_min_rating=1,
            visual_dependency_min_rating=1,
            formatting_min_rating=1,
        ),
        metrics=MetricsProcessor.Config(log_freq=50, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_vanilla_finevisionmax_nopack",
            interval=500,
            initial_load_path=hf_local_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=_compile_model_and_loss(),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_vanilla_finevisionmax_nopack_wsm() -> Trainer.Config:
    """230M vanilla FineVisionMax recipe with warmup + stable LR for WSM."""
    cfg = nanovlm_230m_vanilla_finevisionmax_nopack()
    return _with_wsm_schedule(
        cfg,
        checkpoint_folder="checkpoint_nanovlm_230m_vanilla_finevisionmax_nopack_wsm",
        checkpoint_interval=250,
    )


def nanovlm_230m_vanilla_pretrain_cauldron_pack() -> Trainer.Config:
    """230M vanilla pretraining-style run on The Cauldron with packing.

    Mirrors nanoVLM_main/configs/train.current.yaml:
      - dataset: patrickamadeus/the_cauldron (config 'all')
      - packing enabled, seq_len/max_sample_length=8192
      - local batch size 4, global batch size 64 (grad accum 16 on 1 GPU)
      - compile enabled for model and loss
      - model-only init from lusxvr/nanoVLM-230M-8k
    """
    hf_local_path = _resolve_hf_repo(_NANOVLM_230M_REPO)

    return Trainer.Config(
        hf_assets_path=hf_local_path,
        model_spec=model_registry("230m_vanilla_8k"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=5e-4,
            lr_momh_gate=None,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=64,
            seq_len=8192,
            steps=20000,
            max_norm=1.0,
        ),
        dataloader=_packed_cauldron_dataloader(max_sample_length=8192),
        metrics=MetricsProcessor.Config(log_freq=50, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_vanilla_pretrain_cauldron_pack",
            interval=500,
            initial_load_path=hf_local_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=_compile_model_and_loss(),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_vanilla_pretrain_cauldron_pack_2k() -> Trainer.Config:
    """Pretraining-style packed run on The Cauldron at 2k context.

    Purpose: fast parity validation for the pretraining code path (packing
    enabled) without the heavy 8k memory/latency footprint.
    Uses the same compile policy as the main packed 8k run.
    """
    hf_local_path = _resolve_hf_repo(_NANOVLM_230M_REPO)

    return Trainer.Config(
        hf_assets_path=hf_local_path,
        model_spec=model_registry("230m_vanilla"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=5e-4,
            lr_momh_gate=None,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            global_batch_size=32,
            seq_len=2048,
            steps=20000,
            max_norm=1.0,
        ),
        dataloader=_packed_cauldron_dataloader(max_sample_length=2048),
        metrics=MetricsProcessor.Config(log_freq=50, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_vanilla_pretrain_cauldron_pack_2k",
            interval=500,
            initial_load_path=hf_local_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=_compile_model_and_loss(),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack() -> Trainer.Config:
    """230M MoMH soft-gating (tt_tv) baseline, no packing.

    Ported from:
      nanoVLM_main/configs/train.paper.momh.soft-gating-b5-tttv.nopack.yaml
    """
    ckpt_path = _resolve_soft_gating_base_checkpoint()

    cfg = Trainer.Config(
        hf_assets_path=ckpt_path,
        model_spec=model_registry("230m_momh_softgating"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=1e-5,
            lr_momh_gate=None,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            # Match nanoVLM_main get_lr warmup = max_training_steps * 0.005
            # for this 2000-step schedule.
            warmup_steps=10,
            decay_ratio=None,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=64,
            seq_len=2048,
            steps=2000,
            max_norm=1.0,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["all"],
            use_packing=False,
            streaming=True,
            max_images_per_example=1,
            max_images_per_knapsack=18,
            max_sample_length=2048,
            vit_img_size=512,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=64,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=2048,
            num_workers=2,
            pin_memory=True,
            relevance_min_rating=1,
            image_correspondence_min_rating=1,
            visual_dependency_min_rating=1,
            formatting_min_rating=1,
        ),
        metrics=MetricsProcessor.Config(log_freq=50, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack",
            interval=500,
            initial_load_path=ckpt_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=_compile_model_and_loss(),
        debug=DebugConfig(seed=0),
    )
    return cfg


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm() -> Trainer.Config:
    """Soft-gating paper recipe with warmup + stable LR for offline WSM."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_wsm_schedule(
        cfg,
        checkpoint_folder="checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm",
        checkpoint_interval=250,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_global_step() -> Trainer.Config:
    """Soft-gating config with globally synchronized gate metrics every step."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="global",
        momh_gate_metrics_interval=1,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_local_sparse() -> Trainer.Config:
    """Soft-gating config with local (no cross-rank sync) gate metrics every 50 steps."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=50,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_gating_metrics_global_sparse() -> Trainer.Config:
    """Soft-gating config with globally synchronized gate metrics every 50 steps."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="global",
        momh_gate_metrics_interval=50,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux() -> Trainer.Config:
    """Soft-gating config with `tt`/`tv` auxiliary balance loss diagnostics."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=1,
        momh_balance_mode="aux_loss",
        momh_balance_signal="gate_prob",
        momh_balance_target_tv=0.5,
        momh_balance_aux_weight=0.01,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux_wsm() -> Trainer.Config:
    """Soft-gating aux-balance recipe with sparse diagnostics and WSM schedule."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux()
    cfg = _with_model_overrides(
        cfg,
        momh_gate_metrics_interval=50,
    )
    return _with_wsm_schedule(
        cfg,
        checkpoint_folder="checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux_wsm",
        checkpoint_interval=250,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller() -> Trainer.Config:
    """Soft-gating config with non-gradient `tt`/`tv` balance controller."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=1,
        momh_balance_mode="controller",
        momh_balance_signal="gate_prob",
        momh_balance_target_tv=0.5,
        momh_balance_update_rate=0.01,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm() -> Trainer.Config:
    """Soft-gating controller recipe with sparse diagnostics and WSM schedule."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller()
    cfg = _with_model_overrides(
        cfg,
        momh_gate_metrics_interval=50,
        momh_balance_update_rate=0.001,
    )
    return _with_wsm_schedule(
        cfg,
        checkpoint_folder="checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm",
        checkpoint_interval=250,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm() -> Trainer.Config:
    """Soft-gating controller recipe that preserves head specialization."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm()
    return _with_model_overrides(
        cfg,
        momh_balance_signal="layer_mean_gate_prob",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c1() -> Trainer.Config:
    """Stage-1 controller actuation screen reference (`scale=1`, `u=1e-3`)."""
    return _controller_actuation_screen_config(
        variant="c1",
        scale=1.0,
        update_rate=0.001,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c2() -> Trainer.Config:
    """Stage-1 controller actuation screen (`scale=2`, `u=1e-3`)."""
    return _controller_actuation_screen_config(
        variant="c2",
        scale=2.0,
        update_rate=0.001,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c3() -> Trainer.Config:
    """Stage-1 controller actuation screen (`scale=4`, `u=1e-3`)."""
    return _controller_actuation_screen_config(
        variant="c3",
        scale=4.0,
        update_rate=0.001,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_wsm_screen_c4() -> Trainer.Config:
    """Stage-1 controller actuation screen (`scale=2`, `u=2e-3`)."""
    return _controller_actuation_screen_config(
        variant="c4",
        scale=2.0,
        update_rate=0.002,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm() -> Trainer.Config:
    """Stage-1 screen: split `tt`/`tv` warm init without controller."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    return _split_warm_screen_config(
        cfg,
        variant="split_warm",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm_screen_split_warm() -> Trainer.Config:
    """Stage-1 screen: split warm init plus layer-mean controller."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm()
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_layer_mean",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_low_gate_lr() -> Trainer.Config:
    """Stage-1 retention screen: split warm init with reduced gate LR."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_low_gate_lr",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_low_gate_lr_init1() -> Trainer.Config:
    """Stage-1 retention screen: weaker split warm init (`strength=1.0`)."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    return _soft_gating_actuation_screen_config(
        cfg,
        variant="split_warm_low_gate_lr_init1",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=1.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_low_gate_lr_init05() -> Trainer.Config:
    """Stage-1 retention screen: weaker split warm init (`strength=0.5`)."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    return _soft_gating_actuation_screen_config(
        cfg,
        variant="split_warm_low_gate_lr_init05",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=0.5,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_confirm_split_warm_low_gate_lr_init1() -> Trainer.Config:
    """Stage-2 confirmation: weaker split warm init (`strength=1.0`) with low gate LR."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    cfg.training = dataclasses.replace(
        cfg.training,
        steps=300,
        local_batch_size=32,
        global_batch_size=64,
    )
    cfg.metrics = dataclasses.replace(
        cfg.metrics,
        log_freq=25,
        enable_wandb=True,
    )
    cfg.parallelism = dataclasses.replace(
        cfg.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
    )
    cfg.activation_checkpoint = dataclasses.replace(
        cfg.activation_checkpoint,
        mode="full",
    )
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        enable=True,
        folder=(
            "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
            "confirm_split_warm_low_gate_lr_init1"
        ),
        interval=100,
    )
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=25,
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=1.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_split_warm_low_gate_lr_init1() -> Trainer.Config:
    """Long-run WSM recipe: weaker split warm init (`strength=1.0`) with low gate LR."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    cfg.training = dataclasses.replace(cfg.training, steps=3000)
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        folder=(
            "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
            "wsm_split_warm_low_gate_lr_init1"
        ),
        interval=250,
    )
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=50,
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=1.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_freeze_thaw_low_gate_lr() -> Trainer.Config:
    """Stage-1 retention screen: split warm init with early frozen gate, then thaw."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    cfg = _with_freeze_thaw_momh_gate(cfg, freeze_steps=50)
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_freeze_thaw_low_gate_lr",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_split_warm_freeze_thaw_low_gate_lr() -> Trainer.Config:
    """Long-run WSM recipe: split warm init, early freeze-thaw, low gate LR."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    cfg = _with_freeze_thaw_momh_gate(cfg, freeze_steps=50)
    cfg.training = dataclasses.replace(cfg.training, steps=3000)
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        folder=(
            "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
            "wsm_split_warm_freeze_thaw_low_gate_lr"
        ),
        interval=250,
    )
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=50,
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=2.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_split_warm_frozen_gate() -> Trainer.Config:
    """Long-run WSM recipe: split warm init with permanently frozen gate."""
    cfg = _with_frozen_momh_gate(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    )
    cfg.training = dataclasses.replace(cfg.training, steps=3000)
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        folder=(
            "checkpoint_nanovlm_230m_momh_soft_gating_b5_tttv_nopack_"
            "wsm_split_warm_frozen_gate"
        ),
        interval=250,
    )
    return _with_model_overrides(
        cfg,
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_gate_metrics_interval=50,
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=2.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm_screen_split_warm_low_gate_lr() -> Trainer.Config:
    """Stage-1 retention screen: split warm init, low gate LR, gentle layer-mean controller."""
    cfg = _with_low_momh_gate_lr(
        nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm()
    )
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_layer_mean_low_gate_lr",
        momh_balance_update_rate=5e-4,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_tvwarm() -> Trainer.Config:
    """Stage-1 screen: all text-query heads start biased toward `tv`."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    return _soft_gating_actuation_screen_config(
        cfg,
        variant="tvwarm",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
        momh_soft_gating_init="tt_tv_tvwarm",
        momh_soft_gating_init_strength=2.0,
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm_screen_split_warm_frozen_gate() -> Trainer.Config:
    """Stage-1 screen: split warm init with gate frozen out of LM gradients."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_wsm()
    cfg = _with_frozen_momh_gate(cfg)
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_frozen_gate",
        momh_gate_metrics_enabled=True,
        momh_gate_metrics_mode="local",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm_screen_split_warm_frozen_gate() -> Trainer.Config:
    """Stage-1 screen: split warm init, frozen gate, layer-mean controller."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_controller_layer_mean_wsm()
    cfg = _with_frozen_momh_gate(cfg)
    return _split_warm_screen_config(
        cfg,
        variant="split_warm_layer_mean_frozen_gate",
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_2gpu_ddp() -> Trainer.Config:
    """2-GPU DDP variant of soft-gating b5 tttv config."""
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    cfg.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=1,
    )
    return cfg
