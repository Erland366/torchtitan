"""Paper training configurations for nanoVLM."""

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
_MOMH_SOFT_GATING_B5_TTTV_CKPT = (
    "/home/coder/edd/nanoVLM_root/nanoVLM_main/checkpoints/"
    "momh-gqa-uptrain-paper-a05-pack_nanoVLM_siglip2-base-patch16-512_2048_mp4_"
    "SmolLM2-135M-Instruct_2xGPU_bs128_1000_lr_vision_0.0-language_1e-05-5e-05_"
    "0217-185054/uptraining-result"
)


def _resolve_hf_repo(repo_id: str) -> str:
    """Resolve an HF repo ID to a local cache directory."""
    return snapshot_download(repo_id)


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
        compile=CompileConfig(enable=True, components=["model", "loss"]),
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
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_vanilla_pretrain_cauldron_pack() -> Trainer.Config:
    """230M vanilla pretraining-style run on The Cauldron with packing.

    Mirrors nanoVLM_main/configs/train.current.yaml:
      - dataset: patrickamadeus/the_cauldron (config 'all')
      - packing enabled, seq_len/max_sample_length=8192
      - local batch size 4, global batch size 64 (grad accum 16 on 1 GPU)
      - compile disabled
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
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["all"],
            use_packing=True,
            streaming=True,
            max_images_per_example=1,
            max_images_per_knapsack=18,
            max_sample_length=8192,
            vit_img_size=512,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=64,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=2048,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            relevance_min_rating=1,
            image_correspondence_min_rating=1,
            visual_dependency_min_rating=1,
            formatting_min_rating=1,
        ),
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
        compile=CompileConfig(enable=False, components=["model"]),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_vanilla_pretrain_cauldron_pack_2k() -> Trainer.Config:
    """Pretraining-style packed run on The Cauldron at 2k context.

    Purpose: fast parity validation for the pretraining code path (packing
    enabled) without the heavy 8k memory/latency footprint.
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
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["all"],
            use_packing=True,
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
            persistent_workers=True,
            pin_memory=True,
            relevance_min_rating=1,
            image_correspondence_min_rating=1,
            visual_dependency_min_rating=1,
            formatting_min_rating=1,
        ),
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
        compile=CompileConfig(enable=False, components=["model"]),
        debug=DebugConfig(seed=0),
    )


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack() -> Trainer.Config:
    """230M MoMH soft-gating (tt_tv) baseline, no packing.

    Ported from:
      nanoVLM_main/configs/train.paper.momh.soft-gating-b5-tttv.nopack.yaml
    """
    ckpt_path = _MOMH_SOFT_GATING_B5_TTTV_CKPT

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
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        debug=DebugConfig(seed=0),
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # FSDP sharding hangs at optimizer step on this workload. Prefer DDP for
        # multi-GPU soft-gating runs so the standard launch command remains stable.
        cfg.parallelism = ParallelismConfig(
            data_parallel_replicate_degree=world_size,
            data_parallel_shard_degree=1,
        )
        if cfg.checkpoint.folder:
            cfg.checkpoint.folder = f"{cfg.checkpoint.folder}_ws{world_size}_ddp"
    return cfg


def nanovlm_230m_momh_soft_gating_b5_tttv_nopack_2gpu_ddp() -> Trainer.Config:
    """2-GPU DDP variant of soft-gating b5 tttv config.

    This avoids the observed optimizer-step hang in FSDP sharded mode for this
    workload while keeping all model/data/training hyperparameters unchanged.
    """
    cfg = nanovlm_230m_momh_soft_gating_b5_tttv_nopack()
    cfg.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=1,
    )
    return cfg
