"""Debug training configurations for nanoVLM."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    TrainingConfig,
)
from torchtitan.trainer import Trainer

from .. import model_registry
from ..dataloader import NanoVLMDataLoader
from ..optimizer import NanoVLMOptimizersContainer


def nanovlm_small_debug_momh() -> Trainer.Config:
    """Small debug config for nanoVLM with MoMH attention."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("small_debug_momh"),
        tokenizer=None,  # nanoVLM uses its own tokenizer inside the dataloader
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=40,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["sample_1pct"],
            use_packing=True,
            max_images_per_example=1,
            max_images_per_knapsack=8,
            max_sample_length=512,
            vit_img_size=256,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=16,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=1024,
        ),
        metrics=MetricsProcessor.Config(log_freq=5, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=True,
            last_save_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True),
    )


def nanovlm_small_debug_momh_softgating() -> Trainer.Config:
    """Small debug config for nanoVLM with MoMH soft-gating.

    Based on small_debug_momh with:
      - momh_soft_gating=True, momh_soft_gating_init="zero"
      - momh_kv_groups_text=0 (all non-vision heads become shared with soft gates)
    """
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("small_debug_momh_softgating"),
        tokenizer=None,
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=512,
            steps=40,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["sample_1pct"],
            use_packing=True,
            max_images_per_example=1,
            max_images_per_knapsack=8,
            max_sample_length=512,
            vit_img_size=256,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=16,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=1024,
        ),
        metrics=MetricsProcessor.Config(log_freq=5, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=True,
            last_save_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        # flex_attention + soft-gating score_mod causes torch.compile graph breaks
        compile=CompileConfig(enable=False),
    )


def nanovlm_small_debug_streaming() -> Trainer.Config:
    """Small debug config for nanoVLM with streaming dataset.

    Based on small_debug_momh with:
      - streaming=True (uses HF IterableDataset, no full download)
      - checkpoint interval=20 for testing resume
    """
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("small_debug_momh"),
        tokenizer=None,
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=40,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["sample_1pct"],
            use_packing=True,
            streaming=True,
            packing_num_sequences=16,
            max_images_per_example=1,
            max_images_per_knapsack=8,
            max_sample_length=512,
            vit_img_size=256,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=16,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=1024,
        ),
        metrics=MetricsProcessor.Config(log_freq=5, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            interval=20,
            last_save_model_only=True,
            last_save_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True),
    )


def nanovlm_small_debug_momh_softgating_paramgroups() -> Trainer.Config:
    """Debug config for testing NanoVLMOptimizersContainer per-param-group LRs."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("small_debug_momh_softgating"),
        tokenizer=None,
        optimizer=NanoVLMOptimizersContainer.Config(
            lr=1e-4,
            lr_vision=0,
            lr_projector=1e-5,
            lr_momh_gate=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=512,
            steps=40,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["sample_1pct"],
            use_packing=True,
            max_images_per_example=1,
            max_images_per_knapsack=8,
            max_sample_length=512,
            vit_img_size=256,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=16,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=1024,
        ),
        metrics=MetricsProcessor.Config(log_freq=5, enable_wandb=True),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=True,
            last_save_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=False),
    )


def nanovlm_230m_vanilla_finevisionmax_nopack_wsm_debug() -> Trainer.Config:
    """Debug-friendly stable-LR WSM config for nanoVLM 230M vanilla."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
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
            warmup_steps=2,
            decay_ratio=0.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=1,
            seq_len=128,
            steps=8,
            max_norm=1.0,
        ),
        dataloader=NanoVLMDataLoader.Config(
            dataset_path="patrickamadeus/the_cauldron",
            dataset_name=["sample_1pct"],
            use_packing=False,
            streaming=True,
            max_images_per_example=1,
            max_images_per_knapsack=4,
            max_sample_length=128,
            vit_img_size=256,
            mp_pixel_shuffle_factor=4,
            mp_image_token_length=16,
            tokenizer_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_img_size=512,
            num_workers=0,
            pin_memory=True,
        ),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=False),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_nanovlm_230m_vanilla_finevisionmax_nopack_wsm_debug",
            interval=2,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=CompileConfig(enable=False),
    )
