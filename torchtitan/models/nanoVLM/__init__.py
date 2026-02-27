"""
nanoVLM — Vision-Language Model with MoMH attention for torchtitan.
"""

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger

from .dataloader import NanoVLMDataLoader
from .hooks import nanovlm_post_optimizer_build_fn
from .model import NanoVLMModel
from .parallelize import parallelize_nanovlm
from .state_dict_adapter import NanoVLMStateDictAdapter

__all__ = [
    "NanoVLMModel",
    "NanoVLMDataLoader",
    "parallelize_nanovlm",
    "NanoVLMStateDictAdapter",
    "nanovlm_configs",
    "nanovlm_post_optimizer_build_fn",
    "model_registry",
]


def nanovlm_cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy loss without fp32 cast to save ~3 GB on large vocabs.

    The default torchtitan loss casts logits to float32 before cross_entropy,
    which doubles GPU memory for the [B, T, V] tensor.  nanoVLM_main doesn't
    cast and works fine; PyTorch's cross_entropy CUDA kernel handles numerical
    stability internally.
    """
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_nanovlm_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs
    loss_fn = nanovlm_cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


nanovlm_configs = {
    "small_debug_momh": NanoVLMModel.Config(
        # Vision
        vit_hidden_dim=384,
        vit_inter_dim=512,
        vit_patch_size=16,
        vit_img_size=256,
        vit_n_heads=6,
        vit_n_blocks=2,
        vit_ln_eps=1e-6,
        vit_cls_flag=False,
        vit_dropout=0.0,
        # Language
        lm_hidden_dim=384,
        lm_inter_dim=512,
        lm_rms_eps=1e-5,
        lm_re_base=100000,
        lm_max_position_embeddings=2048,
        lm_vocab_size=49218,
        lm_n_heads=8,
        lm_n_kv_heads=4,
        lm_n_blocks=4,
        lm_attn_scaling=1.0,
        lm_tie_weights=True,
        lm_dropout=0.0,
        # Projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=16,
        # MoMH
        momh_enabled=True,
        momh_kv_groups_vision=1,
        momh_kv_groups_text=1,
    ),
    "small_debug_momh_softgating": NanoVLMModel.Config(
        # Vision
        vit_hidden_dim=384,
        vit_inter_dim=512,
        vit_patch_size=16,
        vit_img_size=256,
        vit_n_heads=6,
        vit_n_blocks=2,
        vit_ln_eps=1e-6,
        vit_cls_flag=False,
        vit_dropout=0.0,
        # Language
        lm_hidden_dim=384,
        lm_inter_dim=512,
        lm_rms_eps=1e-5,
        lm_re_base=100000,
        lm_max_position_embeddings=2048,
        lm_vocab_size=49218,
        lm_n_heads=8,
        lm_n_kv_heads=4,
        lm_n_blocks=4,
        lm_attn_scaling=1.0,
        lm_tie_weights=True,
        lm_dropout=0.0,
        # Projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=16,
        # MoMH + soft-gating
        momh_enabled=True,
        momh_kv_groups_vision=1,
        momh_kv_groups_text=0,
        momh_soft_gating=True,
        momh_soft_gating_init="zero",
    ),
    # --- 230M (SmolLM2-135M LM + SigLIP2-base ViT) ---
    "230m_momh_softgating": NanoVLMModel.Config(
        # Vision — SigLIP2-base (defaults match, explicit for clarity)
        vit_hidden_dim=768,
        vit_inter_dim=3072,
        vit_patch_size=16,
        vit_img_size=512,
        vit_n_heads=12,
        vit_n_blocks=12,
        # Language — SmolLM2-135M
        lm_hidden_dim=576,
        lm_inter_dim=1536,
        lm_n_heads=9,
        lm_n_kv_heads=3,
        lm_n_blocks=30,
        lm_max_position_embeddings=2048,
        lm_vocab_size=49218,
        lm_tie_weights=True,
        # Projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=64,
        # MoMH + soft-gating
        momh_enabled=True,
        momh_kv_groups_vision=1,
        momh_kv_groups_text=0,
        momh_soft_gating=True,
        momh_soft_gating_init="zero",
        momh_soft_gating_pairs="tt_tv",
    ),
    "230m_vanilla": NanoVLMModel.Config(
        # Vision — SigLIP2-base
        vit_hidden_dim=768,
        vit_inter_dim=3072,
        vit_patch_size=16,
        vit_img_size=512,
        vit_n_heads=12,
        vit_n_blocks=12,
        # Language — SmolLM2-135M
        lm_hidden_dim=576,
        lm_inter_dim=1536,
        lm_n_heads=9,
        lm_n_kv_heads=3,
        lm_n_blocks=30,
        lm_max_position_embeddings=2048,
        lm_vocab_size=49218,
        lm_tie_weights=True,
        # Projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=64,
        # No MoMH
        momh_enabled=False,
    ),
    "230m_vanilla_8k": NanoVLMModel.Config(
        # Vision — SigLIP2-base
        vit_hidden_dim=768,
        vit_inter_dim=3072,
        vit_patch_size=16,
        vit_img_size=512,
        vit_n_heads=12,
        vit_n_blocks=12,
        # Language — SmolLM2-135M
        lm_hidden_dim=576,
        lm_inter_dim=1536,
        lm_n_heads=9,
        lm_n_kv_heads=3,
        lm_n_blocks=30,
        lm_max_position_embeddings=8192,
        lm_vocab_size=49218,
        lm_tie_weights=True,
        # Projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=64,
        # No MoMH
        momh_enabled=False,
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="nanoVLM",
        flavor=flavor,
        model=nanovlm_configs[flavor],
        parallelize_fn=parallelize_nanovlm,
        pipelining_fn=None,
        build_loss_fn=build_nanovlm_cross_entropy_loss,
        post_optimizer_build_fn=nanovlm_post_optimizer_build_fn,
        state_dict_adapter=NanoVLMStateDictAdapter,
    )
