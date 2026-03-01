"""
Parallelization for nanoVLM: activation checkpointing, compile, FSDP/DDP.

TP is not supported for MoMH (flex_attention per-head masks don't compose with TP).
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.llama3.parallelize import (
    _op_sac_save_list,
    apply_compile,
    apply_ddp,
    disable_fsdp_gradient_division,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .attention import warmup_flex_attention_compile


def parallelize_nanovlm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply activation checkpointing, torch.compile, and data parallelism to nanoVLM.

    MoMH doesn't support TP (flex_attention per-head masks don't compose with TP).
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor parallelism is not supported for nanoVLM with MoMH attention."
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # 1. Activation checkpointing
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )
        apply_ac(model.vision_encoder, ac_config)

    # 2. Compile flex-attention call paths before model block compile.
    cfg = getattr(model, "config", None)
    if (
        torch.cuda.is_available()
        and cfg is not None
        and bool(getattr(cfg, "momh_enabled", False))
    ):
        n_heads = int(cfg.lm_n_heads)
        hidden_dim = int(cfg.lm_hidden_dim)
        if n_heads <= 0 or hidden_dim <= 0 or hidden_dim % n_heads != 0:
            raise ValueError(
                "Invalid nanoVLM head config for flex-attention warmup: "
                f"n_heads={n_heads}, hidden_dim={hidden_dim}"
            )
        warmup_flex_attention_compile(
            device=torch.device("cuda", torch.cuda.current_device()),
            n_heads=n_heads,
            head_dim=hidden_dim // n_heads,
            # Warm structural flex-attention kernels ahead of training.
            # Keep score_mod warmup disabled to avoid baking synthetic gate
            # values into warmup-specialized compile paths.
            include_soft_gating=False,
        )

    # 3. Use the native Torchtitan per-block compile path.
    if model_compile_enabled:
        apply_compile(model, compile_config)

    # 4. FSDP
    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)

        _apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to nanoVLM")
        else:
            logger.info("Applied FSDP to nanoVLM")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to nanoVLM")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh is not None and dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(model, dp_mesh, enable_compile=model_compile_enabled)

    return model


def _apply_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """Apply FSDP2 with per-block sharding for nanoVLM."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # Token embeddings
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Vision encoder blocks
    for block in model.vision_encoder.blocks.values():
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Modality projector
    fully_shard(
        model.projector,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    # Decoder layers
    for layer in model.layers.values():
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Final norm + output (no reshard for last layers)
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # Top-level model
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division
    disable_fsdp_gradient_division(model)
