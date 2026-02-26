"""
Post-optimizer hook for nanoVLM: MoMH gate metric collection + W&B x-axis setup.

Follows the same pattern as torchtitan's MoE load-balancing hook in
``torchtitan/components/optimizer.py``.
"""

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def nanovlm_post_optimizer_build_fn(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register a post-step hook that collects MoMH gate metrics each optimizer step."""

    def _has_soft_gating(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for block in layers.values():
                if getattr(block.attn, "momh_soft_gating", False):
                    return True
        return False

    if not _has_soft_gating(model_parts):
        return

    # Set up W&B x-axis on rank 0
    if torch.distributed.get_rank() == 0:
        try:
            import wandb

            if wandb.run is not None:
                wandb.define_metric("*", step_metric="n_tokens_seen")
                logger.info("W&B: set n_tokens_seen as x-axis for all metrics")
        except ImportError:
            pass

    pair_names = ["tt", "tv", "vt", "vv"]

    def _collect_momh_gate_metrics(*args, **kwargs):
        """Optimizer post-step hook: stash gate statistics on the model."""
        if torch.distributed.get_rank() != 0:
            return

        for model_part in model_parts:
            extra = {}
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)

            for layer_idx, block in layers.items():
                gate_param = getattr(block.attn, "momh_gate", None)
                if gate_param is None:
                    continue

                gate = gate_param.detach()
                # If FSDP sharded as DTensor, materialise the full tensor
                if hasattr(gate, "full_tensor"):
                    gate = gate.full_tensor()

                # gate shape: [n_heads, 4] — columns are tt, tv, vt, vv
                probs = torch.softmax(gate, dim=-1)  # [n_heads, 4]
                for pair_idx, pair_name in enumerate(pair_names):
                    extra[f"momh_gate/layer_{layer_idx}/{pair_name}_mean"] = (
                        probs[:, pair_idx].mean().item()
                    )

            model_part._nanovlm_extra_metrics = extra

    optimizers.register_step_post_hook(_collect_momh_gate_metrics)
