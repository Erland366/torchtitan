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
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    step_idx = 0

    def _collect_momh_gate_metrics(*args, **kwargs):
        """Optimizer post-step hook: stash gate statistics on the model."""
        nonlocal step_idx
        step_idx += 1

        for model_part in model_parts:
            extra: dict[str, float] = {}
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            first_block = next(iter(layers.values()), None)
            if first_block is None:
                continue

            metrics_enabled = bool(
                getattr(first_block.attn, "momh_gate_metrics_enabled", False)
            )
            if not metrics_enabled:
                continue

            metrics_mode = str(getattr(first_block.attn, "momh_gate_metrics_mode", "local"))
            if metrics_mode not in {"local", "global"}:
                raise ValueError(
                    "momh_gate_metrics_mode must be one of {'local', 'global'}, "
                    f"got {metrics_mode!r}"
                )
            metrics_interval = int(
                getattr(first_block.attn, "momh_gate_metrics_interval", 50)
            )
            if metrics_interval <= 0:
                raise ValueError(
                    "momh_gate_metrics_interval must be > 0, "
                    f"got {metrics_interval}"
                )
            if step_idx != 1 and (step_idx % metrics_interval != 0):
                continue

            for layer_idx, block in layers.items():
                gate_param = getattr(block.attn, "momh_gate", None)
                if gate_param is None:
                    continue

                gate = gate_param.detach()
                # For DTensor, only read local shard to avoid rank-divergent
                # collectives in optimizer post hooks.
                if hasattr(gate, "to_local"):
                    gate = gate.to_local()
                if gate.numel() == 0:
                    local_sum = torch.zeros(4, device=gate_param.device, dtype=torch.float32)
                    local_count = torch.zeros(1, device=gate_param.device, dtype=torch.float32)
                else:
                    # gate shape: [n_heads, 4] — columns are tt, tv, vt, vv
                    probs = torch.softmax(gate.float(), dim=-1)  # [local_heads, 4]
                    local_sum = probs.sum(dim=0)
                    local_count = torch.tensor(
                        [float(probs.size(0))],
                        device=probs.device,
                        dtype=torch.float32,
                    )

                if metrics_mode == "global" and torch.distributed.is_initialized():
                    stats = torch.cat([local_sum, local_count], dim=0)
                    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                    global_sum = stats[:4]
                    global_count = float(stats[4].item())
                else:
                    global_sum = local_sum
                    global_count = float(local_count.item())

                if rank == 0 and global_count > 0.0:
                    for pair_idx, pair_name in enumerate(pair_names):
                        extra[f"momh_gate/layer_{layer_idx}/{pair_name}_mean"] = float(
                            global_sum[pair_idx].item() / global_count
                        )

            if rank == 0:
                model_part._nanovlm_extra_metrics = extra

    optimizers.register_step_post_hook(_collect_momh_gate_metrics)
