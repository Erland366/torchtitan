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
from torchtitan.models.nanoVLM.attention import (
    compute_tt_tv_balance_stats,
    get_tt_tv_pair_logits,
)


def nanovlm_post_optimizer_build_fn(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register a post-step hook that collects MoMH gate metrics each optimizer step."""
    del parallel_dims

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

    # Set up W&B x-axis on rank 0.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
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

    def _to_local_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        return tensor

    def _zero_stats(device: torch.device) -> tuple[torch.Tensor, ...]:
        zeros4 = torch.zeros(4, device=device, dtype=torch.float32)
        zero1 = torch.zeros(1, device=device, dtype=torch.float32)
        return (
            zeros4,
            zero1,
            zero1.clone(),
            zero1.clone(),
            zero1.clone(),
            zero1.clone(),
            zero1.clone(),
            zero1.clone(),
        )

    def _local_gate_stats(
        gate: torch.Tensor,
        *,
        device: torch.device,
        target_tv: float,
        gate_scale: float,
    ) -> tuple[torch.Tensor, ...]:
        if gate.numel() == 0:
            return _zero_stats(device)

        probs = torch.softmax(gate.float(), dim=-1)
        tt_prob, tv_prob, balance_loss = compute_tt_tv_balance_stats(
            gate,
            target_tv=target_tv,
        )
        pair_logits = get_tt_tv_pair_logits(gate).float()
        abs_pair_delta = torch.abs(pair_logits[:, 1] - pair_logits[:, 0])
        effective_abs_pair_delta = abs_pair_delta * float(gate_scale)
        return (
            probs.sum(dim=0),
            torch.tensor([float(probs.size(0))], device=probs.device, dtype=torch.float32),
            torch.tensor([float(tt_prob.sum().item())], device=probs.device, dtype=torch.float32),
            torch.tensor([float(tv_prob.sum().item())], device=probs.device, dtype=torch.float32),
            torch.tensor(
                [float(balance_loss.item() * probs.size(0))],
                device=probs.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [float((tv_prob - target_tv).sum().item())],
                device=probs.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [float(effective_abs_pair_delta.sum().item())],
                device=probs.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [float(effective_abs_pair_delta.max().item())],
                device=probs.device,
                dtype=torch.float32,
            ),
        )

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
            balance_mode = str(getattr(first_block.attn, "momh_balance_mode", "off"))
            controller_enabled = balance_mode == "controller"
            target_tv = float(first_block.attn.momh_balance_target_tv)
            gate_scale = float(first_block.attn.momh_soft_gating_scale)

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
            should_log_metrics = metrics_enabled and (
                step_idx == 1 or (step_idx % metrics_interval == 0)
            )
            if not should_log_metrics and not controller_enabled:
                continue

            for layer_idx, block in layers.items():
                gate_param = getattr(block.attn, "momh_gate", None)
                if gate_param is None:
                    continue

                gate_data = _to_local_if_needed(gate_param.data)
                if controller_enabled and gate_data.numel() > 0:
                    _, tv_prob_local, _ = compute_tt_tv_balance_stats(
                        gate_data.detach(),
                        target_tv=target_tv,
                    )
                    direction = torch.sign(target_tv - tv_prob_local).to(gate_data.dtype)
                    updates = direction * float(first_block.attn.momh_balance_update_rate)
                    gate_data[:, 0].sub_(updates)
                    gate_data[:, 1].add_(updates)

                if not should_log_metrics:
                    continue

                # For DTensor, only read local shard to avoid rank-divergent
                # collectives in optimizer post hooks.
                gate = _to_local_if_needed(gate_param.detach())
                (
                    local_sum,
                    local_count,
                    local_tt_sum,
                    local_tv_sum,
                    local_balance_sum,
                    local_tv_error_sum,
                    local_effective_abs_delta_sum,
                    local_effective_abs_delta_max,
                ) = _local_gate_stats(
                    gate,
                    device=gate_param.device,
                    target_tv=target_tv,
                    gate_scale=gate_scale,
                )

                if metrics_mode == "global" and torch.distributed.is_initialized():
                    sum_stats = torch.cat(
                        [local_sum, local_count, local_tt_sum, local_tv_sum, local_balance_sum, local_tv_error_sum, local_effective_abs_delta_sum],
                        dim=0,
                    )
                    torch.distributed.all_reduce(
                        sum_stats, op=torch.distributed.ReduceOp.SUM
                    )
                    max_stats = local_effective_abs_delta_max.clone()
                    torch.distributed.all_reduce(
                        max_stats, op=torch.distributed.ReduceOp.MAX
                    )
                    global_sum = sum_stats[:4]
                    global_count = float(sum_stats[4].item())
                    global_tt_sum = float(sum_stats[5].item())
                    global_tv_sum = float(sum_stats[6].item())
                    global_balance_sum = float(sum_stats[7].item())
                    global_tv_error_sum = float(sum_stats[8].item())
                    global_effective_abs_delta_sum = float(sum_stats[9].item())
                    global_effective_abs_delta_max = float(max_stats[0].item())
                else:
                    global_sum = local_sum
                    global_count = float(local_count.item())
                    global_tt_sum = float(local_tt_sum.item())
                    global_tv_sum = float(local_tv_sum.item())
                    global_balance_sum = float(local_balance_sum.item())
                    global_tv_error_sum = float(local_tv_error_sum.item())
                    global_effective_abs_delta_sum = float(
                        local_effective_abs_delta_sum.item()
                    )
                    global_effective_abs_delta_max = float(
                        local_effective_abs_delta_max.item()
                    )

                if rank == 0 and global_count > 0.0:
                    for pair_idx, pair_name in enumerate(pair_names):
                        extra[f"momh_gate/layer_{layer_idx}/{pair_name}_mean"] = float(
                            global_sum[pair_idx].item() / global_count
                        )
                    extra[f"momh_balance/layer_{layer_idx}/tt_prob_mean"] = (
                        global_tt_sum / global_count
                    )
                    extra[f"momh_balance/layer_{layer_idx}/tv_prob_mean"] = (
                        global_tv_sum / global_count
                    )
                    extra[f"momh_balance/layer_{layer_idx}/tv_error_mean"] = (
                        global_tv_error_sum / global_count
                    )
                    extra[f"momh_balance/layer_{layer_idx}/aux_loss_mean"] = (
                        global_balance_sum / global_count
                    )
                    extra[
                        f"momh_gate_effect/layer_{layer_idx}/tt_tv_abs_mean"
                    ] = (global_effective_abs_delta_sum / global_count)
                    extra[
                        f"momh_gate_effect/layer_{layer_idx}/tt_tv_abs_max"
                    ] = global_effective_abs_delta_max
                    extra[f"momh_gate_effect/layer_{layer_idx}/scale"] = gate_scale

            if rank == 0:
                existing = getattr(model_part, "_nanovlm_extra_metrics", None)
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(extra)
                model_part._nanovlm_extra_metrics = merged

    optimizers.register_step_post_hook(_collect_momh_gate_metrics)
