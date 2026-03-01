"""Per-param-group optimizer for nanoVLM.

Classifies model parameters into groups with independent learning rates:
  - vision:    vision_encoder params
  - projector: modality projector params
  - momh_gate: MoMH soft-gating gate params
  - lm:        everything else (tok_embeddings, decoder layers, norm, output)

Each group gets its own LR. The LR scheduler applies relative scaling,
so per-group ratios are preserved throughout training.
"""

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.logging import logger


class NanoVLMOptimizersContainer(OptimizersContainer):

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        lr_vision: float | None = None
        """Learning rate for vision encoder. None = use base lr."""

        lr_projector: float | None = None
        """Learning rate for modality projector. None = use base lr."""

        lr_momh_gate: float | None = None
        """Learning rate for MoMH gate parameters. None = use base lr."""

        implementation: Literal["for-loop", "foreach", "fused"] = "foreach"
        """Default to foreach for nanoVLM parity with nanoVLM_main AdamW behavior."""

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        base_kwargs = self._build_optimizer_kwargs(config)

        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        for model in self.model_parts:
            param_groups = _build_param_groups(model, config, base_kwargs)
            params_in_groups = [p for g in param_groups for p in g["params"]]

            # lr lives in each param group; remove from top-level kwargs
            optim_kwargs = {k: v for k, v in base_kwargs.items() if k != "lr"}
            self.optimizers.append(optimizer_cls(param_groups, **optim_kwargs))
            all_params.extend(params_in_groups)

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, base_kwargs)


def _build_param_groups(
    model: nn.Module,
    config: NanoVLMOptimizersContainer.Config,
    base_kwargs: dict,
) -> list[dict]:
    """Classify model parameters into optimizer groups with per-component LRs."""
    vision_params = []
    projector_params = []
    gate_params = []
    lm_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_encoder."):
            vision_params.append(param)
        elif name.startswith("projector."):
            projector_params.append(param)
        elif ".momh_gate" in name:
            # Match nanoVLM_main semantics:
            # - lr_momh_gate is None: gate params stay in LM group
            # - lr_momh_gate is set:  gate params get their own group
            if config.lr_momh_gate is None:
                lm_params.append(param)
            else:
                gate_params.append(param)
        else:
            lm_params.append(param)

    base_lr = config.lr
    groups = []

    specs: list[tuple[str, list[nn.Parameter], float | None]] = [
        ("lm", lm_params, base_lr),
        ("vision", vision_params, config.lr_vision),
        ("projector", projector_params, config.lr_projector),
    ]
    if config.lr_momh_gate is not None:
        specs.append(("momh_gate", gate_params, config.lr_momh_gate))

    for group_name, params, lr in specs:
        if not params:
            continue
        lr = lr if lr is not None else base_lr
        if lr == 0:
            # Freeze completely: no gradients, no optimizer states.
            # Matches nanoVLM_main behavior — saves optimizer state memory
            # and avoids backward through frozen modules (e.g. ViT).
            for p in params:
                p.requires_grad = False
            logger.info(
                f"Frozen group '{group_name}': {len(params)} params "
                f"(lr=0 → requires_grad=False)"
            )
            continue
        group = {
            "params": params,
            "lr": lr,
            "name": group_name,
            "max_lr": lr,
        }
        groups.append(group)
        logger.info(
            f"Optimizer group '{group_name}': {len(params)} params, lr={lr}"
        )

    return groups
