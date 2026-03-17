import torch
import torch.nn as nn

from torchtitan.models.nanoVLM.optimizer import (
    NanoVLMOptimizersContainer,
    _build_param_groups,
)


class _DummyAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.momh_gate = nn.Parameter(torch.zeros(2, 4))
        self.q_proj = nn.Linear(4, 4, bias=False)


class _DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _DummyAttn()
        self.ff = nn.Linear(4, 4, bias=False)


class _DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock()])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4, bias=False)
        self.projector = nn.Linear(4, 4, bias=False)
        self.decoder = _DummyDecoder()


def _build_groups(
    lr_momh_gate: float | None,
    *,
    momh_gate_freeze_steps: int = 0,
):
    model = _DummyModel()
    config = NanoVLMOptimizersContainer.Config(
        name="AdamW",
        lr=1e-4,
        lr_vision=0.0,
        lr_projector=1e-5,
        lr_momh_gate=lr_momh_gate,
        momh_gate_freeze_steps=momh_gate_freeze_steps,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        implementation="for-loop",
    )
    base_kwargs = NanoVLMOptimizersContainer._build_optimizer_kwargs(config)
    return model, _build_param_groups(model, config, base_kwargs)


def test_gate_params_stay_in_lm_group_when_lr_momh_gate_is_none():
    model, groups = _build_groups(lr_momh_gate=None)
    group_map = {group["name"]: group for group in groups}

    assert "momh_gate" not in group_map
    assert "lm" in group_map
    gate_param = model.decoder.blocks[0].attn.momh_gate
    lm_param_ids = {id(param) for param in group_map["lm"]["params"]}
    assert id(gate_param) in lm_param_ids


def test_gate_params_split_when_lr_momh_gate_is_explicit():
    model, groups = _build_groups(lr_momh_gate=0.01)
    group_map = {group["name"]: group for group in groups}

    assert "momh_gate" in group_map
    gate_param = model.decoder.blocks[0].attn.momh_gate
    lm_param_ids = {id(param) for param in group_map["lm"]["params"]}
    gate_param_ids = {id(param) for param in group_map["momh_gate"]["params"]}

    assert id(gate_param) in gate_param_ids
    assert id(gate_param) not in lm_param_ids
    assert group_map["momh_gate"]["lr"] == 0.01


def test_gate_group_can_carry_freeze_thaw_metadata():
    _, groups = _build_groups(lr_momh_gate=0.01, momh_gate_freeze_steps=50)
    group_map = {group["name"]: group for group in groups}

    assert group_map["momh_gate"]["lr"] == 0.01
    assert group_map["momh_gate"]["max_lr"] == 0.01
    assert group_map["momh_gate"]["freeze_steps"] == 50
