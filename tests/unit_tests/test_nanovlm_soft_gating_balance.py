from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.models.nanoVLM.attention import (
    compute_tt_tv_balance_stats,
    generate_soft_gating_score_mod,
    get_tt_tv_pair_logits,
)
from torchtitan.models.nanoVLM.hooks import nanovlm_post_optimizer_build_fn
from torchtitan.models.nanoVLM.model import NanoVLMGQAttention, NanoVLMModel
from torchtitan.trainer import Trainer


def test_compute_tt_tv_balance_stats_is_zero_when_balanced():
    gate = torch.zeros(3, 4, dtype=torch.float32)
    tt_prob, tv_prob, balance_loss = compute_tt_tv_balance_stats(
        gate,
        target_tv=0.5,
    )

    assert torch.allclose(tt_prob, torch.full_like(tt_prob, 0.5))
    assert torch.allclose(tv_prob, torch.full_like(tv_prob, 0.5))
    assert balance_loss.item() == pytest.approx(0.0)


def test_compute_tt_tv_balance_stats_penalizes_tv_skew():
    gate = torch.tensor(
        [[3.0, -3.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    _, tv_prob, balance_loss = compute_tt_tv_balance_stats(
        gate,
        target_tv=0.5,
    )

    assert tv_prob[0].item() < 0.5
    assert tv_prob[1].item() > 0.5
    assert balance_loss.item() > 0.0


def test_get_tt_tv_pair_logits_matches_first_two_columns():
    gate = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float32,
    )

    pair_logits = get_tt_tv_pair_logits(gate)

    assert torch.equal(pair_logits, gate[:, :2])


def test_nanovlm_balance_mode_requires_soft_gating():
    cfg = NanoVLMModel.Config(
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        momh_enabled=True,
        momh_soft_gating=False,
        momh_balance_mode="aux_loss",
        momh_balance_aux_weight=0.1,
    )

    with pytest.raises(ValueError, match="requires momh_soft_gating=True"):
        cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))


def test_nanovlm_balance_mode_requires_tt_tv_pairs():
    cfg = NanoVLMModel.Config(
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        momh_enabled=True,
        momh_soft_gating=True,
        momh_soft_gating_pairs="all",
        momh_balance_mode="controller",
        momh_balance_update_rate=0.01,
    )

    with pytest.raises(ValueError, match="supports only .*'tt_tv'"):
        cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))


def test_nanovlm_soft_gating_scale_must_be_positive():
    cfg = NanoVLMModel.Config(
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        momh_enabled=True,
        momh_soft_gating=True,
        momh_soft_gating_scale=0.0,
    )

    with pytest.raises(ValueError, match="momh_soft_gating_scale must be > 0"):
        cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))


def test_tt_tv_specific_warm_init_requires_tt_tv_pairs():
    cfg = NanoVLMModel.Config(
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        momh_enabled=True,
        momh_soft_gating=True,
        momh_soft_gating_pairs="all",
        momh_soft_gating_init="tt_tv_split_warm",
    )

    with pytest.raises(ValueError, match="tt_tv-specific warm init requires"):
        cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))


def test_nanovlm_soft_gating_init_strength_must_be_positive():
    cfg = NanoVLMModel.Config(
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        momh_enabled=True,
        momh_soft_gating=True,
        momh_soft_gating_pairs="tt_tv",
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=0.0,
    )

    with pytest.raises(
        ValueError, match="momh_soft_gating_init_strength must be > 0"
    ):
        cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))


def test_compute_momh_balance_aux_loss_returns_none_when_disabled():
    attn = NanoVLMGQAttention(
        SimpleNamespace(
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_hidden_dim=16,
            lm_dropout=0.0,
            momh_enabled=True,
            momh_soft_gating=True,
            momh_soft_gating_pairs="tt_tv",
            momh_balance_mode="off",
        )
    )

    assert attn.compute_momh_balance_aux_loss() is None


def test_compute_momh_balance_aux_loss_tracks_gate_proxy():
    attn = NanoVLMGQAttention(
        SimpleNamespace(
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_hidden_dim=16,
            lm_dropout=0.0,
            momh_enabled=True,
            momh_soft_gating=True,
            momh_soft_gating_pairs="tt_tv",
            momh_balance_mode="aux_loss",
            momh_balance_target_tv=0.5,
        )
    )
    with torch.no_grad():
        attn.momh_gate.copy_(
            torch.tensor(
                [
                    [4.0, -4.0, 0.0, 0.0],
                    [4.0, -4.0, 0.0, 0.0],
                    [4.0, -4.0, 0.0, 0.0],
                    [4.0, -4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )

    balance_loss = attn.compute_momh_balance_aux_loss()
    assert balance_loss is not None
    assert balance_loss.item() > 0.0


def _assert_tt_tv_split_warm_gate(gate: torch.Tensor, delta: float) -> None:
    half = gate.shape[0] // 2
    kwargs = {"device": gate.device, "dtype": gate.dtype}
    assert torch.equal(gate[:half, 0], torch.full((half,), delta, **kwargs))
    assert torch.equal(gate[:half, 1], torch.full((half,), -delta, **kwargs))
    assert torch.equal(
        gate[half:, 0], torch.full((gate.shape[0] - half,), -delta, **kwargs)
    )
    assert torch.equal(
        gate[half:, 1], torch.full((gate.shape[0] - half,), delta, **kwargs)
    )


def test_tt_tv_split_warm_initializes_opposing_head_groups():
    attn = NanoVLMGQAttention(
        SimpleNamespace(
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_hidden_dim=16,
            lm_dropout=0.0,
            momh_enabled=True,
            momh_soft_gating=True,
            momh_soft_gating_pairs="tt_tv",
            momh_soft_gating_init="tt_tv_split_warm",
            momh_soft_gating_init_strength=2.0,
        )
    )

    _assert_tt_tv_split_warm_gate(attn.momh_gate.detach(), delta=2.0)


def test_tt_tv_tvwarm_initializes_all_heads_toward_tv():
    attn = NanoVLMGQAttention(
        SimpleNamespace(
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_hidden_dim=16,
            lm_dropout=0.0,
            momh_enabled=True,
            momh_soft_gating=True,
            momh_soft_gating_pairs="tt_tv",
            momh_soft_gating_init="tt_tv_tvwarm",
            momh_soft_gating_init_strength=1.5,
        )
    )

    gate = attn.momh_gate.detach()
    assert torch.equal(gate[:, 0], torch.full((4,), -1.5))
    assert torch.equal(gate[:, 1], torch.full((4,), 1.5))


def test_model_init_weights_reapplies_tt_tv_split_warm():
    cfg = NanoVLMModel.Config(
        vit_hidden_dim=16,
        vit_inter_dim=32,
        vit_patch_size=4,
        vit_img_size=8,
        vit_n_heads=2,
        vit_n_blocks=1,
        vit_cls_flag=False,
        lm_hidden_dim=16,
        lm_inter_dim=32,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_blocks=1,
        lm_vocab_size=32,
        mp_pixel_shuffle_factor=1,
        mp_image_token_length=4,
        momh_enabled=True,
        momh_soft_gating=True,
        momh_soft_gating_pairs="tt_tv",
        momh_soft_gating_init="tt_tv_split_warm",
        momh_soft_gating_init_strength=2.0,
    )
    cfg.update_from_config(trainer_config=SimpleNamespace(dataloader=None))

    model = NanoVLMModel(cfg)
    with torch.no_grad():
        model.layers["0"].attn.momh_gate.zero_()

    model.init_weights()

    _assert_tt_tv_split_warm_gate(model.layers["0"].attn.momh_gate.detach(), delta=2.0)


def test_generate_soft_gating_score_mod_applies_gate_scale():
    momh_gate = torch.tensor([[1.5, -0.5, 0.0, 0.0]], dtype=torch.float32)
    is_vision = torch.tensor([[False, True]], dtype=torch.bool)
    score_mod = generate_soft_gating_score_mod(
        momh_gate=momh_gate,
        is_vision=is_vision,
        active_pairs="tt_tv",
        gate_scale=4.0,
    )

    text_to_text = score_mod(torch.tensor(0.0), 0, 0, 0, 0)
    text_to_vision = score_mod(torch.tensor(0.0), 0, 0, 0, 1)

    assert text_to_text.item() == pytest.approx(6.0)
    assert text_to_vision.item() == pytest.approx(-2.0)


def test_trainer_forward_backward_step_adds_balance_loss_after_token_norm():
    class _DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(momh_balance_aux_weight=0.5)
            self.last_aux = torch.tensor(2.0)

        def forward(self, inputs, **kwargs):
            del kwargs
            return inputs

        def consume_momh_balance_aux_loss(self):
            return self.last_aux

    trainer = Trainer.__new__(Trainer)
    trainer.model_parts = [_DummyModel()]
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.train_context = nullcontext
    trainer.maybe_enable_amp = nullcontext()
    trainer.loss_fn = lambda pred, labels: pred.sum()
    trainer.gradient_accumulation_steps = 4
    trainer.nanovlm_balance_aux_loss_last = None
    trainer.post_dataloading_process = lambda input_dict, labels: (
        input_dict["input"],
        labels,
        {},
        {},
    )

    input_tensor = torch.tensor([[2.0, 2.0]], dtype=torch.float32, requires_grad=True)
    labels = torch.zeros(1, 2, dtype=torch.long)
    loss = Trainer.forward_backward_step(
        trainer,
        input_dict={"input": input_tensor},
        labels=labels,
        global_valid_tokens=torch.tensor(2.0),
    )

    expected = (input_tensor.sum().item() / 2.0) + (2.0 * 0.5 / 4.0)
    assert loss.item() == pytest.approx(expected)
    assert trainer.nanovlm_balance_aux_loss_last == pytest.approx(2.0)


class _HookDummyAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.momh_soft_gating = True
        self.momh_gate = nn.Parameter(
            torch.tensor(
                [[3.0, -3.0, 0.0, 0.0], [3.0, -3.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )
        self.momh_gate_metrics_enabled = True
        self.momh_gate_metrics_mode = "local"
        self.momh_gate_metrics_interval = 1
        self.momh_balance_mode = "controller"
        self.momh_balance_target_tv = 0.5
        self.momh_balance_update_rate = 0.1
        self.momh_balance_signal = "gate_prob"
        self.momh_soft_gating_scale = 1.0


class _HookDummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _HookDummyAttn()


class _HookDummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _HookDummyBlock()})


class _DummyOptimizers:
    def __init__(self) -> None:
        self.pre_hook = None
        self.hook = None
        self.optimizers = []

    def register_step_pre_hook(self, hook):
        self.pre_hook = hook

    def register_step_post_hook(self, hook):
        self.hook = hook


def test_controller_hook_updates_tt_tv_and_emits_balance_metrics():
    optimizers = _DummyOptimizers()
    model = _HookDummyModel()
    nanovlm_post_optimizer_build_fn(
        optimizers,
        [model],
        parallel_dims=SimpleNamespace(),
    )

    assert optimizers.hook is not None
    before = model.layers["0"].attn.momh_gate.detach().clone()
    optimizers.hook()
    after = model.layers["0"].attn.momh_gate.detach().clone()

    assert torch.all(after[:, 0] < before[:, 0])
    assert torch.all(after[:, 1] > before[:, 1])
    assert torch.all(after[:, 2] == before[:, 2])
    assert torch.all(after[:, 3] == before[:, 3])

    metrics = model._nanovlm_extra_metrics
    assert "momh_balance/layer_0/tt_prob_mean" in metrics
    assert "momh_balance/layer_0/tv_prob_mean" in metrics
    assert "momh_gate_effect/layer_0/tt_tv_abs_mean" in metrics
    assert "momh_gate_effect/layer_0/tt_tv_abs_max" in metrics
    assert "momh_gate_effect/layer_0/tt_tv_signed_mean" in metrics
    assert "momh_gate_effect/layer_0/tt_tv_signed_std" in metrics
    assert metrics["momh_gate_effect/layer_0/scale"] == pytest.approx(1.0)


def test_layer_mean_controller_applies_uniform_tt_tv_shift():
    optimizers = _DummyOptimizers()
    model = _HookDummyModel()
    model.layers["0"].attn.momh_gate.data.copy_(
        torch.tensor(
            [[2.0, -2.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
    )
    model.layers["0"].attn.momh_balance_signal = "layer_mean_gate_prob"
    nanovlm_post_optimizer_build_fn(
        optimizers,
        [model],
        parallel_dims=SimpleNamespace(),
    )

    assert optimizers.hook is not None
    before = model.layers["0"].attn.momh_gate.detach().clone()
    optimizers.hook()
    after = model.layers["0"].attn.momh_gate.detach().clone()

    tt_update = after[:, 0] - before[:, 0]
    tv_update = after[:, 1] - before[:, 1]
    assert torch.allclose(tt_update, tt_update[0].expand_as(tt_update))
    assert torch.allclose(tv_update, tv_update[0].expand_as(tv_update))
    assert tt_update[0].item() < 0.0
    assert tv_update[0].item() > 0.0


def test_frozen_gate_metrics_fallback_to_global_reduction(monkeypatch):
    optimizers = _DummyOptimizers()
    model = _HookDummyModel()
    model.layers["0"].attn.momh_gate = nn.Parameter(
        torch.zeros(0, 4, dtype=torch.float32),
        requires_grad=False,
    )
    model.layers["0"].attn.momh_balance_mode = "off"
    model.layers["0"].attn.momh_gate_metrics_mode = "local"

    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    def _fake_is_initialized():
        return True

    def _fake_get_rank():
        return 0

    reduced_sum_stats = torch.tensor(
        [
            1.0,
            1.0,
            0.0,
            0.0,
            2.0,
            1.6,
            0.4,
            0.02,
            -0.6,
            8.0,
            -8.0,
            32.0,
        ],
        dtype=torch.float32,
    )

    def _fake_all_reduce(tensor, op=None):
        del op
        if tensor.numel() == 12:
            tensor.copy_(reduced_sum_stats.to(dtype=tensor.dtype, device=tensor.device))
        elif tensor.numel() == 1:
            tensor.fill_(4.5)
        else:
            raise AssertionError(f"unexpected all_reduce tensor shape: {tuple(tensor.shape)}")

    monkeypatch.setattr(torch.distributed, "is_initialized", _fake_is_initialized)
    monkeypatch.setattr(torch.distributed, "get_rank", _fake_get_rank)
    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "ReduceOp", _FakeReduceOp)

    nanovlm_post_optimizer_build_fn(
        optimizers,
        [model],
        parallel_dims=SimpleNamespace(),
    )

    assert optimizers.hook is not None
    optimizers.hook()

    metrics = model._nanovlm_extra_metrics
    assert metrics["momh_gate_effect/layer_0/tt_tv_abs_mean"] == pytest.approx(4.0)
    assert metrics["momh_gate_effect/layer_0/tt_tv_abs_max"] == pytest.approx(4.5)
    assert metrics["momh_gate_effect/layer_0/tt_tv_signed_mean"] == pytest.approx(-4.0)
    assert metrics["momh_gate_effect/layer_0/tt_tv_signed_std"] == pytest.approx(0.0)


def test_freeze_thaw_pre_hook_zeroes_gate_lr_until_thaw_step():
    optimizers = _DummyOptimizers()
    model = _HookDummyModel()
    optimizers.optimizers = [
        SimpleNamespace(
            param_groups=[
                {
                    "name": "momh_gate",
                    "lr": 1e-5,
                    "max_lr": 1e-5,
                    "freeze_steps": 2,
                }
            ]
        )
    ]
    model.layers["0"].attn.momh_balance_mode = "off"
    nanovlm_post_optimizer_build_fn(
        optimizers,
        [model],
        parallel_dims=SimpleNamespace(),
    )

    assert optimizers.pre_hook is not None
    assert optimizers.hook is not None
    gate_group = optimizers.optimizers[0].param_groups[0]

    optimizers.pre_hook()
    assert gate_group["lr"] == pytest.approx(0.0)
    optimizers.hook()
    gate_group["lr"] = gate_group["max_lr"]

    optimizers.pre_hook()
    assert gate_group["lr"] == pytest.approx(0.0)
    optimizers.hook()
    gate_group["lr"] = gate_group["max_lr"]

    optimizers.pre_hook()
    assert gate_group["lr"] == pytest.approx(1e-5)
