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
        self.hook = None

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
    assert metrics["momh_gate_effect/layer_0/scale"] == pytest.approx(1.0)
