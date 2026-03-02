from types import SimpleNamespace

import pytest
import torch

from torchtitan.trainer import (
    _compute_effective_token_ratio,
    _extract_images_per_sample,
    Trainer,
)


def test_compute_effective_token_ratio_requires_positive_capacity():
    with pytest.raises(ValueError):
        _compute_effective_token_ratio(1, 0)

    assert _compute_effective_token_ratio(5, 20) == pytest.approx(0.25)


def test_extract_images_per_sample_handles_common_layouts():
    nested_images = [[torch.zeros(3, 224, 224)], [torch.zeros(3, 224, 224)]]
    assert _extract_images_per_sample(nested_images, fallback_batch_size=0) == [1, 1]

    stacked_images = torch.zeros(4, 3, 224, 224)
    assert _extract_images_per_sample(stacked_images, fallback_batch_size=0) == [
        1,
        1,
        1,
        1,
    ]

    assert _extract_images_per_sample(None, fallback_batch_size=3) == [0, 0, 0]


def test_collect_nanovlm_model_extra_metrics_clears_cached_values():
    trainer = Trainer.__new__(Trainer)
    model_part = torch.nn.Module()
    model_part._nanovlm_extra_metrics = {"momh_gate/layer_0/tt_mean": 0.5}
    trainer.model_parts = [model_part]

    metrics = Trainer._collect_nanovlm_model_extra_metrics(trainer)
    assert metrics == {"momh_gate/layer_0/tt_mean": pytest.approx(0.5)}
    assert not hasattr(model_part, "_nanovlm_extra_metrics")


def test_build_nanovlm_lr_metrics_maps_group_names():
    trainer = Trainer.__new__(Trainer)
    trainer.optimizers = SimpleNamespace(
        optimizers=[
            SimpleNamespace(
                param_groups=[
                    {"name": "projector", "lr": 1e-3},
                    {"name": "vision", "lr": 2e-4},
                    {"name": "lm", "lr": 5e-5},
                    {"name": "momh_gate", "lr": 1e-1},
                ]
            )
        ]
    )

    metrics = Trainer._build_nanovlm_lr_metrics(trainer)
    assert metrics["training_stats/lr_mp"] == pytest.approx(1e-3)
    assert metrics["training_stats/lr_vision_backbone"] == pytest.approx(2e-4)
    assert metrics["training_stats/lr_language_backbone"] == pytest.approx(5e-5)
    assert metrics["training_stats/momh_gate"] == pytest.approx(1e-1)


def test_load_state_dict_is_backward_compatible_for_effective_tokens():
    trainer = Trainer.__new__(Trainer)
    trainer.step = 0
    trainer.ntokens_seen = 0
    trainer.nanovlm_effective_tokens_seen = 0

    Trainer.load_state_dict(trainer, {"step": 12, "ntokens_seen": 3456})
    assert trainer.step == 12
    assert trainer.ntokens_seen == 3456
    assert trainer.nanovlm_effective_tokens_seen == 0
