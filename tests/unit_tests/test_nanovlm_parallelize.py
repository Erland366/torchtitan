from types import SimpleNamespace

import torch.nn as nn

from torchtitan.models.nanoVLM.parallelize import _apply_fsdp


class _DummyVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({"block_0": nn.Linear(4, 4)})


class _DummyNanoVLM(nn.Module):
    def __init__(self, *, tie_weights: bool):
        super().__init__()
        self.config = SimpleNamespace(lm_tie_weights=tie_weights)
        self.vision_encoder = _DummyVisionEncoder()
        self.projector = nn.Linear(4, 4)
        self.layers = nn.ModuleDict({"layer_0": nn.Linear(4, 4)})
        self.tok_embeddings = nn.Embedding(8, 4)
        self.norm = nn.LayerNorm(4)
        self.output = nn.Linear(4, 8, bias=False)

        if tie_weights:
            self.output.weight = self.tok_embeddings.weight


def test_apply_fsdp_keeps_tied_embedding_and_output_in_one_group(monkeypatch):
    model = _DummyNanoVLM(tie_weights=True)
    shard_targets = []

    def fake_fully_shard(target, **kwargs):
        shard_targets.append(target)

    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.parallelize.fully_shard", fake_fully_shard
    )
    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.parallelize.disable_fsdp_gradient_division",
        lambda model: None,
    )

    _apply_fsdp(
        model,
        dp_mesh=object(),
        param_dtype=None,
        reduce_dtype=None,
        pp_enabled=False,
    )

    grouped_targets = [
        target for target in shard_targets if isinstance(target, list) and len(target) == 3
    ]
    assert grouped_targets == [[model.tok_embeddings, model.norm, model.output]]
    assert model.tok_embeddings not in shard_targets
    assert [model.norm, model.output] not in shard_targets


def test_apply_fsdp_keeps_untied_embedding_and_output_separate(monkeypatch):
    model = _DummyNanoVLM(tie_weights=False)
    shard_targets = []

    def fake_fully_shard(target, **kwargs):
        shard_targets.append(target)

    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.parallelize.fully_shard", fake_fully_shard
    )
    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.parallelize.disable_fsdp_gradient_division",
        lambda model: None,
    )

    _apply_fsdp(
        model,
        dp_mesh=object(),
        param_dtype=None,
        reduce_dtype=None,
        pp_enabled=False,
    )

    assert model.tok_embeddings in shard_targets
    assert [model.norm, model.output] in shard_targets
