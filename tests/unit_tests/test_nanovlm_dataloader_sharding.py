from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchtitan.models.nanoVLM.dataloader import ConstantLengthDataset, VQADataset


def _make_vqa_dataset(raw_dataset):
    dataset = VQADataset.__new__(VQADataset)
    dataset.dataset = raw_dataset
    dataset.tokenizer = SimpleNamespace(pad_token_id=0)
    dataset.mp_image_token_length = 16
    dataset.max_sample_length = None
    dataset.max_images_per_example = 1
    dataset.relevance_min_rating = 1
    dataset.image_correspondence_min_rating = 1
    dataset.visual_dependency_min_rating = 1
    dataset.formatting_min_rating = 1
    dataset.prefix_len = 0
    dataset._process_data = lambda item: item
    return dataset


def _sample() -> dict[str, torch.Tensor | list]:
    return {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "labels": torch.tensor([1, -100], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "images": [],
    }


class _StreamingShard:
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        return iter(self._values)


class _StreamingDataset:
    def __init__(self, values):
        self.values = values
        self.shard_calls = []

    def shard(self, *, num_shards: int, index: int, contiguous: bool):
        self.shard_calls.append((num_shards, index, contiguous))
        return _StreamingShard(self.values[index::num_shards])

    def __iter__(self):
        return iter(self.values)


def test_vqa_iter_for_worker_shards_map_style_before_processing():
    raw_dataset = list(range(6))
    dataset = _make_vqa_dataset(raw_dataset)

    assert list(dataset.iter_for_worker(worker_id=1, num_workers=2)) == [1, 3, 5]


def test_vqa_iter_for_worker_uses_streaming_shard_before_processing():
    raw_dataset = _StreamingDataset(list(range(8)))
    dataset = _make_vqa_dataset(raw_dataset)

    assert list(dataset.iter_for_worker(worker_id=1, num_workers=2)) == [1, 3, 5, 7]
    assert raw_dataset.shard_calls == [(2, 1, False)]


def test_constant_length_dataset_uses_worker_aware_base_iterator(monkeypatch):
    calls = []

    class FakeDataset:
        mp_image_token_length = 16
        tokenizer = SimpleNamespace(pad_token_id=0)

        def iter_for_worker(self, *, worker_id: int, num_workers: int):
            calls.append((worker_id, num_workers))
            yield _sample()

    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.dataloader.get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=3),
    )

    dataset = ConstantLengthDataset(
        FakeDataset(),
        seq_length=4,
        max_sample_length=8,
        num_of_sequences=1,
    )

    packed = list(dataset)
    assert calls == [(1, 3)]
    assert len(packed) == 1


def test_constant_length_dataset_raises_when_producer_fails():
    class FailingDataset:
        mp_image_token_length = 16
        tokenizer = SimpleNamespace(pad_token_id=0)

        def iter_for_worker(self, *, worker_id: int, num_workers: int):
            del worker_id, num_workers
            raise ValueError("boom")
            yield  # pragma: no cover

    dataset = ConstantLengthDataset(
        FailingDataset(),
        seq_length=4,
        max_sample_length=8,
        num_of_sequences=1,
    )

    with pytest.raises(RuntimeError, match="producer failed"):
        list(dataset)
