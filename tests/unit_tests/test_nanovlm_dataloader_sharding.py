from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest
import torch

from torchtitan.models.nanoVLM.dataloader import ConstantLengthDataset, VQACollator, VQADataset


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


def test_iterable_dataset_passes_packing_queue_size_to_constant_length(monkeypatch):
    captured = {}

    class FakePackedDataset:
        def __iter__(self):
            return iter(())

    class FakeHFDataset(list):
        def shuffle(self, seed):
            del seed
            return self

    class FakeVQADataset:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.mp_image_token_length = 16
            self.tokenizer = SimpleNamespace(pad_token_id=0)

    class FakeDatasetsModule:
        @staticmethod
        def load_dataset(*args, **kwargs):
            del args, kwargs
            return FakeHFDataset([0])

        @staticmethod
        def load_from_disk(*args, **kwargs):
            del args, kwargs
            return FakeHFDataset([0])

        @staticmethod
        def concatenate_datasets(datasets):
            return datasets[0]

        @staticmethod
        def interleave_datasets(datasets, stopping_strategy):
            del stopping_strategy
            return datasets[0]

        @staticmethod
        def get_dataset_config_names(dataset_path):
            del dataset_path
            return ["default"]

    class FakeDistributedModule:
        @staticmethod
        def split_dataset_by_node(dataset, rank, world_size):
            del rank, world_size
            return dataset

    def fake_constant_length_dataset(*args, **kwargs):
        del args
        captured["queue_size"] = kwargs["queue_size"]
        captured["num_of_sequences"] = kwargs["num_of_sequences"]
        return FakePackedDataset()

    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.dataloader.VQADataset",
        FakeVQADataset,
    )
    monkeypatch.setattr(
        "torchtitan.models.nanoVLM.dataloader.ConstantLengthDataset",
        fake_constant_length_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetsModule)
    monkeypatch.setitem(sys.modules, "datasets.distributed", FakeDistributedModule)

    dataset = __import__(
        "torchtitan.models.nanoVLM.dataloader",
        fromlist=["NanoVLMIterableDataset"],
    ).NanoVLMIterableDataset(
        dataset_path="dummy",
        dataset_name=["default"],
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_processor=None,
        mp_image_token_length=16,
        seq_len=32,
        use_packing=True,
        streaming=False,
        packing_num_sequences=8,
        packing_queue_size=4,
        max_sample_length=64,
        max_images_per_example=1,
        max_images_per_knapsack=2,
        relevance_min_rating=1,
        image_correspondence_min_rating=1,
        visual_dependency_min_rating=1,
        formatting_min_rating=1,
        image_token_id=0,
        dp_rank=0,
        dp_world_size=1,
    )

    assert dataset.packed_dataset is not None
    assert captured == {"queue_size": 4, "num_of_sequences": 8}


def test_vqa_collator_returns_flat_image_tensor():
    collator = VQACollator(SimpleNamespace(pad_token_id=0), max_length=4)
    image_a = torch.randn(2, 3, 4, 4)
    image_b = torch.randn(1, 3, 4, 4)

    batch = [
        {
            "input_ids": torch.tensor([1, 2], dtype=torch.long),
            "labels": torch.tensor([1, -100], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "images": [image_a],
        },
        {
            "input_ids": torch.tensor([3], dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "images": [image_b],
        },
    ]

    collated = collator(batch)

    assert collated["input_ids"].shape == (2, 4)
    assert collated["labels"].shape == (2, 4)
    assert collated["attention_mask"].shape == (2, 4)
    assert collated["images"].shape == (3, 3, 4, 4)
    assert torch.equal(collated["images"], torch.cat([image_a, image_b], dim=0))


def test_vqa_collator_returns_empty_image_tensor_for_empty_batch():
    collator = VQACollator(SimpleNamespace(pad_token_id=0), max_length=2)

    collated = collator([None])

    assert collated["input_ids"] == []
    assert collated["labels"] == []
    assert collated["attention_mask"] == []
    assert collated["images"].shape == (0, 3, 1, 1)
