"""
Self-contained data pipeline for nanoVLM in torchtitan.

Ported from nanoVLM_main/data/ — tokenizer setup, image processing,
VQA dataset, constant-length packing, collation.
"""

import io
import itertools
import logging
import math
import os
import random
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms.functional import resize, InterpolationMode

from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.tools.logging import logger

random.seed(42)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========================== Image Utilities ==========================


def _normalize_numpy_image(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported numpy image shape: {image.shape}.")
    arr = image
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size > 0 else 0.0
            if max_value <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def coerce_image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(_normalize_numpy_image(image)).convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as img:
            return img.convert("RGB")
    if isinstance(image, dict):
        for key in ("image", "array", "bytes", "path"):
            if image.get(key) is not None:
                return coerce_image_to_pil(image[key])
        raise ValueError("Unsupported image dict payload.")
    raise ValueError(f"Unsupported image type: {type(image)}.")


# ========================== Image Transforms ==========================


class DynamicResize(nn.Module):
    def __init__(self, patch_size: int, max_side_len: int, resize_to_max_side_len: bool = False):
        super().__init__()
        self.p = int(patch_size)
        self.m = int(max_side_len)
        self.resize_to_max_side_len = resize_to_max_side_len

    def _get_new_hw(self, h: int, w: int):
        long, short = (w, h) if w >= h else (h, w)
        target_long = self.m if self.resize_to_max_side_len else min(self.m, math.ceil(long / self.p) * self.p)
        scale = target_long / long
        target_short = math.ceil(short * scale / self.p) * self.p
        target_short = max(target_short, self.p)
        return (target_short, target_long) if w >= h else (target_long, target_short)

    def forward(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
        if torch.is_tensor(img):
            batched = img.ndim == 4
            imgs = img if batched else img.unsqueeze(0)
            _, _, h, w = imgs.shape
            new_h, new_w = self._get_new_hw(h, w)
            out = resize(imgs, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
            return out if batched else out.squeeze(0)
        raise TypeError(f"Unsupported type: {type(img)}")


class SplitImage(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        n_h, n_w = h // self.p, w // self.p
        # Rearrange: b c (nh ph) (nw pw) -> (b nh nw) c ph pw
        x = x.reshape(b, c, n_h, self.p, n_w, self.p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(b * n_h * n_w, c, self.p, self.p)
        return x, (n_h, n_w)


class GlobalAndSplitImages(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size
        self.splitter = SplitImage(patch_size)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        patches, grid = self.splitter(x)
        if grid == (1, 1):
            return patches, grid
        global_patch = resize(x, [self.p, self.p])
        return torch.cat([global_patch, patches], dim=0), grid


# ========================== Tokenizer Setup ==========================

import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

VLM_EXTRA_TOKENS = {
    "image_token": "<|image|>",
    "global_image_token": "<|global_image|>",
}
# Add grid tokens
for r in range(1, 9):
    for c in range(1, 9):
        VLM_EXTRA_TOKENS[f"r{r}c{c}"] = f"<row_{r}_col_{c}>"


def get_tokenizer(name, extra_special_tokens=None, chat_template=None, model_max_length=None):
    if name not in TOKENIZERS_CACHE:
        from transformers import AutoTokenizer

        kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    tokenizer = TOKENIZERS_CACHE[name]
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    return tokenizer


def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])


def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f"r{i+1}c{j+1}")
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string


# ========================== Dataset ==========================


LM_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


class VQADataset:
    """Visual Question Answering dataset processor."""

    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        mp_image_token_length,
        max_sample_length: int | None = None,
        max_images_per_example=1,
        relevance_min_rating=1,
        image_correspondence_min_rating=1,
        visual_dependency_min_rating=1,
        formatting_min_rating=1,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_sample_length = max_sample_length
        self.max_images_per_example = max_images_per_example
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string = "xzyvd"
        templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string}],
            tokenize=False,
            add_special_tokens=False,
        )
        loc = templated.find(random_string)
        return len(self.tokenizer.encode(templated[:loc]))

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _raw_data_source_for_worker(self, worker_id: int, num_workers: int):
        data_source = self.dataset
        if num_workers > 1 and hasattr(data_source, "shard"):
            return data_source.shard(
                num_shards=num_workers,
                index=worker_id,
                contiguous=False,
            )
        return data_source

    def iter_for_worker(
        self,
        *,
        worker_id: int = 0,
        num_workers: int = 1,
    ):
        data_source = self._raw_data_source_for_worker(worker_id, num_workers)

        if hasattr(data_source, "__len__"):
            all_indices = range(len(data_source))
            worker_indices = (
                itertools.islice(all_indices, worker_id, None, num_workers)
                if num_workers > 1
                else all_indices
            )

            for idx in worker_indices:
                yield self._process_data(data_source[idx])
            return

        data_iterator = iter(data_source)
        consecutive_fetch_failures = 0
        try:
            while True:
                try:
                    data = next(data_iterator)
                    consecutive_fetch_failures = 0
                except StopIteration:
                    break
                except Exception as exc:
                    consecutive_fetch_failures += 1
                    logger.warning(
                        "Streaming dataset fetch failed (%s/%s): %s",
                        consecutive_fetch_failures,
                        32,
                        exc,
                    )
                    if consecutive_fetch_failures >= 32:
                        raise RuntimeError(
                            "Too many consecutive streaming dataset fetch failures."
                        ) from exc
                    continue

                yield self._process_data(data)
        finally:
            close_fn = getattr(data_iterator, "close", None)
            if callable(close_fn):
                close_fn()

    def _process_data(self, item):
        if item["images"] is None:
            images_data = []
        else:
            images_data = item["images"]
            if not isinstance(images_data, list):
                images_data = [images_data]

        processed_images = []
        splitted_image_counts = []
        if images_data:
            for image in images_data:
                pil_image = coerce_image_to_pil(image)
                processed_image, splitted_image_count = self.image_processor(pil_image)
                if (
                    not hasattr(self.tokenizer, "global_image_token")
                    and splitted_image_count[0] * splitted_image_count[1]
                    == len(processed_image) - 1
                ):
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)

        messages = self._get_messages(item, splitted_image_counts)
        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        if (
            self.max_sample_length is not None
            and len(input_ids) > int(self.max_sample_length)
        ):
            # Filter over-length samples before DataLoader batching to keep
            # microbatch shapes stable for torch.compile in DDP.
            return None
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_messages(self, item, splitted_image_counts):
        messages = []
        for index, text in enumerate(item["texts"]):
            try:
                if (
                    item.get("relevance_ratings") is not None
                    and item["relevance_ratings"][index] is not None
                    and item["relevance_ratings"][index] < self.relevance_min_rating
                ):
                    continue
                if (
                    item.get("image_correspondence_ratings") is not None
                    and item["image_correspondence_ratings"][index] is not None
                    and item["image_correspondence_ratings"][index]
                    < self.image_correspondence_min_rating
                ):
                    continue
                if (
                    item.get("visual_dependency_ratings") is not None
                    and item["visual_dependency_ratings"][index] is not None
                    and item["visual_dependency_ratings"][index]
                    < self.visual_dependency_min_rating
                ):
                    continue
                if (
                    item.get("formatting_ratings") is not None
                    and item["formatting_ratings"][index] is not None
                    and item["formatting_ratings"][index]
                    < self.formatting_min_rating
                ):
                    continue
            except Exception as exc:
                logging.warning(
                    "Error processing ratings at index %s for sample: %s",
                    index,
                    exc,
                )

            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if len(messages) == 0:
            return messages

        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if len(splitted_image_counts) > 0:
            image_string = get_image_string(
                self.tokenizer, splitted_image_counts, self.mp_image_token_length
            )
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False, return_dict=True
            )
            seg_len = len(segment_ids["input_ids"])
            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)
            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"]),
        )


# ========================== Packing Dataset ==========================


class ConstantLengthDataset(IterableDataset):
    """Packs variable-length samples into fixed-length sequences using knapsack."""

    def __init__(
        self,
        dataset,
        seq_length: int = 1024,
        max_sample_length: int = 1024,
        max_images_per_example: int = 4,
        max_images_per_knapsack: int = 18,
        num_of_sequences: int = 1024,
        queue_size: int = 2,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_sample_length = max_sample_length
        self.seq_length = seq_length
        self.max_length = seq_length * num_of_sequences
        self.max_images_per_example = max_images_per_example
        self.max_images_per_knapsack = max_images_per_knapsack
        self.queue_size = max(queue_size, 1)
        self._sentinel = object()
        self._error_sentinel = object()
        self._average_length_per_sample = self.dataset.mp_image_token_length + 198

    def __len__(self):
        return int(
            len(self.dataset) * self._average_length_per_sample / self.seq_length
        )

    @staticmethod
    def _worker_context() -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def _make_base_iterator(self, worker_id: int, num_workers: int):
        return self.dataset.iter_for_worker(
            worker_id=worker_id,
            num_workers=num_workers,
        )

    def __iter__(self) -> Iterator[dict]:
        worker_id, num_workers = self._worker_context()

        queue: Queue = Queue(maxsize=self.queue_size)
        producer = threading.Thread(
            target=self._producer,
            args=(lambda: self._make_base_iterator(worker_id, num_workers), queue),
            daemon=True,
        )
        producer.start()

        while True:
            batch_of_batches = queue.get()
            if batch_of_batches is self._sentinel:
                break
            if isinstance(batch_of_batches, tuple) and batch_of_batches[0] is self._error_sentinel:
                raise RuntimeError("ConstantLengthDataset producer failed") from batch_of_batches[1]
            for batch in batch_of_batches:
                yield batch

    def _producer(self, make_iterator, queue: Queue):
        iterator = make_iterator()
        try:
            more_examples = True

            while more_examples:
                buffer, buffer_len = [], 0
                while buffer_len < self.max_length:
                    try:
                        sample = next(iterator)
                    except StopIteration:
                        more_examples = False
                        break

                    if sample is None:
                        continue
                    if len(sample["input_ids"]) >= self.max_sample_length:
                        continue
                    if len(sample["images"]) > self.max_images_per_example:
                        continue

                    sample["input_ids"] = torch.cat(
                        [sample["input_ids"], torch.tensor([self.dataset.tokenizer.pad_token_id])]
                    )
                    sample["attention_mask"] = torch.cat(
                        [sample["attention_mask"], torch.tensor([0])]
                    )
                    sample["labels"] = torch.cat([sample["labels"], torch.tensor([-100])])

                    buffer.append(sample)
                    buffer_len += len(sample["input_ids"])

                if not buffer:
                    break

                groups = self._balanced_greedy_knapsack(
                    buffer, self.seq_length, delta=5, max_images_per_knapsack=self.max_images_per_knapsack
                )

                packed_group = []
                for g in groups:
                    packed = self._pack_one_group(g, buffer, self.seq_length)
                    packed_group.append(
                        {
                            "input_ids": packed[0],
                            "labels": packed[1],
                            "attention_mask": packed[2],
                            "images": packed[3],
                        }
                    )

                if packed_group:
                    queue.put(packed_group)
        except Exception as exc:
            queue.put((self._error_sentinel, exc))
            return

        queue.put(self._sentinel)

    def _balanced_greedy_knapsack(self, buffer, L, delta=0, max_images_per_knapsack=None):
        lengths = [len(x["input_ids"]) for x in buffer]
        image_counts = [len(x["images"]) for x in buffer]

        items = sorted(
            enumerate(zip(lengths, image_counts)), key=lambda x: x[1][0], reverse=True
        )

        min_knapsacks = (sum(lengths) + L - 1) // L + delta
        knapsack_load = [0] * min_knapsacks
        knapsack_image_counts = [0] * min_knapsacks
        knapsack_groups = [[] for _ in range(min_knapsacks)]

        for idx, (item_len, item_image_count) in items:
            suitable_knapsack = None
            for ks_id in sorted(range(len(knapsack_load)), key=knapsack_load.__getitem__):
                length_fits = knapsack_load[ks_id] + item_len <= L
                image_fits = (
                    max_images_per_knapsack is None
                    or knapsack_image_counts[ks_id] + item_image_count <= max_images_per_knapsack
                )
                if length_fits and image_fits:
                    suitable_knapsack = ks_id
                    break

            if suitable_knapsack is None:
                suitable_knapsack = len(knapsack_load)
                knapsack_load.append(0)
                knapsack_image_counts.append(0)
                knapsack_groups.append([])

            knapsack_groups[suitable_knapsack].append(idx)
            knapsack_load[suitable_knapsack] += item_len
            knapsack_image_counts[suitable_knapsack] += item_image_count

        random.shuffle(knapsack_groups)
        return [g for g in knapsack_groups if g]

    def _pack_one_group(self, group_indices, batch, max_len):
        ids, lbl, am, ims = [], [], [], []
        for i in group_indices:
            ids.extend(batch[i]["input_ids"])
            lbl.extend(batch[i]["labels"])
            am.extend(batch[i]["attention_mask"])
            ims.extend(batch[i]["images"])

        if len(ids) > max_len:
            raise ValueError(f"Packed length {len(ids)} > max_len {max_len}")

        ids_t = torch.stack(ids)
        lbl_t = torch.stack(lbl)
        am_t = torch.stack(am)

        # Left-pad to max_len so all packed sequences have uniform length
        pad_len = max_len - len(ids_t)
        if pad_len > 0:
            pad_id = self.dataset.tokenizer.pad_token_id
            ids_t = torch.nn.functional.pad(ids_t, (pad_len, 0), value=pad_id)
            lbl_t = torch.nn.functional.pad(lbl_t, (pad_len, 0), value=-100)
            am_t = torch.nn.functional.pad(am_t, (pad_len, 0), value=0)

        return ids_t, lbl_t, am_t, ims


# ========================== Collator ==========================


class VQACollator:
    """Collates batches with left-padding and image stacking."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        batch_dict = {k: [item[k] for item in batch] for k in batch[0]}

        # Discard samples that are too long
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(
                batch_dict["input_ids"],
                batch_dict["labels"],
                batch_dict["attention_mask"],
                batch_dict["images"],
            )
            if len(ids) <= self.max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        ids_list, lbl_list, attn_list, img_list = zip(*filtered)
        max_len = self.max_length

        # Left-pad
        input_ids = [
            torch.nn.functional.pad(ids, (max_len - len(ids), 0), value=self.tokenizer.pad_token_id)
            for ids in ids_list
        ]
        labels = [
            torch.nn.functional.pad(lbl, (max_len - len(lbl), 0), value=-100)
            for lbl in lbl_list
        ]
        attention_mask = [
            torch.nn.functional.pad(attn, (max_len - len(attn), 0), value=0)
            for attn in attn_list
        ]

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "images": list(img_list),
            "labels": torch.stack(labels),
        }


# ========================== Wrapper Iterable Dataset ==========================


class NanoVLMIterableDataset(IterableDataset, Stateful):
    """Wraps VQADataset + optional ConstantLengthDataset into IterableDataset yielding torchtitan format."""

    def __init__(
        self,
        *,
        dataset_path: str,
        dataset_name: list[str],
        tokenizer,
        image_processor,
        mp_image_token_length: int,
        seq_len: int,
        use_packing: bool,
        streaming: bool = False,
        packing_num_sequences: int = 1024,
        max_sample_length: int,
        max_images_per_example: int,
        max_images_per_knapsack: int,
        relevance_min_rating: int,
        image_correspondence_min_rating: int,
        visual_dependency_min_rating: int,
        formatting_min_rating: int,
        image_token_id: int,
        dp_rank: int,
        dp_world_size: int,
    ):
        super().__init__()
        # Stabilize multi-process streaming reads on this environment.
        # The hf_transfer/xet fast paths can trigger teardown-time file descriptor
        # races in distributed streaming jobs.
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from datasets import (
            concatenate_datasets,
            get_dataset_config_names,
            interleave_datasets,
            load_dataset,
            load_from_disk,
        )
        from datasets.distributed import split_dataset_by_node

        self._streaming = streaming
        self._sample_idx = 0
        self._dp_rank = int(dp_rank)
        self._dp_world_size = int(dp_world_size)

        dataset_names_to_load = list(dataset_name) if dataset_name else ["default"]
        if "shards" in dataset_names_to_load:
            total_shards = 56
            dataset_names_to_load = [
                f"{dataset_path}/shard_{idx}" for idx in range(total_shards)
            ]
        if "_all_" in dataset_names_to_load:
            dataset_names_to_load = get_dataset_config_names(dataset_path)

        def _load_train_split(config_name: str):
            if "shard_" in config_name:
                return load_from_disk(config_name)
            hf_config_name = None if config_name in ("default", "", None) else config_name
            return load_dataset(
                dataset_path,
                name=hf_config_name,
                split="train",
                streaming=streaming,
                on_bad_files="warn",
            )

        loaded_datasets = []
        for config_name in dataset_names_to_load:
            try:
                ds = _load_train_split(config_name)
                if not streaming:
                    # Keep a lightweight sanity probe for map-style datasets only.
                    # Probing streaming datasets here can block multi-rank startup.
                    ds[0]
                loaded_datasets.append(ds)
            except Exception as exc:
                logger.warning(
                    "Failed to load dataset config '%s' from '%s': %s",
                    config_name,
                    dataset_path,
                    exc,
                )

        if not loaded_datasets:
            raise ValueError(
                "No valid train datasets were loaded. Check dataset path and config names."
            )

        if len(loaded_datasets) == 1:
            hf_dataset = loaded_datasets[0]
        elif streaming:
            hf_dataset = interleave_datasets(
                loaded_datasets,
                stopping_strategy="all_exhausted",
            )
        else:
            hf_dataset = concatenate_datasets(loaded_datasets)

        if not streaming:
            hf_dataset = hf_dataset.shuffle(seed=0)

        if dp_world_size > 1:
            if streaming:
                hf_dataset = split_dataset_by_node(
                    hf_dataset,
                    rank=dp_rank,
                    world_size=dp_world_size,
                )
            else:
                hf_dataset = hf_dataset.shard(
                    num_shards=dp_world_size,
                    index=dp_rank,
                    contiguous=False,
                )

        self._hf_data = hf_dataset if streaming else None

        self.vqa_dataset = VQADataset(
            hf_dataset,
            tokenizer,
            image_processor,
            mp_image_token_length,
            max_sample_length=max_sample_length,
            max_images_per_example=max_images_per_example,
            relevance_min_rating=relevance_min_rating,
            image_correspondence_min_rating=image_correspondence_min_rating,
            visual_dependency_min_rating=visual_dependency_min_rating,
            formatting_min_rating=formatting_min_rating,
        )
        self.use_packing = use_packing
        self.seq_len = seq_len
        self.image_token_id = image_token_id

        if use_packing:
            self.packed_dataset = ConstantLengthDataset(
                self.vqa_dataset,
                seq_length=seq_len,
                max_sample_length=max_sample_length,
                max_images_per_example=max_images_per_example,
                max_images_per_knapsack=max_images_per_knapsack,
                num_of_sequences=packing_num_sequences,
            )
        else:
            self.packed_dataset = None

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if self.use_packing:
            for sample in self.packed_dataset:
                self._sample_idx += 1
                # Yield raw sample dict; batch-level collation is handled by
                # DataLoader collate_fn for parity with nanoVLM_main.
                yield sample
        else:
            for sample in self.vqa_dataset.iter_for_worker(
                worker_id=worker_id,
                num_workers=num_workers,
            ):
                if sample is None:
                    continue
                self._sample_idx += 1
                yield sample

    def state_dict(self):
        if self._streaming and self._hf_data is not None:
            return {
                "data": self._hf_data.state_dict(),
                "sample_idx": self._sample_idx,
            }
        return {"sample_idx": self._sample_idx}

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict.get("sample_idx", 0)
        if self._streaming and self._hf_data is not None and "data" in state_dict:
            self._hf_data.load_state_dict(state_dict["data"])

# ========================== DataLoader ==========================


class NanoVLMDataLoader(ParallelAwareDataloader):
    """Torchtitan-compatible dataloader for nanoVLM."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_name: list[str] = field(default_factory=lambda: ["sample_1pct"])
        use_packing: bool = True
        max_images_per_example: int = 1
        max_images_per_knapsack: int = 8
        relevance_min_rating: int = 1
        image_correspondence_min_rating: int = 1
        visual_dependency_min_rating: int = 1
        formatting_min_rating: int = 1
        max_sample_length: int = 512
        vit_img_size: int = 256
        mp_pixel_shuffle_factor: int = 4
        mp_image_token_length: int = 16
        tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
        max_img_size: int = 1024
        resize_to_max_side_len: bool = False
        streaming: bool = False
        packing_num_sequences: int | None = None
        image_token_id: int = -1  # Set dynamically

    def __init__(
        self,
        config: Config,
        *,
        dp_rank: int,
        dp_world_size: int,
        tokenizer=None,
        seq_len: int = 512,
        local_batch_size: int = 4,
    ):
        # Build nanoVLM tokenizer (not the torchtitan one)
        nanovlm_tokenizer = get_tokenizer(
            config.tokenizer_name,
            extra_special_tokens=VLM_EXTRA_TOKENS,
            chat_template=LM_CHAT_TEMPLATE,
            model_max_length=config.max_sample_length,
        )

        image_token_id = int(nanovlm_tokenizer.image_token_id)
        config.image_token_id = image_token_id
        collator_max_len = seq_len if config.use_packing else config.max_sample_length
        self._vqa_collator = VQACollator(
            nanovlm_tokenizer,
            max_length=collator_max_len,
        )
        packing_num_sequences = (
            config.packing_num_sequences
            if config.packing_num_sequences is not None
            else local_batch_size * 4
        )

        image_processor = get_image_processor(
            max_img_size=config.max_img_size,
            splitted_image_size=config.vit_img_size,
            resize_to_max_side_len=config.resize_to_max_side_len,
        )

        dataset = NanoVLMIterableDataset(
            dataset_path=config.dataset_path or "patrickamadeus/the_cauldron",
            dataset_name=config.dataset_name,
            tokenizer=nanovlm_tokenizer,
            image_processor=image_processor,
            mp_image_token_length=config.mp_image_token_length,
            seq_len=seq_len,
            use_packing=config.use_packing,
            streaming=config.streaming,
            packing_num_sequences=packing_num_sequences,
            max_sample_length=config.max_sample_length,
            max_images_per_example=config.max_images_per_example,
            max_images_per_knapsack=config.max_images_per_knapsack,
            relevance_min_rating=config.relevance_min_rating,
            image_correspondence_min_rating=config.image_correspondence_min_rating,
            visual_dependency_min_rating=config.visual_dependency_min_rating,
            formatting_min_rating=config.formatting_min_rating,
            image_token_id=image_token_id,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

        super().__init__(
            dataset=dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(0),
        )

        self._image_token_id = image_token_id

    def _collate_fn(self, batch: list[dict[str, Any]]):
        """Mirror nanoVLM_main collation, then adapt to torchtitan tuple format."""
        collated = self._vqa_collator(batch)
        if len(collated["input_ids"]) == 0:
            empty = torch.zeros(0, 0, dtype=torch.long)
            return (
                {
                    "input": empty,
                    "images": torch.zeros(0, 3, 1, 1),
                    "attention_mask": empty,
                },
                empty,
            )

        images_tensor = self._flatten_images(collated["images"])
        return (
            {
                "input": collated["input_ids"],
                "images": images_tensor,
                "attention_mask": collated["attention_mask"],
            },
            collated["labels"],
        )

    @staticmethod
    def _flatten_images(images: list[Any]) -> torch.Tensor:
        """Flatten collated per-sample image containers into a single image tensor."""
        flat_images: list[torch.Tensor] = []
        for sample_images in images:
            if isinstance(sample_images, list):
                for image in sample_images:
                    if isinstance(image, torch.Tensor):
                        flat_images.append(image)
            elif isinstance(sample_images, torch.Tensor):
                flat_images.append(sample_images)

        if not flat_images:
            return torch.zeros(0, 3, 1, 1)
        if flat_images[0].ndim == 4:
            return torch.cat(flat_images, dim=0)
        return torch.stack(flat_images, dim=0)

    @property
    def image_token_id(self) -> int:
        return self._image_token_id
