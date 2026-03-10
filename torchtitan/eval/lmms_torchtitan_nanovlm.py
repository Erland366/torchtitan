"""Standalone lmms-eval model class for TorchTitan nanoVLM fallback eval."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from tqdm import tqdm

from torchtitan.eval.nanovlm_cached_runtime import NanoVLMCachedGenerator
from torchtitan.models.nanoVLM.dataloader import (
    LM_CHAT_TEMPLATE,
    VLM_EXTRA_TOKENS,
    get_image_processor,
    get_image_string,
    get_tokenizer,
)
from torchtitan.models.nanoVLM.model import NanoVLMModel
from torchtitan.models.nanoVLM.state_dict_adapter import NanoVLMStateDictAdapter

_DEFAULT_TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
_ALLOWED_MISSING_EXACT = {"rotary_embd.inv_freq"}
_ALLOWED_MISSING_PREFIXES = ("vision_encoder.layers.",)


def _load_checkpoint_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config.json in checkpoint folder: {model_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_model_config(raw_config: dict[str, Any]) -> NanoVLMModel.Config:
    field_names = {field.name for field in dataclasses.fields(NanoVLMModel.Config)}
    config_kwargs = {
        key: value for key, value in raw_config.items() if key in field_names
    }

    if (
        "lm_max_position_embeddings" not in config_kwargs
        and "lm_max_length" in raw_config
    ):
        config_kwargs["lm_max_position_embeddings"] = int(raw_config["lm_max_length"])

    if "lm_vocab_size" not in config_kwargs:
        lm_base_vocab_size = raw_config.get("lm_base_vocab_size")
        extra_token_amount = raw_config.get("extra_token_amount")
        if lm_base_vocab_size is None or extra_token_amount is None:
            raise ValueError(
                "Checkpoint config must define lm_vocab_size or "
                "(lm_base_vocab_size + extra_token_amount)."
            )
        config_kwargs["lm_vocab_size"] = int(lm_base_vocab_size) + int(
            extra_token_amount
        )

    return NanoVLMModel.Config(**config_kwargs)


def _flatten_visuals(visual_batches: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for visual_batch in visual_batches:
        if visual_batch is None:
            flattened.append(None)
            continue
        flattened.extend(visual_batch)
    return flattened


def _is_allowed_missing_key(key: str) -> bool:
    if key in _ALLOWED_MISSING_EXACT:
        return True
    return any(key.startswith(prefix) for prefix in _ALLOWED_MISSING_PREFIXES)


def _coerce_visual_to_image(visual: Any) -> Image.Image:
    if isinstance(visual, Image.Image):
        return visual.convert("RGB")
    if isinstance(visual, str):
        return Image.open(visual).convert("RGB")
    if isinstance(visual, np.ndarray):
        return Image.fromarray(visual).convert("RGB")
    raise ValueError(
        f"Unsupported visual input type: {type(visual)}. "
        "Expected PIL Image, image path, or numpy array."
    )


def _resolve_generation_params(
    generation_kwargs: dict[str, Any],
) -> tuple[int, float, float, bool]:
    max_new_tokens = int(generation_kwargs.get("max_new_tokens", 50))
    temperature = float(generation_kwargs.get("temperature", 0.0))
    top_p = float(generation_kwargs.get("top_p", 1.0))
    greedy = generation_kwargs.get("do_sample", False) is False or temperature == 0.0
    return max_new_tokens, temperature, top_p, greedy


def _benchmark_formatting(task_name: str) -> dict[str, Any]:
    benchmark_formats: dict[Any, dict[str, str]] = {
        ("ai2d", "mmstar", "seedbench", "scienceqa"): {
            "text_replacements": {
                "\nOptions:": "\nChoices:",
                "\nA. ": "\nChoices:\nA. ",
                "Please select the correct answer from the options above.": "Answer with the letter.",
                "Answer with the option's letter from the given choices directly": "Answer with the letter directly",
            },
            "assistant_prefix": "Answer:",
            "user_prefix": "",
            "user_suffix": "",
        },
        ("docvqa_val", "docvqa_test"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": (
                "Give a short and terse answer to the following question. "
                "Do not paraphrase or reformat the text you see in the image. "
                "Do not include any full stops. Just give the answer without "
                "additional explanation. Question: "
            ),
            "user_suffix": "",
        },
        "chartvqa": {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": (
                "For the question below, follow the following instructions:\n"
                "-The answer should contain as few words as possible.\n"
                "-Don't paraphrase or reformat the text you see in the image.\n"
                "-Answer a binary question with Yes or No.\n"
                "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                "-Don't include any units in the answer.\n"
                "-Do not include any full stops at the end of the answer.\n"
                "-Try to include the full label from the graph when asked about an entity.\n"
                "Question: "
            ),
            "user_suffix": "",
        },
        ("textvqa_val", "textvqa_test"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": (
                "Answer the following question about the image using as few words as possible. "
                "Follow these additional instructions:\n"
                "-Always answer a binary question with Yes or No.\n"
                "-When asked what time it is, reply with the time seen in the image.\n"
                "-Do not put any full stops at the end of the answer.\n"
                "-Do not put quotation marks around the answer.\n"
                "-An answer with one or two words is favorable.\n"
                "-Do not apply common sense knowledge. The answer can be found in the image.\n"
                "Question: "
            ),
            "user_suffix": "",
        },
        ("mmmu_val", "mmmu_test"): {
            "text_replacements": {
                "Question:": "",
                "Answer with the option's letter from the given choices directly.": "Answer with the letter directly.",
                "\nA. ": "\nChoices:\nA. ",
            },
            "assistant_prefix": "Answer:",
            "user_prefix": "",
            "user_suffix": "",
        },
        ("infovqa_val", "mme", "ocrbench"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": "",
            "user_suffix": "\nGive a very brief answer.",
        },
    }

    if task_name in benchmark_formats:
        return benchmark_formats[task_name]
    for key, formatting in benchmark_formats.items():
        if isinstance(key, tuple) and task_name in key:
            return formatting
    return {
        "text_replacements": {},
        "assistant_prefix": "",
        "user_prefix": "",
        "user_suffix": "",
    }


def _apply_benchmark_formatting(
    *, context_str: str, prompt: str, task_name: str
) -> tuple[str, str]:
    formatting = _benchmark_formatting(task_name)
    user_prefix = formatting["user_prefix"]
    if user_prefix:
        context_str = user_prefix + context_str

    for old_text, new_text in formatting["text_replacements"].items():
        context_str = context_str.replace(old_text, new_text)

    user_suffix = formatting["user_suffix"]
    if user_suffix:
        context_str = context_str + user_suffix

    assistant_prefix = formatting["assistant_prefix"]
    if assistant_prefix:
        prompt = prompt + assistant_prefix

    return context_str, prompt


class TorchTitanNanoVLM(lmms):
    """Standalone TorchTitan nanoVLM model backend for lmms-eval."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int | None = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint path not found: {model_path}")

        raw_config = _load_checkpoint_config(model_path)
        model_config = _build_model_config(raw_config)

        tokenizer_name = str(raw_config.get("lm_tokenizer", _DEFAULT_TOKENIZER_NAME))
        chat_template = str(raw_config.get("lm_chat_template", LM_CHAT_TEMPLATE))
        self.tokenizer = get_tokenizer(
            tokenizer_name,
            extra_special_tokens=VLM_EXTRA_TOKENS,
            chat_template=chat_template,
        )
        # The cache in get_tokenizer is keyed by name. Re-assert template so
        # per-checkpoint templates remain explicit.
        self.tokenizer.chat_template = chat_template
        image_token_id = int(self.tokenizer.image_token_id)
        model_config.image_token_id = image_token_id

        self.model = NanoVLMModel(model_config).to(device)
        self._load_weights_from_checkpoint(model_path)
        self.model.eval()
        self.cached_generator = NanoVLMCachedGenerator(self.model, self.tokenizer)

        self.device = device
        self.batch_size = int(batch_size)
        self._max_length = (
            int(max_length)
            if max_length is not None
            else int(model_config.lm_max_position_embeddings)
        )

        max_img_size = int(raw_config.get("max_img_size", model_config.vit_img_size))
        resize_to_max_side_len = bool(raw_config.get("resize_to_max_side_len", False))
        self.image_processor = get_image_processor(
            max_img_size,
            model_config.vit_img_size,
            resize_to_max_side_len,
        )

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

    def _load_weights_from_checkpoint(self, model_path: Path) -> None:
        adapter = NanoVLMStateDictAdapter(self.model.config, hf_assets_path=None)

        native_state_dict = self.model.state_dict()
        hf_state_dict = adapter.to_hf(native_state_dict)
        hf_state_dict = adapter.adapt_hf_state_dict_for_checkpoint(
            hf_state_dict,
            str(model_path),
        )
        storage_reader = adapter.get_hf_storage_reader(str(model_path))
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        mapped_native_state_dict = adapter.from_hf(hf_state_dict)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            mapped_native_state_dict,
            strict=False,
        )
        disallowed_missing = [
            key
            for key in missing_keys
            if not _is_allowed_missing_key(key)
        ]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint load mismatch for TorchTitanNanoVLM. "
                f"missing={disallowed_missing}, unexpected={unexpected_keys}"
            )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size_per_gpu(self) -> int:
        return self.batch_size

    def _prepare_visual_input(
        self, visual_list: list[Any]
    ) -> tuple[list[torch.Tensor] | None, list[tuple[int, int]] | None]:
        if not visual_list or visual_list[0] is None:
            return None, None

        processed_image_batches: list[torch.Tensor] = []
        split_ratios: list[tuple[int, int]] = []

        for visual in visual_list:
            image = _coerce_visual_to_image(visual)
            image_tiles, split_ratio = self.image_processor(image)
            if (
                not hasattr(self.tokenizer, "global_image_token")
                and split_ratio[0] * split_ratio[1] == len(image_tiles) - 1
            ):
                image_tiles = image_tiles[1:]
            processed_image_batches.append(image_tiles)
            split_ratios.append(split_ratio)

        return processed_image_batches, split_ratios

    def _model_dtype(self) -> torch.dtype:
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    @torch.inference_mode()
    def _generate_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor | None,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        return self.cached_generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
        )

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for TorchTitanNanoVLM.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        def _collate(request_args):
            token_count = len(self.tokenizer.encode(request_args[0]))
            return -token_count, request_args[0]

        progress = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        reordered = utils.Collator(
            [request.args for request in requests],
            _collate,
            grouping=True,
        )
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)
        responses: list[str] = []

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visuals = [
                mapper(self.task_dict[t][s][index])
                for mapper, index, t, s in zip(doc_to_visual, doc_id, task, split)
            ]
            flat_visuals = _flatten_visuals(list(visuals))
            processed_images, split_ratios = self._prepare_visual_input(flat_visuals)

            prompts_messages = []
            split_idx = 0
            for item_idx, context in enumerate(contexts):
                formatted_context, _ = _apply_benchmark_formatting(
                    context_str=context,
                    prompt="",
                    task_name=task[item_idx],
                )

                image_count = 0 if visuals[item_idx] is None else len(visuals[item_idx])
                image_string = ""
                for _ in range(image_count):
                    assert split_ratios is not None
                    image_string += get_image_string(
                        self.tokenizer,
                        [split_ratios[split_idx]],
                        self.model.config.mp_image_token_length,
                    )
                    split_idx += 1

                prompts_messages.append(
                    [{"role": "user", "content": image_string + formatted_context}]
                )

            prompts = self.tokenizer.apply_chat_template(
                prompts_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            for item_idx, prompt in enumerate(prompts):
                _, prompts[item_idx] = _apply_benchmark_formatting(
                    context_str="",
                    prompt=prompt,
                    task_name=task[item_idx],
                )

            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                padding_side="left",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            image_tensor = None
            if processed_images:
                image_tensor = torch.cat(processed_images, dim=0).to(
                    device=self.device,
                    dtype=self._model_dtype(),
                )

            generation_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens, temperature, top_p, greedy = _resolve_generation_params(
                generation_kwargs
            )

            generated_token_ids = self._generate_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
            )
            generated_texts = self.tokenizer.batch_decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            responses.extend(generated_texts)
            progress.update(len(contexts))

        progress.close()
        return reordered.get_original(responses)

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        del requests
        raise NotImplementedError(
            "Multi-round generation is not implemented for TorchTitanNanoVLM."
        )
