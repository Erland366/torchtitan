# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP

_NANOVLM_CONFIG_PASSTHROUGH_FIELDS = (
    "vit_hidden_dim",
    "vit_inter_dim",
    "vit_patch_size",
    "vit_img_size",
    "vit_n_heads",
    "vit_n_blocks",
    "vit_ln_eps",
    "vit_cls_flag",
    "vit_dropout",
    "lm_hidden_dim",
    "lm_inter_dim",
    "lm_rms_eps",
    "lm_re_base",
    "lm_max_position_embeddings",
    "lm_vocab_size",
    "lm_n_heads",
    "lm_n_kv_heads",
    "lm_n_blocks",
    "lm_attn_scaling",
    "lm_tie_weights",
    "lm_dropout",
    "mp_pixel_shuffle_factor",
    "mp_image_token_length",
    "momh_enabled",
    "momh_kv_groups_vision",
    "momh_kv_groups_text",
    "momh_soft_gating",
    "momh_soft_gating_init",
    "momh_soft_gating_pairs",
    "momh_structural_mask_only",
    "momh_causal_gating",
)


def _config_as_dict(model_config) -> dict:
    if hasattr(model_config, "__dataclass_fields__"):
        return asdict(model_config)
    return dict(model_config)


def _nanovlm_model_config_to_vlm_config(model_config) -> dict:
    """Map TorchTitan nanoVLM config fields to nanoVLM_main VLMConfig fields."""
    cfg = _config_as_dict(model_config)

    vlm_cfg = {
        field: cfg[field]
        for field in _NANOVLM_CONFIG_PASSTHROUGH_FIELDS
        if field in cfg
    }

    if "lm_max_position_embeddings" in cfg:
        vlm_cfg["lm_max_length"] = cfg["lm_max_position_embeddings"]

    vlm_cfg.setdefault("lm_use_tokens", False)
    vlm_cfg.setdefault("lm_base_vocab_size", 49152)
    vlm_cfg.setdefault("extra_token_amount", cfg.get("lm_vocab_size", 49218) - 49152)
    return vlm_cfg


def _emit_nanovlm_bundle(output_dir: Path, model_config) -> None:
    """Emit nanoVLM_main-compatible files (config.json + model.safetensors)."""
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(_nanovlm_model_config_to_vlm_config(model_config), indent=4),
        encoding="utf-8",
    )

    single_file = output_dir / "model.safetensors"
    shard_files = sorted(output_dir.glob("model-*.safetensors"))
    if single_file.exists():
        return
    if len(shard_files) != 1:
        raise RuntimeError(
            "nanoVLM bundle emission expects a single consolidated shard. "
            f"Found {len(shard_files)} shard files under '{output_dir}'."
        )
    shutil.copyfile(shard_files[0], single_file)


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # pyrefly: ignore[bad-instantiation, not-callable]
    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    if model_name == "nanoVLM":
        _emit_nanovlm_bundle(Path(output_dir), model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
