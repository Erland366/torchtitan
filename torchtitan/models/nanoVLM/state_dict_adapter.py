"""
State dict adapter for nanoVLM: load pretrained SigLIP2 + SmolLM2 weights,
and export torchtitan checkpoints back to original nanoVLM format.
"""

import json
import os
import re
from typing import Any

import torch

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger

from torch.distributed.checkpoint import HuggingFaceStorageReader


class NanoVLMStateDictAdapter(BaseStateDictAdapter):
    """Load pretrained SigLIP2 (vision) + SmolLM2 (decoder) weights."""

    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.fqn_to_index_mapping = None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan nanoVLM state dict to original nanoVLM format.

        Key mapping (torchtitan → original nanoVLM):
            tok_embeddings.*          → decoder.token_embedding.*
            layers.{i}.*             → decoder.blocks.{i}.*
            norm.*                   → decoder.norm.*
            output.*                 → decoder.head.*
            projector.*              → MP.*
            vision_encoder.blocks.*  → vision_encoder.blocks.* (unchanged)
            vision_encoder.layer_norm/patch_embedding → unchanged

        Skipped keys:
            vision_encoder.layers.*  — alias of vision_encoder.blocks (duplicates)
            rotary_embd.inv_freq     — recomputed on load
        """
        hf_sd: dict[str, Any] = {}

        for raw_key, tensor in state_dict.items():
            # Canonicalize compiled wrapper keys so eager/compiled models map
            # identically when exporting to HF format.
            key = raw_key.replace("._orig_mod.", ".")

            # Skip the nn.ModuleDict alias (layers == blocks for vision encoder)
            if key.startswith("vision_encoder.layers."):
                continue
            # Skip rotary embedding buffer (recomputed on load)
            if key == "rotary_embd.inv_freq":
                continue
            # Skip MoMH soft-gating params (not present in vanilla checkpoints).
            if ".momh_gate" in key:
                continue

            # --- Decoder mappings ---
            tie_weights = getattr(self.model_config, "lm_tie_weights", True)
            if key == "tok_embeddings.weight":
                if tie_weights:
                    # Tied weights: checkpoint only has decoder.head.weight,
                    # so emit only that key to match what dcp.load() will find.
                    hf_sd["decoder.head.weight"] = tensor
                else:
                    hf_sd["decoder.token_embedding.weight"] = tensor
                continue
            if key == "output.weight":
                # If weights are tied, tok_embeddings already handled this;
                # if untied, emit it here.
                if "decoder.head.weight" not in hf_sd:
                    hf_sd["decoder.head.weight"] = tensor
                continue
            if key.startswith("layers."):
                hf_key = "decoder.blocks." + key[len("layers."):]
                hf_sd[hf_key] = tensor
                continue
            if key.startswith("norm."):
                hf_key = "decoder.norm." + key[len("norm."):]
                hf_sd[hf_key] = tensor
                continue

            # --- Projector mapping ---
            if key.startswith("projector."):
                hf_key = "MP." + key[len("projector."):]
                hf_sd[hf_key] = tensor
                continue

            # --- Vision encoder (keys pass through unchanged) ---
            if key.startswith("vision_encoder."):
                hf_sd[key] = tensor
                continue

            logger.warning(f"to_hf: unmapped key '{key}', passing through")
            hf_sd[key] = tensor

        return hf_sd

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an HF-keyed state dict to torchtitan nanoVLM format.

        Auto-detects two source formats:
          1. Original nanoVLM (decoder.blocks.*, MP.*, vision_encoder.*)
             — from trained nanoVLM checkpoints (e.g. lusxvr/nanoVLM-230M-8k)
          2. SigLIP2 + SmolLM2 (vision_model.*, model.layers.*)
             — from raw pretrained backbone weights
        """
        # Detect format: original nanoVLM uses "decoder." or "MP." prefixes
        has_nanovlm_keys = any(
            k.startswith(("decoder.", "MP.")) for k in hf_state_dict
        )
        if has_nanovlm_keys:
            return self._from_nanovlm_hf(hf_state_dict)
        return self._from_backbone_hf(hf_state_dict)

    def _from_nanovlm_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Reverse of to_hf(): original nanoVLM keys → torchtitan keys.

        Key mapping (original nanoVLM → torchtitan):
            decoder.token_embedding.*  → tok_embeddings.*
            decoder.head.*             → output.*
            decoder.blocks.{i}.*       → layers.{i}.*
            decoder.norm.*             → norm.*
            MP.*                       → projector.*
            vision_encoder.*           → vision_encoder.* (unchanged)
        """
        native_sd: dict[str, Any] = {}
        cfg = self.model_config

        for raw_key, tensor in hf_state_dict.items():
            # Checkpoints saved from compiled modules include `._orig_mod.` in keys.
            # Strip it so they map to the eager module names used by torchtitan.
            key = raw_key.replace("._orig_mod.", ".")

            # Skip rotary embedding buffer (recomputed on load)
            if "rotary_embd" in key:
                continue

            # --- Decoder mappings ---
            if key == "decoder.token_embedding.weight":
                native_sd["tok_embeddings.weight"] = tensor
                if cfg.lm_tie_weights:
                    native_sd["output.weight"] = tensor
                continue
            if key == "decoder.head.weight":
                native_sd["output.weight"] = tensor
                if cfg.lm_tie_weights and "tok_embeddings.weight" not in native_sd:
                    native_sd["tok_embeddings.weight"] = tensor
                continue
            if key.startswith("decoder.blocks."):
                native_key = "layers." + key[len("decoder.blocks."):]
                native_sd[native_key] = tensor
                continue
            if key.startswith("decoder.norm."):
                native_key = "norm." + key[len("decoder.norm."):]
                native_sd[native_key] = tensor
                continue

            # --- Projector mapping ---
            if key.startswith("MP."):
                native_key = "projector." + key[len("MP."):]
                native_sd[native_key] = tensor
                continue

            # --- Vision encoder (pass through unchanged) ---
            if key.startswith("vision_encoder."):
                native_sd[key] = tensor
                continue

            logger.warning(f"_from_nanovlm_hf: unmapped key '{key}', skipping")

        return native_sd

    def _from_backbone_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """SigLIP2 + SmolLM2 backbone keys → torchtitan nanoVLM keys."""
        native_sd = {}
        cfg = self.model_config

        # ---- Vision encoder (SigLIP2) ----
        vit_mapping = {
            "vision_model.embeddings.patch_embedding.weight": "vision_encoder.patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "vision_encoder.patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "vision_encoder.patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "vision_encoder.layer_norm.weight",
            "vision_model.post_layernorm.bias": "vision_encoder.layer_norm.bias",
        }

        for i in range(cfg.vit_n_blocks):
            vit_mapping.update(
                {
                    f"vision_model.encoder.layers.{i}.layer_norm1.weight": f"vision_encoder.blocks.{i}.ln1.weight",
                    f"vision_model.encoder.layers.{i}.layer_norm1.bias": f"vision_encoder.blocks.{i}.ln1.bias",
                    f"vision_model.encoder.layers.{i}.layer_norm2.weight": f"vision_encoder.blocks.{i}.ln2.weight",
                    f"vision_model.encoder.layers.{i}.layer_norm2.bias": f"vision_encoder.blocks.{i}.ln2.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.weight": f"vision_encoder.blocks.{i}.mlp.fc1.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.bias": f"vision_encoder.blocks.{i}.mlp.fc1.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc2.weight": f"vision_encoder.blocks.{i}.mlp.fc2.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc2.bias": f"vision_encoder.blocks.{i}.mlp.fc2.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight": f"vision_encoder.blocks.{i}.attn.out_proj.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias": f"vision_encoder.blocks.{i}.attn.out_proj.bias",
                }
            )

        for hf_key, native_key in vit_mapping.items():
            if hf_key in hf_state_dict:
                tensor = hf_state_dict[hf_key]
                if "position_embedding" in hf_key and tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                native_sd[native_key] = tensor

        # Handle QKV concatenation for vision encoder
        for i in range(cfg.vit_n_blocks):
            q_key = f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
            k_key = f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
            v_key = f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
            if q_key in hf_state_dict and k_key in hf_state_dict and v_key in hf_state_dict:
                qkv_weight = torch.cat(
                    [hf_state_dict[q_key], hf_state_dict[k_key], hf_state_dict[v_key]],
                    dim=0,
                )
                native_sd[f"vision_encoder.blocks.{i}.attn.qkv_proj.weight"] = qkv_weight

            q_bias_key = f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
            k_bias_key = f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
            v_bias_key = f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
            if (
                q_bias_key in hf_state_dict
                and k_bias_key in hf_state_dict
                and v_bias_key in hf_state_dict
            ):
                qkv_bias = torch.cat(
                    [
                        hf_state_dict[q_bias_key],
                        hf_state_dict[k_bias_key],
                        hf_state_dict[v_bias_key],
                    ],
                    dim=0,
                )
                native_sd[f"vision_encoder.blocks.{i}.attn.qkv_proj.bias"] = qkv_bias

        # ---- Language model (SmolLM2) ----
        lm_mapping = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
        }

        for i in range(cfg.lm_n_blocks):
            lp = f"model.layers.{i}."
            bp = f"layers.{i}."
            lm_mapping.update(
                {
                    f"{lp}self_attn.q_proj.weight": f"{bp}attn.q_proj.weight",
                    f"{lp}self_attn.k_proj.weight": f"{bp}attn.k_proj.weight",
                    f"{lp}self_attn.v_proj.weight": f"{bp}attn.v_proj.weight",
                    f"{lp}self_attn.o_proj.weight": f"{bp}attn.out_proj.weight",
                    f"{lp}mlp.gate_proj.weight": f"{bp}mlp.gate_proj.weight",
                    f"{lp}mlp.up_proj.weight": f"{bp}mlp.up_proj.weight",
                    f"{lp}mlp.down_proj.weight": f"{bp}mlp.down_proj.weight",
                    f"{lp}input_layernorm.weight": f"{bp}norm1.weight",
                    f"{lp}post_attention_layernorm.weight": f"{bp}norm2.weight",
                }
            )

        for hf_key, native_key in lm_mapping.items():
            if hf_key in hf_state_dict:
                tensor = hf_state_dict[hf_key]

                # Handle vocab size mismatch for embeddings
                if hf_key == "model.embed_tokens.weight":
                    original_vocab = tensor.shape[0]
                    target_vocab = cfg.lm_vocab_size
                    if original_vocab < target_vocab:
                        logger.info(
                            f"Extending token embeddings from {original_vocab} to {target_vocab}"
                        )
                        extra = torch.empty(
                            target_vocab - original_vocab,
                            tensor.shape[1],
                            dtype=tensor.dtype,
                        )
                        torch.nn.init.normal_(extra, mean=0.0, std=0.02)
                        tensor = torch.cat([tensor, extra], dim=0)
                        # Also set output head weight (tied or not)
                        native_sd["output.weight"] = tensor.clone()

                native_sd[native_key] = tensor

        # Handle lm_head if separate
        if "lm_head.weight" in hf_state_dict and not cfg.lm_tie_weights:
            lm_head = hf_state_dict["lm_head.weight"]
            original_vocab = lm_head.shape[0]
            target_vocab = cfg.lm_vocab_size
            if original_vocab < target_vocab:
                extra = torch.empty(
                    target_vocab - original_vocab,
                    lm_head.shape[1],
                    dtype=lm_head.dtype,
                )
                torch.nn.init.normal_(extra, mean=0.0, std=0.02)
                lm_head = torch.cat([lm_head, extra], dim=0)
            native_sd["output.weight"] = lm_head

        return native_sd

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        return HuggingFaceStorageReader(path)

    def _get_hf_checkpoint_keys(self, checkpoint_id: str) -> set[str]:
        """Return tensor keys available in an HF safetensors checkpoint folder."""
        index_path = os.path.join(checkpoint_id, "model.safetensors.index.json")
        single_file_path = os.path.join(checkpoint_id, "model.safetensors")

        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if isinstance(weight_map, dict):
                return set(weight_map.keys())

        if os.path.isfile(single_file_path):
            from safetensors import safe_open

            with safe_open(single_file_path, framework="pt", device="cpu") as f:
                return set(f.keys())

        return set()

    def adapt_hf_state_dict_for_checkpoint(
        self, hf_state_dict: dict[str, Any], checkpoint_id: str
    ) -> dict[str, Any]:
        """Adapt expected HF keys to checkpoint contents.

        Some older checkpoints store compiled decoder keys
        (`decoder.blocks.{i}._orig_mod.*`). When this is detected, remap
        expected keys so `dcp.load` can populate tensors correctly.
        """
        available_keys = self._get_hf_checkpoint_keys(checkpoint_id)
        if not available_keys:
            return hf_state_dict

        filtered: dict[str, Any] = {}
        remapped = 0
        dropped = 0

        for key, tensor in hf_state_dict.items():
            if key in available_keys:
                filtered[key] = tensor
                continue

            match = re.match(r"^(decoder\.blocks\.\d+\.)(.+)$", key)
            if match:
                compiled_key = f"{match.group(1)}_orig_mod.{match.group(2)}"
                if compiled_key in available_keys:
                    filtered[compiled_key] = tensor
                    remapped += 1
                    continue

            dropped += 1

        if remapped or dropped:
            logger.info(
                "HF key adaptation: remapped=%d, dropped=%d, kept=%d",
                remapped,
                dropped,
                len(filtered),
            )

        return filtered
