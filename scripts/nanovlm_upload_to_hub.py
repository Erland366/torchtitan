#!/usr/bin/env python3
"""
Convert a torchtitan nanoVLM checkpoint to original nanoVLM format and
optionally upload to HuggingFace Hub.

Usage:
    # Local export only
    python scripts/nanovlm_upload_to_hub.py \
        --checkpoint-dir outputs/checkpoints/step-40 \
        --model-flavor small_debug_momh \
        --export-dtype bfloat16

    # Export + upload
    python scripts/nanovlm_upload_to_hub.py \
        --checkpoint-dir outputs/checkpoints/step-40 \
        --model-flavor small_debug_momh \
        --repo-id my-org/nanovlm-momh \
        --private
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Ensure the repo root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

MODEL_CARD_TEMPLATE = """\
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
  - torchtitan
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed \
for efficient training and experimentation. This checkpoint was trained using \
the [torchtitan](https://github.com/pytorch/torchtitan) port of nanoVLM.

For more information, check out:
- Original nanoVLM: https://github.com/huggingface/nanoVLM
- torchtitan: https://github.com/pytorch/torchtitan

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""


def model_config_to_vlm_config(model_config) -> dict:
    """Map NanoVLMModel.Config fields to original VLMConfig field names."""
    cfg = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else dict(model_config)

    vlm_cfg = {}
    # Direct pass-through fields (same name in both configs)
    pass_through = [
        "vit_hidden_dim", "vit_inter_dim", "vit_patch_size", "vit_img_size",
        "vit_n_heads", "vit_n_blocks", "vit_ln_eps", "vit_cls_flag", "vit_dropout",
        "lm_hidden_dim", "lm_inter_dim", "lm_rms_eps", "lm_re_base",
        "lm_max_position_embeddings", "lm_vocab_size", "lm_n_heads",
        "lm_n_kv_heads", "lm_n_blocks", "lm_attn_scaling", "lm_tie_weights",
        "lm_dropout", "mp_pixel_shuffle_factor", "mp_image_token_length",
        "momh_enabled", "momh_kv_groups_vision", "momh_kv_groups_text",
        "momh_soft_gating", "momh_soft_gating_init", "momh_soft_gating_pairs",
        "momh_structural_mask_only", "momh_causal_gating",
    ]
    for field in pass_through:
        if field in cfg:
            vlm_cfg[field] = cfg[field]

    # Renamed fields
    if "lm_max_position_embeddings" in cfg:
        vlm_cfg["lm_max_length"] = cfg["lm_max_position_embeddings"]

    # VLMConfig-specific defaults not present in torchtitan config
    vlm_cfg.setdefault("lm_use_tokens", False)
    vlm_cfg.setdefault("lm_base_vocab_size", 49152)
    vlm_cfg.setdefault("extra_token_amount", cfg.get("lm_vocab_size", 49218) - 49152)

    return vlm_cfg


def load_dcp_checkpoint(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load a torchtitan DCP checkpoint directory into a flat state dict."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    checkpoint_dir = str(Path(checkpoint_dir).resolve())

    # Check if this is already a safetensors export
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)

    # Check for consolidated .pt file (from dcp_to_torch_save)
    consolidated_path = os.path.join(checkpoint_dir, "consolidated.pt")
    if os.path.exists(consolidated_path):
        return torch.load(consolidated_path, map_location="cpu", weights_only=True)

    # DCP sharded checkpoint — convert to single-file format
    print(f"Converting DCP checkpoint at {checkpoint_dir} ...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "consolidated.pt")
        dcp_to_torch_save(checkpoint_dir, output_path)
        return torch.load(output_path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert torchtitan nanoVLM checkpoint to original nanoVLM format"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to torchtitan checkpoint directory (DCP or safetensors)",
    )
    parser.add_argument(
        "--model-flavor", required=True,
        help="Model flavor name from nanovlm_configs (e.g. small_debug_momh)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for converted model (default: <checkpoint-dir>/hf_export)",
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="HuggingFace Hub repo ID to upload to (e.g. my-org/nanovlm-momh)",
    )
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    parser.add_argument(
        "--export-dtype", default="bfloat16", choices=list(TORCH_DTYPE_MAP.keys()),
        help="Dtype for exported weights (default: bfloat16)",
    )
    args = parser.parse_args()

    # Get model config
    from torchtitan.models.nanoVLM import nanovlm_configs
    if args.model_flavor not in nanovlm_configs:
        print(f"Error: Unknown model flavor '{args.model_flavor}'. "
              f"Available: {list(nanovlm_configs.keys())}")
        sys.exit(1)
    model_config = nanovlm_configs[args.model_flavor]

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_dir} ...")
    state_dict = load_dcp_checkpoint(args.checkpoint_dir)
    print(f"  Loaded {len(state_dict)} keys")

    # Apply to_hf mapping
    from torchtitan.models.nanoVLM.state_dict_adapter import NanoVLMStateDictAdapter
    adapter = NanoVLMStateDictAdapter(model_config, hf_assets_path=None)
    hf_state_dict = adapter.to_hf(state_dict)
    print(f"  Mapped to {len(hf_state_dict)} HF keys")

    # Cast dtype
    export_dtype = TORCH_DTYPE_MAP[args.export_dtype]
    for key in hf_state_dict:
        if hf_state_dict[key].is_floating_point():
            hf_state_dict[key] = hf_state_dict[key].to(export_dtype)

    # Determine output directory
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "hf_export")
    os.makedirs(output_dir, exist_ok=True)

    # Save config.json
    vlm_config = model_config_to_vlm_config(model_config)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vlm_config, f, indent=4)
    print(f"  Saved config.json to {config_path}")

    # Save model.safetensors
    from safetensors.torch import save_file
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(hf_state_dict, safetensors_path)
    print(f"  Saved model.safetensors to {safetensors_path}")

    # Save README.md
    repo_id = args.repo_id or args.model_flavor
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))
    print(f"  Saved README.md to {readme_path}")

    # Upload to HuggingFace Hub if requested
    if args.repo_id:
        from huggingface_hub import create_repo, upload_folder

        print(f"Uploading to {args.repo_id} ...")
        repo_url = create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        upload_folder(
            repo_id=repo_url.repo_id,
            repo_type="model",
            folder_path=output_dir,
            commit_message="Upload nanoVLM (torchtitan port) checkpoint",
        )
        print(f"  Uploaded to https://huggingface.co/{repo_url.repo_id}")
    else:
        print(f"\nExport complete. To upload later:\n"
              f"  python {__file__} --checkpoint-dir {args.checkpoint_dir} "
              f"--model-flavor {args.model_flavor} --repo-id YOUR_REPO_ID")


if __name__ == "__main__":
    main()
