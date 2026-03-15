# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import re
from pathlib import Path

from torchtitan.components.checkpoint_merge import (
    WSMMergeMethod,
    checkpoint_weights_for_merge_method,
    merge_model_state_dicts,
)

from convert_to_hf import (
    build_model_wrapper_and_adapter,
    load_tt_model_state_dict,
    save_hf_state_dict,
)

_STEP_DIR_PATTERN = re.compile(r"^step-(\d+)$")


def _sorted_step_dirs(checkpoint_dir: Path) -> list[Path]:
    step_dirs: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir():
            continue
        match = _STEP_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        step_dirs.append((int(match.group(1)), child))
    return [path for _, path in sorted(step_dirs)]


def _resolve_input_dirs(
    *,
    checkpoint_dir: Path | None,
    input_dirs: list[Path],
    last_n: int | None,
) -> list[Path]:
    if checkpoint_dir is not None:
        if input_dirs:
            raise ValueError(
                "Provide either --checkpoint_dir or explicit --input_dir values, not both"
            )
        step_dirs = _sorted_step_dirs(checkpoint_dir)
        if not step_dirs:
            raise ValueError(f"No step-* checkpoint directories found in {checkpoint_dir}")
        if last_n is not None:
            if last_n <= 0:
                raise ValueError(f"--last_n must be positive, got {last_n}")
            step_dirs = step_dirs[-last_n:]
        return step_dirs

    if not input_dirs:
        raise ValueError("Provide at least one --input_dir or set --checkpoint_dir")
    return input_dirs


def _write_merge_metadata(
    *,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    merge_method: WSMMergeMethod,
    checkpoint_paths: list[Path],
    checkpoint_weights: list[float],
) -> None:
    metadata = {
        "model_name": model_name,
        "model_flavor": model_flavor,
        "merge_method": merge_method,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_weights": checkpoint_weights,
    }
    (output_dir / "wsm_merge_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def merge_to_hf(
    *,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: Path,
    export_dtype: str,
    merge_method: WSMMergeMethod,
    checkpoint_dir: Path | None,
    input_dirs: list[Path],
    last_n: int | None,
) -> None:
    checkpoint_paths = _resolve_input_dirs(
        checkpoint_dir=checkpoint_dir,
        input_dirs=input_dirs,
        last_n=last_n,
    )
    checkpoint_weights = checkpoint_weights_for_merge_method(
        merge_method,
        num_checkpoints=len(checkpoint_paths),
    )

    model, model_config, sd_adapter = build_model_wrapper_and_adapter(
        model_name=model_name,
        model_flavor=model_flavor,
        hf_assets_path=hf_assets_path,
    )

    state_dicts = [
        load_tt_model_state_dict(path, model)
        for path in checkpoint_paths
    ]
    merged_state_dict = merge_model_state_dicts(state_dicts, checkpoint_weights)
    hf_state_dict = sd_adapter.to_hf(merged_state_dict)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"Output directory {output_dir} already exists and is not empty"
        )

    save_hf_state_dict(
        output_dir=output_dir,
        hf_state_dict=hf_state_dict,
        sd_adapter=sd_adapter,
        model_name=model_name,
        model_config=model_config,
        export_dtype=export_dtype,
    )
    _write_merge_metadata(
        output_dir=output_dir,
        model_name=model_name,
        model_flavor=model_flavor,
        merge_method=merge_method,
        checkpoint_paths=checkpoint_paths,
        checkpoint_weights=checkpoint_weights,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge TorchTitan DCP checkpoints into one HF-format checkpoint."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the merged HF checkpoint.",
    )
    parser.add_argument(
        "--input_dir",
        action="append",
        default=[],
        type=Path,
        help="Explicit checkpoint directory to merge. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Parent directory containing step-* checkpoint folders.",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=None,
        help="If using --checkpoint_dir, merge only the latest N step-* folders.",
    )
    parser.add_argument("--model_name", type=str, default="llama3")
    parser.add_argument("--model_flavor", type=str, default="8B")
    parser.add_argument(
        "--merge_method",
        type=str,
        choices=["mean", "linear", "cosine", "inv_sqrt"],
        default="mean",
        help="WSM merge weighting scheme to apply across the selected checkpoints.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory for index mapping and adapters.",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for the merged HF checkpoint.",
    )
    args = parser.parse_args()

    merge_to_hf(
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        hf_assets_path=args.hf_assets_path,
        export_dtype=args.export_dtype,
        merge_method=args.merge_method,
        checkpoint_dir=args.checkpoint_dir,
        input_dirs=args.input_dir,
        last_n=args.last_n,
    )
