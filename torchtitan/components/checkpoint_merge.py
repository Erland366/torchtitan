# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

WSMMergeMethod = Literal["mean", "linear", "cosine", "inv_sqrt"]


def _validate_num_checkpoints(num_checkpoints: int) -> None:
    if num_checkpoints <= 0:
        raise ValueError(
            f"num_checkpoints must be positive, got {num_checkpoints}"
        )


def checkpoint_weights_from_gradient_coefficients(
    gradient_coefficients: Sequence[float],
) -> list[float]:
    """Convert monotone gradient decay coefficients into checkpoint weights.

    Implements the paper's mapping from per-update decay coefficients `w_i` to
    non-negative checkpoint merge weights `c_j`. If there are `k` gradient
    coefficients, the returned list contains `k + 1` checkpoint weights.
    """
    weights = [float(value) for value in gradient_coefficients]
    if not weights:
        return [1.0]

    for value in weights:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "gradient coefficients must stay within [0, 1], "
                f"got {value!r}"
            )

    for prev, curr in zip(weights, weights[1:]):
        if prev < curr:
            raise ValueError(
                "gradient coefficients must be monotonically non-increasing, "
                f"got {prev!r} then {curr!r}"
            )

    checkpoint_weights = [1.0 - weights[0]]
    checkpoint_weights.extend(prev - curr for prev, curr in zip(weights, weights[1:]))
    checkpoint_weights.append(weights[-1])
    return checkpoint_weights


def gradient_coefficients_for_merge_method(
    method: Literal["linear", "cosine", "inv_sqrt"],
    *,
    num_checkpoints: int,
) -> list[float]:
    """Generate paper-shaped gradient coefficients for a merge window."""
    _validate_num_checkpoints(num_checkpoints)
    if num_checkpoints == 1:
        return []

    if method == "linear":
        return [
            1.0 - (step / float(num_checkpoints))
            for step in range(1, num_checkpoints)
        ]
    if method == "cosine":
        return [
            0.5 * (1.0 + math.cos(math.pi * step / float(num_checkpoints)))
            for step in range(1, num_checkpoints)
        ]
    if method == "inv_sqrt":
        return [1.0 / math.sqrt(step + 1.0) for step in range(1, num_checkpoints)]

    raise ValueError(f"Unsupported merge method: {method!r}")


def checkpoint_weights_for_merge_method(
    method: WSMMergeMethod,
    *,
    num_checkpoints: int,
) -> list[float]:
    """Resolve checkpoint weights for a merge window."""
    _validate_num_checkpoints(num_checkpoints)
    if method == "mean":
        return [1.0 / float(num_checkpoints)] * num_checkpoints

    checkpoint_weights = checkpoint_weights_from_gradient_coefficients(
        gradient_coefficients_for_merge_method(
            method,
            num_checkpoints=num_checkpoints,
        )
    )
    if len(checkpoint_weights) != num_checkpoints:
        raise RuntimeError(
            "checkpoint-weight generation returned an unexpected length: "
            f"expected {num_checkpoints}, got {len(checkpoint_weights)}"
        )
    return checkpoint_weights


def merge_model_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    checkpoint_weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Merge model state dicts with explicit checkpoint weights.

    Floating-point tensors are averaged numerically. Non-floating tensors must
    match exactly across all checkpoints and are copied from the newest state
    dict after validation.
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    if len(state_dicts) != len(checkpoint_weights):
        raise ValueError(
            "checkpoint_weights must have the same length as state_dicts, "
            f"got {len(checkpoint_weights)} weights for {len(state_dicts)} state dicts"
        )

    weights = [float(weight) for weight in checkpoint_weights]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError(
            f"checkpoint_weights must sum to a positive value, got {weight_sum}"
        )
    normalized_weights = [weight / weight_sum for weight in weights]

    reference_keys = set(state_dicts[0].keys())
    for idx, state_dict in enumerate(state_dicts[1:], start=1):
        keys = set(state_dict.keys())
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"state_dict at index {idx} has mismatched keys; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    merged_state_dict: dict[str, torch.Tensor] = {}
    newest_state_dict = state_dicts[-1]

    for key in state_dicts[0]:
        tensors = [state_dict[key] for state_dict in state_dicts]
        reference = tensors[0]

        for idx, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != reference.shape:
                raise ValueError(
                    f"state_dict key {key!r} has mismatched shape at index {idx}: "
                    f"expected {tuple(reference.shape)}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != reference.dtype:
                raise ValueError(
                    f"state_dict key {key!r} has mismatched dtype at index {idx}: "
                    f"expected {reference.dtype}, got {tensor.dtype}"
                )

        if reference.is_floating_point():
            accumulation_dtype = (
                torch.float32
                if reference.dtype in (torch.float16, torch.bfloat16)
                else reference.dtype
            )

            merged = torch.zeros_like(reference, dtype=accumulation_dtype)
            for weight, tensor in zip(normalized_weights, tensors, strict=True):
                merged = merged + tensor.to(accumulation_dtype) * weight
            merged_state_dict[key] = merged.to(reference.dtype)
            continue

        newest = newest_state_dict[key]
        for idx, tensor in enumerate(tensors[:-1]):
            if not torch.equal(tensor, newest):
                raise ValueError(
                    f"non-floating tensor for key {key!r} differs between "
                    f"checkpoint {idx} and newest checkpoint"
                )
        merged_state_dict[key] = newest.clone()

    return merged_state_dict
