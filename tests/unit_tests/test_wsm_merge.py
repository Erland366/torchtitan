import pytest
import torch

from torchtitan.components.checkpoint_merge import (
    checkpoint_weights_for_merge_method,
    checkpoint_weights_from_gradient_coefficients,
    gradient_coefficients_for_merge_method,
    merge_model_state_dicts,
)


def test_checkpoint_weights_from_gradient_coefficients_rejects_non_monotone():
    with pytest.raises(ValueError, match="monotonically non-increasing"):
        checkpoint_weights_from_gradient_coefficients([0.5, 0.7])


def test_checkpoint_weights_from_gradient_coefficients_rejects_out_of_range():
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        checkpoint_weights_from_gradient_coefficients([0.8, -0.1])


def test_linear_merge_weights_match_uniform_mean():
    linear = checkpoint_weights_for_merge_method("linear", num_checkpoints=4)

    assert linear == pytest.approx([0.25, 0.25, 0.25, 0.25])


@pytest.mark.parametrize("method", ["cosine", "inv_sqrt"])
def test_decay_merge_weights_are_non_negative_and_normalized(method: str):
    weights = checkpoint_weights_for_merge_method(method, num_checkpoints=5)

    assert len(weights) == 5
    assert sum(weights) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in weights)


def test_mean_method_produces_uniform_checkpoint_weights():
    weights = checkpoint_weights_for_merge_method("mean", num_checkpoints=3)

    assert weights == pytest.approx([1.0 / 3.0] * 3)


def test_gradient_coefficients_for_linear_follow_expected_shape():
    coefficients = gradient_coefficients_for_merge_method(
        "linear",
        num_checkpoints=5,
    )

    assert coefficients == pytest.approx([0.8, 0.6, 0.4, 0.2])


def test_merge_model_state_dicts_merges_floating_tensors():
    merged = merge_model_state_dicts(
        [
            {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)},
            {"weight": torch.tensor([5.0, 6.0], dtype=torch.float32)},
        ],
        [0.25, 0.75],
    )

    torch.testing.assert_close(
        merged["weight"],
        torch.tensor([4.0, 5.0], dtype=torch.float32),
    )


def test_merge_model_state_dicts_preserves_equal_non_floating_tensors():
    merged = merge_model_state_dicts(
        [
            {"position_ids": torch.tensor([1, 2, 3], dtype=torch.int64)},
            {"position_ids": torch.tensor([1, 2, 3], dtype=torch.int64)},
        ],
        [0.4, 0.6],
    )

    assert torch.equal(
        merged["position_ids"],
        torch.tensor([1, 2, 3], dtype=torch.int64),
    )


def test_merge_model_state_dicts_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="mismatched keys"):
        merge_model_state_dicts(
            [
                {"a": torch.tensor([1.0])},
                {"b": torch.tensor([1.0])},
            ],
            [0.5, 0.5],
        )


def test_merge_model_state_dicts_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="mismatched shape"):
        merge_model_state_dicts(
            [
                {"weight": torch.ones(2)},
                {"weight": torch.ones(3)},
            ],
            [0.5, 0.5],
        )


def test_merge_model_state_dicts_rejects_non_floating_tensor_mismatch():
    with pytest.raises(ValueError, match="non-floating tensor"):
        merge_model_state_dicts(
            [
                {"mask": torch.tensor([1, 2], dtype=torch.int64)},
                {"mask": torch.tensor([1, 3], dtype=torch.int64)},
            ],
            [0.5, 0.5],
        )
