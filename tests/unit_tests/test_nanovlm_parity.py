import pytest

from torchtitan.tools.nanovlm_parity import (
    paired_loss_diff,
    parse_nanovlm_log,
    parse_torchtitan_optimizer_groups,
    parse_torchtitan_log,
)


def test_parse_nanovlm_log_extracts_loss_and_tps():
    text = """
[INFO] [TRAIN] step=1 batch_loss=0.1234 step_loss=0.9000 tokens_per_second=10000.0
[INFO] [TRAIN] step=2 batch_loss=0.1200 step_loss=0.8000 tokens_per_second=11000.5
wandb: 🚀 View run at https://wandb.ai/team/project/runs/abc12345
"""
    parsed = parse_nanovlm_log(text)
    assert parsed.losses.values_by_step == {1: 0.9, 2: 0.8}
    assert parsed.throughput.values_by_step == {1: 10000.0, 2: 11000.5}
    assert parsed.wandb_run_id == "abc12345"
    assert parsed.throughput.median_excluding_first_step() == 11000.5


def test_parse_torchtitan_log_strips_ansi_and_parses_commas():
    text = (
        "[titan] ... \x1b[31mstep:  1  \x1b[32mloss:  0.64648  "
        "\x1b[34mtps: 7,432 ...\n"
        "[titan] ... step:  2  loss:  0.59824  tps: 54,089\n"
        "wandb: 🚀 View run at https://wandb.ai/team/project/runs/xyZ0987\n"
    )
    parsed = parse_torchtitan_log(text)
    assert parsed.losses.values_by_step == {1: 0.64648, 2: 0.59824}
    assert parsed.throughput.values_by_step == {1: 7432.0, 2: 54089.0}
    assert parsed.wandb_run_id == "xyZ0987"
    assert parsed.throughput.median_excluding_first_step() == 54089.0


def test_paired_loss_diff_reports_stats():
    baseline = parse_nanovlm_log(
        "[INFO] [TRAIN] step=1 step_loss=1.0 tokens_per_second=1.0\n"
        "[INFO] [TRAIN] step=2 step_loss=0.5 tokens_per_second=1.0\n"
    ).losses
    candidate = parse_torchtitan_log(
        "[titan] step:  1 loss: 0.8 tps: 1\n"
        "[titan] step:  2 loss: 0.4 tps: 1\n"
    ).losses

    rows, stats = paired_loss_diff(baseline, candidate)
    assert len(rows) == 2
    assert stats["steps_compared"] == 2
    assert stats["max_abs_diff"] == pytest.approx(0.2)
    assert stats["step_of_max_abs_diff"] == 1
    assert stats["mean_abs_diff"] == pytest.approx(0.15)


def test_parse_torchtitan_optimizer_groups():
    text = """
[titan] INFO - Optimizer group 'lm': 272 params, lr=0.0001
[titan] INFO - Frozen group 'vision': 149 params (lr=0 → requires_grad=False)
[titan] INFO - Optimizer group 'projector': 1 params, lr=1e-05
[titan] INFO - Optimizer group 'momh_gate': 30 params, lr=0.01
"""
    groups = parse_torchtitan_optimizer_groups(text)
    assert groups == [
        {"name": "lm", "param_count": 272, "lr": 1e-4},
        {"name": "projector", "param_count": 1, "lr": 1e-5},
        {"name": "momh_gate", "param_count": 30, "lr": 1e-2},
    ]
