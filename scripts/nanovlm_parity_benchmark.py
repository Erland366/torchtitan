#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from torchtitan.tools.nanovlm_parity import (
    paired_loss_diff,
    parse_nanovlm_log,
    parse_torchtitan_optimizer_groups,
    parse_torchtitan_log,
)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    log_path: Path
    mem_trace_path: Path
    return_code: int
    elapsed_sec: float
    peak_mem_mib: int | None
    peak_mem_at_sec: float | None


def _sample_gpu_memory(mem_trace_path: Path, stop_event: threading.Event) -> None:
    start = time.perf_counter()
    with mem_trace_path.open("w", encoding="utf-8") as f:
        f.write("elapsed_sec,memory_mib\n")
        while not stop_event.is_set():
            now = time.perf_counter() - start
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            except Exception:
                # If nvidia-smi is unavailable, stop sampling without failing
                # the run command itself.
                break
            first_line = out.splitlines()[0].strip() if out else ""
            try:
                mib = int(first_line)
            except ValueError:
                break
            f.write(f"{now:.3f},{mib}\n")
            f.flush()
            time.sleep(1.0)


def _compute_peak_memory(mem_trace_path: Path) -> tuple[int | None, float | None]:
    if not mem_trace_path.exists():
        return None, None

    peak_mem: int | None = None
    peak_at: float | None = None
    with mem_trace_path.open("r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            elapsed_text, mib_text = line.split(",", maxsplit=1)
            elapsed = float(elapsed_text)
            mib = int(mib_text)
            if peak_mem is None or mib > peak_mem:
                peak_mem = mib
                peak_at = elapsed
    return peak_mem, peak_at


def _run_command(
    *,
    command: str,
    workdir: Path,
    log_path: Path,
    mem_trace_path: Path,
    env_overrides: dict[str, str],
) -> RunOutcome:
    env = os.environ.copy()
    env.update(env_overrides)

    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu_memory,
        args=(mem_trace_path, stop_event),
        daemon=True,
    )
    sampler.start()

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = proc.wait()
    elapsed_sec = time.perf_counter() - start

    stop_event.set()
    sampler.join(timeout=3.0)
    peak_mem_mib, peak_mem_at_sec = _compute_peak_memory(mem_trace_path)

    return RunOutcome(
        log_path=log_path,
        mem_trace_path=mem_trace_path,
        return_code=return_code,
        elapsed_sec=elapsed_sec,
        peak_mem_mib=peak_mem_mib,
        peak_mem_at_sec=peak_mem_at_sec,
    )


def _format_markdown_summary(summary: dict) -> str:
    lines = [
        "# nanoVLM vs Torchtitan Parity Summary",
        "",
        "## Run Metadata",
        f"- mode: `{summary['mode']}`",
        f"- steps_target: `{summary['steps_target']}`",
        f"- output_dir: `{summary['output_dir']}`",
        "",
        "## Baseline (nanoVLM_main)",
        f"- return_code: `{summary['baseline']['return_code']}`",
        f"- elapsed_sec: `{summary['baseline']['elapsed_sec']:.2f}`",
        f"- peak_mem_mib: `{summary['baseline']['peak_mem_mib']}`",
        f"- median_tps_excl_step1: `{summary['baseline']['median_tps_excl_step1']}`",
        f"- wandb_run_id: `{summary['baseline']['wandb_run_id']}`",
        "",
        "## Candidate (Torchtitan)",
        f"- return_code: `{summary['torchtitan']['return_code']}`",
        f"- elapsed_sec: `{summary['torchtitan']['elapsed_sec']:.2f}`",
        f"- peak_mem_mib: `{summary['torchtitan']['peak_mem_mib']}`",
        f"- median_tps_excl_step1: `{summary['torchtitan']['median_tps_excl_step1']}`",
        f"- wandb_run_id: `{summary['torchtitan']['wandb_run_id']}`",
        "",
        "## Parity",
        f"- steps_compared: `{summary['loss_diff']['steps_compared']}`",
        f"- mean_abs_diff: `{summary['loss_diff']['mean_abs_diff']}`",
        f"- max_abs_diff: `{summary['loss_diff']['max_abs_diff']}`",
        f"- step_of_max_abs_diff: `{summary['loss_diff']['step_of_max_abs_diff']}`",
    ]
    return "\n".join(lines) + "\n"


def _baseline_can_ignore_return_code(
    *,
    return_code: int,
    parsed_max_step: int,
    steps_target: int,
    log_text: str,
) -> bool:
    if return_code == 0 or parsed_max_step < steps_target:
        return False
    shutdown_signatures = (
        "Fatal Python error: PyGILState_Release",
        "Python runtime state: finalizing",
        "terminate called without an active exception",
    )
    return any(signature in log_text for signature in shutdown_signatures)


def _build_baseline_config(
    *,
    source_config_path: Path,
    output_config_path: Path,
    steps: int,
    run_name_prefix: str,
    wandb_entity: str,
    wandb_project: str,
) -> None:
    with source_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "train" not in config:
        raise ValueError(f"Unexpected config structure: {source_config_path}")

    train_cfg = config["train"]
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Unexpected train section in: {source_config_path}")

    train_cfg["max_training_steps"] = steps
    train_cfg["stats_log_interval"] = 1
    train_cfg["stop_unit"] = "steps"
    train_cfg["wandb_entity"] = wandb_entity
    train_cfg["wandb_project"] = wandb_project
    train_cfg["wandb_run_name_prefix"] = run_name_prefix
    train_cfg["push_checkpoints_to_hub"] = False
    train_cfg["push_final_model_to_hub"] = False

    with output_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible 100-step parity benchmark between nanoVLM_main "
            "and Torchtitan with external VRAM sampling."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["vanilla", "soft-gating"],
        default="vanilla",
        help="Which parity track to run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of optimizer steps to run for each framework.",
    )
    parser.add_argument(
        "--nanovlm-root",
        type=Path,
        default=Path("../nanoVLM_main"),
        help="Path to nanoVLM_main repository.",
    )
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path("."),
        help="Path to torchtitan repository.",
    )
    parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("../nanoVLM_main/.venv/bin/activate"),
        help="Path to the shared virtualenv activate script.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="patrickirawan-mbzuai",
        help="W&B entity/team.",
    )
    parser.add_argument(
        "--wandb-project",
        default="momh",
        help="W&B project used by both runs.",
    )
    parser.add_argument(
        "--run-suffix",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Suffix appended to run names and output folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for logs and summaries.",
    )
    parser.add_argument(
        "--max-abs-loss-diff",
        type=float,
        default=None,
        help=(
            "If set, exits non-zero when max absolute loss diff exceeds this threshold."
        ),
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    torchtitan_root = args.torchtitan_root.resolve()
    nanovlm_root = args.nanovlm_root.resolve()
    activate_path = args.venv_activate.resolve()

    if not activate_path.exists():
        raise FileNotFoundError(f"Missing venv activate script: {activate_path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            torchtitan_root
            / "outputs"
            / "nanovlm_parity_benchmarks"
            / f"{args.mode}-{args.run_suffix}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "vanilla":
        baseline_cfg = "configs/train.paper.vanilla-finevisionmax.nopack.yaml"
        torchtitan_cfg = "nanovlm_230m_vanilla_finevisionmax_nopack"
    else:
        baseline_cfg = "configs/train.paper.momh.soft-gating-b5-tttv.nopack.yaml"
        torchtitan_cfg = "nanovlm_230m_momh_soft_gating_b5_tttv_nopack"

    baseline_name = f"baseline-{args.mode}-{args.steps}step-{args.run_suffix}"
    torchtitan_name = f"torchtitan-{args.mode}-{args.steps}step-{args.run_suffix}"

    baseline_source_cfg = nanovlm_root / baseline_cfg
    baseline_runtime_cfg = output_dir / "baseline.runtime.yaml"
    _build_baseline_config(
        source_config_path=baseline_source_cfg,
        output_config_path=baseline_runtime_cfg,
        steps=args.steps,
        run_name_prefix=baseline_name,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )

    baseline_cmd = (
        f"source {activate_path} && "
        f"python train.py --config {baseline_runtime_cfg}"
    )

    torchtitan_cmd = (
        f"source {activate_path} && "
        "torchrun --nproc_per_node=1 -m torchtitan.train "
        "--module nanoVLM "
        f"--config {torchtitan_cfg} "
        f"--training.steps {args.steps} "
        "--metrics.log_freq 1 "
        f"--checkpoint.folder parity_{args.mode}_{args.steps}_{args.run_suffix}"
    )

    env_common = {
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
    }

    print(f"[parity] output_dir={output_dir}")
    print(f"[parity] mode={args.mode}, steps={args.steps}")

    baseline_outcome = _run_command(
        command=baseline_cmd,
        workdir=nanovlm_root,
        log_path=output_dir / "baseline.log",
        mem_trace_path=output_dir / "baseline.mem.csv",
        env_overrides=env_common,
    )
    baseline_text = baseline_outcome.log_path.read_text(encoding="utf-8")
    baseline_parsed = parse_nanovlm_log(baseline_text)
    if baseline_outcome.return_code != 0:
        if _baseline_can_ignore_return_code(
            return_code=baseline_outcome.return_code,
            parsed_max_step=baseline_parsed.losses.max_step,
            steps_target=args.steps,
            log_text=baseline_text,
        ):
            print(
                "[parity] baseline returned non-zero after full steps due known "
                "shutdown finalization crash; continuing with parsed metrics."
            )
        else:
            print("[parity] baseline run failed", file=sys.stderr)
            return baseline_outcome.return_code

    torchtitan_outcome = _run_command(
        command=torchtitan_cmd,
        workdir=torchtitan_root,
        log_path=output_dir / "torchtitan.log",
        mem_trace_path=output_dir / "torchtitan.mem.csv",
        env_overrides={
            **env_common,
            "WANDB_RUN_NAME": torchtitan_name,
        },
    )
    if torchtitan_outcome.return_code != 0:
        print("[parity] torchtitan run failed", file=sys.stderr)
        return torchtitan_outcome.return_code

    torchtitan_text = torchtitan_outcome.log_path.read_text(encoding="utf-8")
    torchtitan_parsed = parse_torchtitan_log(torchtitan_text)
    torchtitan_optimizer_groups = parse_torchtitan_optimizer_groups(torchtitan_text)

    paired_rows, diff_stats = paired_loss_diff(
        baseline_parsed.losses, torchtitan_parsed.losses
    )

    summary = {
        "mode": args.mode,
        "steps_target": args.steps,
        "output_dir": str(output_dir),
        "baseline": {
            "return_code": baseline_outcome.return_code,
            "elapsed_sec": baseline_outcome.elapsed_sec,
            "peak_mem_mib": baseline_outcome.peak_mem_mib,
            "peak_mem_at_sec": baseline_outcome.peak_mem_at_sec,
            "steps_logged": baseline_parsed.losses.max_step,
            "median_tps_excl_step1": baseline_parsed.throughput.median_excluding_first_step(),
            "wandb_run_id": baseline_parsed.wandb_run_id,
        },
        "torchtitan": {
            "return_code": torchtitan_outcome.return_code,
            "elapsed_sec": torchtitan_outcome.elapsed_sec,
            "peak_mem_mib": torchtitan_outcome.peak_mem_mib,
            "peak_mem_at_sec": torchtitan_outcome.peak_mem_at_sec,
            "steps_logged": torchtitan_parsed.losses.max_step,
            "median_tps_excl_step1": torchtitan_parsed.throughput.median_excluding_first_step(),
            "wandb_run_id": torchtitan_parsed.wandb_run_id,
            "optimizer_groups": torchtitan_optimizer_groups,
        },
        "loss_diff": diff_stats,
        "validity": {
            "baseline_full_steps": baseline_parsed.losses.max_step >= args.steps,
            "torchtitan_full_steps": torchtitan_parsed.losses.max_step >= args.steps,
            "paired_full_steps": diff_stats["steps_compared"] >= args.steps,
        },
        "paired_loss_rows": [
            {
                "step": step,
                "baseline_loss": baseline_loss,
                "torchtitan_loss": torchtitan_loss,
                "abs_diff": abs_diff,
            }
            for step, baseline_loss, torchtitan_loss, abs_diff in paired_rows
        ],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _format_markdown_summary(summary),
        encoding="utf-8",
    )

    print(f"[parity] summary.json: {output_dir / 'summary.json'}")
    print(f"[parity] summary.md: {output_dir / 'summary.md'}")

    if diff_stats["steps_compared"] < args.steps:
        print(
            "[parity] not enough overlapping steps in parsed losses: "
            f"{diff_stats['steps_compared']} < {args.steps}",
            file=sys.stderr,
        )
        return 2

    if args.max_abs_loss_diff is not None:
        max_abs = float(diff_stats["max_abs_diff"])
        if max_abs > args.max_abs_loss_diff:
            print(
                f"[parity] max_abs_loss_diff={max_abs:.6f} "
                f"exceeds threshold={args.max_abs_loss_diff:.6f}",
                file=sys.stderr,
            )
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
