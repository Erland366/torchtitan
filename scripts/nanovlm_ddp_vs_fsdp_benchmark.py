#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from torchtitan.tools.nanovlm_parity import parse_torchtitan_log


OOM_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cuda out of memory",
)

WAND_URL_RE = re.compile(r"https://wandb\.ai/\S+/runs/\w+")
FAMILIES = ("ddp", "fsdp")
DEFAULT_ACTIVATE_RELATIVE_PATH = ("nanoVLM_main", ".venv", "bin", "activate")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    config_name: str
    family: str
    ac_mode: str
    local_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    steps_target: int
    return_code: int
    oom_detected: bool
    timed_out: bool
    elapsed_sec: float
    peak_mem_mib: int | None
    peak_mem_total_mib: int | None
    max_step: int
    final_loss: float | None
    median_tps_excl_step1: float | None
    log_path: str
    mem_trace_path: str
    wandb_run_url: str | None


def _divisors_desc(value: int) -> list[int]:
    divisors = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    return sorted(divisors, reverse=True)


def _safe_name(value: str) -> str:
    return value.replace("/", "_")


def _gradient_accumulation_steps(
    *, global_batch_size: int, local_batch_size: int, nproc_per_node: int
) -> int:
    denom = local_batch_size * nproc_per_node
    if global_batch_size % denom != 0:
        raise ValueError(
            "global_batch_size must be divisible by local_batch_size * nproc_per_node"
        )
    return global_batch_size // denom


def _parallelism_flags(family: str, nproc_per_node: int) -> str:
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family}")
    if family == "ddp":
        replicate_degree = nproc_per_node
        shard_degree = 1
    else:
        replicate_degree = 1
        shard_degree = nproc_per_node
    return (
        f"--parallelism.data_parallel_replicate_degree {replicate_degree} "
        f"--parallelism.data_parallel_shard_degree {shard_degree}"
    )


def _build_command(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    config_name: str,
    family: str,
    ac_mode: str,
    steps: int,
    global_batch_size: int,
    local_batch_size: int,
    nproc_per_node: int,
    dataloader_num_workers: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    compile_enable: bool,
    wandb_enable: bool,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> str:
    compile_flag = "--compile.enable" if compile_enable else "--compile.no-enable"
    wandb_flag = "" if wandb_enable else "--metrics.no-enable-wandb "
    env_prefix = _wandb_env_prefix(
        wandb_enable=wandb_enable,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    return (
        f"source {activate_path} && "
        f"cd {torchtitan_root} && "
        f"{env_prefix}"
        f"torchrun --standalone --max-restarts=0 --nproc_per_node={nproc_per_node} "
        "-m torchtitan.train "
        "--module nanoVLM "
        f"--config {config_name} "
        f"--training.steps {steps} "
        f"--training.global-batch-size {global_batch_size} "
        f"--training.local-batch-size {local_batch_size} "
        f"--dataloader.num_workers {dataloader_num_workers} "
        f"{_parallelism_flags(family, nproc_per_node)} "
        f"--comm.init-timeout-seconds {comm_init_timeout_seconds} "
        f"--comm.train-timeout-seconds {comm_train_timeout_seconds} "
        "--metrics.log_freq 1 "
        "--checkpoint.no-enable "
        f"{compile_flag} "
        f"--activation-checkpoint.mode {ac_mode} "
        f"{wandb_flag}"
    )


def _wandb_env_prefix(
    *,
    wandb_enable: bool,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> str:
    if not wandb_enable:
        return ""

    env_parts = []
    if wandb_entity:
        env_parts.append(f"WANDB_ENTITY={wandb_entity}")
    if wandb_project:
        env_parts.append(f"WANDB_PROJECT={wandb_project}")
    if wandb_run_name:
        env_parts.append(f"WANDB_RUN_NAME={wandb_run_name}")
    return f"{' '.join(env_parts)} " if env_parts else ""


def _sample_gpu_memory(
    *,
    stop_event: threading.Event,
    mem_trace_path: Path,
    samples: list[int],
    total_samples: list[int],
    errors: list[str],
) -> None:
    with mem_trace_path.open("w", encoding="utf-8") as f:
        f.write("elapsed_sec,max_memory_mib,total_memory_mib\n")
        start = time.perf_counter()
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError as exc:
                errors.append(f"nvidia-smi failed with return code {exc.returncode}")
                return
            except FileNotFoundError:
                errors.append("nvidia-smi not found")
                return

            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if not lines:
                errors.append("nvidia-smi returned empty output")
                return

            try:
                values = [int(line) for line in lines]
            except ValueError:
                errors.append(f"failed parsing nvidia-smi output: {raw!r}")
                return

            max_mib = max(values)
            total_mib = sum(values)
            samples.append(max_mib)
            total_samples.append(total_mib)
            f.write(f"{elapsed:.3f},{max_mib},{total_mib}\n")
            f.flush()
            time.sleep(1.0)


def _run_once(
    *,
    command: str,
    workdir: Path,
    log_path: Path,
    mem_trace_path: Path,
    max_run_seconds: int,
) -> tuple[int, bool, float, int | None, int | None, str]:
    stop_event = threading.Event()
    mem_samples: list[int] = []
    mem_total_samples: list[int] = []
    mem_errors: list[str] = []
    sampler_thread = threading.Thread(
        target=_sample_gpu_memory,
        kwargs={
            "stop_event": stop_event,
            "mem_trace_path": mem_trace_path,
            "samples": mem_samples,
            "total_samples": mem_total_samples,
            "errors": mem_errors,
        },
        daemon=True,
    )
    sampler_thread.start()

    start = time.perf_counter()
    timed_out = False
    run_env = os.environ.copy()
    run_env.setdefault(
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        str(max_run_seconds + 120),
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=run_env,
            preexec_fn=os.setsid,
        )
        pgid = os.getpgid(proc.pid)
        assert proc.stdout is not None
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > max_run_seconds:
                timed_out = True
                timeout_line = (
                    f"[ddp-fsdp-bench] run timed out after {max_run_seconds}s; "
                    "terminating process group.\n"
                )
                print(timeout_line, end="")
                log_file.write(timeout_line)
                log_file.flush()
                break

            if proc.poll() is not None:
                break

            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                continue

            line = proc.stdout.readline()
            if not line:
                continue
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            time.sleep(1.0)
            if proc.poll() is not None:
                break
        return_code = proc.wait()
    elapsed = time.perf_counter() - start

    stop_event.set()
    sampler_thread.join(timeout=3.0)
    if mem_errors:
        raise RuntimeError(mem_errors[0])

    peak_mem = max(mem_samples) if mem_samples else None
    peak_mem_total = max(mem_total_samples) if mem_total_samples else None
    log_text = log_path.read_text(encoding="utf-8")
    return return_code, timed_out, elapsed, peak_mem, peak_mem_total, log_text


def _is_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def _summarize_log(log_text: str) -> tuple[int, float | None, float | None, str | None]:
    parsed = parse_torchtitan_log(log_text)
    max_step = parsed.losses.max_step
    final_loss = parsed.losses.values_by_step.get(max_step) if max_step > 0 else None
    median_tps = parsed.throughput.median_excluding_first_step()
    wandb_match = WAND_URL_RE.search(log_text)
    wandb_run_url = wandb_match.group(0) if wandb_match else None
    return max_step, final_loss, median_tps, wandb_run_url


def _candidate_dir(
    *,
    output_dir: Path,
    config_name: str,
    family: str,
    ac_mode: str,
    phase_name: str,
    local_batch_size: int,
) -> Path:
    path = (
        output_dir
        / _safe_name(config_name)
        / family
        / ac_mode
        / f"{phase_name}-lb{local_batch_size}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_benchmark_case(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    output_dir: Path,
    config_name: str,
    family: str,
    ac_mode: str,
    local_batch_size: int,
    global_batch_size: int,
    steps: int,
    phase_name: str,
    nproc_per_node: int,
    dataloader_num_workers: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    run_timeout_seconds: int,
    compile_enable: bool,
    wandb_enable: bool,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> RunOutcome:
    candidate_dir = _candidate_dir(
        output_dir=output_dir,
        config_name=config_name,
        family=family,
        ac_mode=ac_mode,
        phase_name=phase_name,
        local_batch_size=local_batch_size,
    )
    log_path = candidate_dir / "train.log"
    mem_trace_path = candidate_dir / "mem.csv"
    command = _build_command(
        activate_path=activate_path,
        torchtitan_root=torchtitan_root,
        config_name=config_name,
        family=family,
        ac_mode=ac_mode,
        steps=steps,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        nproc_per_node=nproc_per_node,
        dataloader_num_workers=dataloader_num_workers,
        comm_init_timeout_seconds=comm_init_timeout_seconds,
        comm_train_timeout_seconds=comm_train_timeout_seconds,
        compile_enable=compile_enable,
        wandb_enable=wandb_enable,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    return_code, timed_out, elapsed, peak_mem, peak_mem_total, log_text = _run_once(
        command=command,
        workdir=torchtitan_root,
        log_path=log_path,
        mem_trace_path=mem_trace_path,
        max_run_seconds=run_timeout_seconds,
    )
    max_step, final_loss, median_tps, wandb_run_url = _summarize_log(log_text)
    outcome = RunOutcome(
        config_name=config_name,
        family=family,
        ac_mode=ac_mode,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=_gradient_accumulation_steps(
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            nproc_per_node=nproc_per_node,
        ),
        steps_target=steps,
        return_code=return_code,
        oom_detected=_is_oom(log_text),
        timed_out=timed_out,
        elapsed_sec=elapsed,
        peak_mem_mib=peak_mem,
        peak_mem_total_mib=peak_mem_total,
        max_step=max_step,
        final_loss=final_loss,
        median_tps_excl_step1=median_tps,
        log_path=str(log_path),
        mem_trace_path=str(mem_trace_path),
        wandb_run_url=wandb_run_url,
    )
    (candidate_dir / "run.json").write_text(
        json.dumps(asdict(outcome), indent=2),
        encoding="utf-8",
    )
    (candidate_dir / "meta.txt").write_text(
        "\n".join(
            (
                f"return_code={outcome.return_code}",
                f"elapsed_sec={outcome.elapsed_sec:.3f}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return outcome


def _valid_local_batch_sizes(
    *, global_batch_size: int, nproc_per_node: int
) -> list[int]:
    local_candidates = [
        candidate
        for candidate in _divisors_desc(global_batch_size)
        if global_batch_size % (candidate * nproc_per_node) == 0
    ]
    if not local_candidates:
        raise ValueError(
            "No valid local batch candidates for given global batch size and nproc_per_node."
        )
    return local_candidates


def _search_family_candidates(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    output_dir: Path,
    config_name: str,
    family: str,
    ac_mode: str,
    global_batch_size: int,
    search_steps: int,
    nproc_per_node: int,
    dataloader_num_workers: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    search_run_timeout_seconds: int,
    compile_enable: bool,
) -> list[RunOutcome]:
    attempts: list[RunOutcome] = []
    for local_batch_size in _valid_local_batch_sizes(
        global_batch_size=global_batch_size,
        nproc_per_node=nproc_per_node,
    ):
        outcome = _run_benchmark_case(
            activate_path=activate_path,
            torchtitan_root=torchtitan_root,
            output_dir=output_dir,
            config_name=config_name,
            family=family,
            ac_mode=ac_mode,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            steps=search_steps,
            phase_name=f"search{search_steps}",
            nproc_per_node=nproc_per_node,
            dataloader_num_workers=dataloader_num_workers,
            comm_init_timeout_seconds=comm_init_timeout_seconds,
            comm_train_timeout_seconds=comm_train_timeout_seconds,
            run_timeout_seconds=search_run_timeout_seconds,
            compile_enable=compile_enable,
            wandb_enable=False,
            wandb_entity=None,
            wandb_project=None,
            wandb_run_name=None,
        )
        attempts.append(outcome)
    return attempts


def _is_successful(outcome: RunOutcome) -> bool:
    return (
        outcome.return_code == 0
        and not outcome.oom_detected
        and not outcome.timed_out
        and outcome.max_step >= outcome.steps_target
    )


def _successful_outcome_from_dict(data: dict[str, Any] | None) -> RunOutcome | None:
    if data is None:
        return None
    outcome = RunOutcome(**data)
    return outcome if _is_successful(outcome) else None


def _winner_sort_key(outcome: RunOutcome) -> tuple[float, int, float, int]:
    peak_mem = outcome.peak_mem_mib if outcome.peak_mem_mib is not None else 10**12
    median_tps = (
        outcome.median_tps_excl_step1 if outcome.median_tps_excl_step1 is not None else 0.0
    )
    return (
        float(outcome.elapsed_sec),
        int(peak_mem),
        -float(median_tps),
        -int(outcome.local_batch_size),
    )


def _pick_family_winner(outcomes: list[RunOutcome]) -> RunOutcome | None:
    successful = [outcome for outcome in outcomes if _is_successful(outcome)]
    if not successful:
        return None
    return min(successful, key=_winner_sort_key)


def _verdict(*, ddp: RunOutcome | None, fsdp: RunOutcome | None) -> str:
    if ddp is None or fsdp is None:
        return "inconclusive"
    ddp_mem = ddp.peak_mem_mib if ddp.peak_mem_mib is not None else 10**12
    fsdp_mem = fsdp.peak_mem_mib if fsdp.peak_mem_mib is not None else 10**12
    if fsdp.elapsed_sec < ddp.elapsed_sec and fsdp_mem <= ddp_mem:
        return "fsdp_better"
    if ddp.elapsed_sec < fsdp.elapsed_sec and ddp_mem <= fsdp_mem:
        return "ddp_better"
    return "tradeoff"


def _render_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# DDP vs FSDP Benchmark Summary")
    lines.append("")
    lines.append(f"- Search steps: `{summary['search_steps']}`")
    lines.append(f"- Final steps: `{summary['final_steps']}`")
    lines.append(f"- Global batch size: `{summary['global_batch_size']}`")
    lines.append(f"- nproc_per_node: `{summary['nproc_per_node']}`")
    lines.append(f"- Families: `{', '.join(summary['families'])}`")
    lines.append(f"- AC modes: `{', '.join(summary['ac_modes'])}`")
    lines.append(f"- Compile enabled: `{summary['compile_enable']}`")
    lines.append(f"- Final W&B enabled: `{summary['enable_wandb_final']}`")
    lines.append(
        f"- Dataloader workers: `{summary['dataloader_num_workers']}`"
    )
    lines.append(f"- Output dir: `{summary['output_dir']}`")
    lines.append("")

    for config_name, config_summary in summary["configs"].items():
        lines.append(f"## Config: `{config_name}`")
        lines.append("")
        lines.append("| Family | AC | Local batch | Grad accum | Status | Elapsed (s) | Peak VRAM (MiB) | Peak Total VRAM (MiB) | Median TPS excl step1 | Final loss |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for outcome in config_summary["search_runs"]:
            status = "ok" if outcome["successful"] else "failed"
            median_tps = outcome["median_tps_excl_step1"]
            final_loss = outcome["final_loss"]
            lines.append(
                f"| `{outcome['family']}` | `{outcome['ac_mode']}` | {outcome['local_batch_size']} "
                f"| {outcome['gradient_accumulation_steps']} | `{status}` "
                f"| {outcome['elapsed_sec']:.2f} | {outcome['peak_mem_mib']} "
                f"| {outcome['peak_mem_total_mib']} "
                f"| {('%.2f' % median_tps) if median_tps is not None else 'n/a'} "
                f"| {('%.5f' % final_loss) if final_loss is not None else 'n/a'} |"
            )
        lines.append("")

        lines.append("### Final Winners")
        lines.append("")
        lines.append("| Family | AC | Local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB) | Median TPS excl step1 | Final loss | W&B |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for family in summary["families"]:
            run = config_summary["final_runs"].get(family)
            if run is None:
                lines.append(f"| `{family}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            median_tps = run["median_tps_excl_step1"]
            final_loss = run["final_loss"]
            wandb_url = run["wandb_run_url"] or "n/a"
            lines.append(
                f"| `{family}` | `{run['ac_mode']}` | {run['local_batch_size']} "
                f"| {run['gradient_accumulation_steps']} | {run['elapsed_sec']:.2f} "
                f"| {run['peak_mem_mib']} "
                f"| {('%.2f' % median_tps) if median_tps is not None else 'n/a'} "
                f"| {('%.5f' % final_loss) if final_loss is not None else 'n/a'} "
                f"| {wandb_url} |"
            )
        lines.append("")
        lines.append(f"- Verdict: `{config_summary['verdict']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _validate_args(args: argparse.Namespace) -> None:
    if args.search_steps <= 0:
        raise ValueError("--search-steps must be > 0")
    if args.final_steps <= 0:
        raise ValueError("--final-steps must be > 0")
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be > 0")
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be > 0")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader-num-workers must be >= 0")
    if args.search_run_timeout_seconds <= 0:
        raise ValueError("--search-run-timeout-seconds must be > 0")
    if args.final_run_timeout_seconds <= 0:
        raise ValueError("--final-run-timeout-seconds must be > 0")


def _resolve_activate_path(
    *, torchtitan_root: Path, venv_activate: Path | None
) -> Path:
    if venv_activate is not None:
        activate_path = venv_activate.resolve()
    else:
        activate_path = torchtitan_root.parent.joinpath(
            *DEFAULT_ACTIVATE_RELATIVE_PATH
        )
    if not activate_path.exists():
        raise FileNotFoundError(f"Missing venv activate script: {activate_path}")
    return activate_path


def _resolve_output_dir(*, torchtitan_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved = (
            torchtitan_root
            / "outputs"
            / "ddp_vs_fsdp_benchmarks"
            / f"best-of-family-{timestamp}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark best-of-family DDP vs FSDP nanoVLM training at fixed "
            "effective batch size."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "nanovlm_230m_vanilla_finevisionmax_nopack",
            "nanovlm_230m_momh_soft_gating_b5_tttv_nopack",
        ],
        help="TorchTitan config function names to benchmark.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["ddp", "fsdp"],
        choices=list(FAMILIES),
        help="Distributed families to compare.",
    )
    parser.add_argument(
        "--ac-modes",
        nargs="+",
        default=["none", "full"],
        choices=["none", "full"],
        help="Activation checkpoint modes to explore inside each family.",
    )
    parser.add_argument(
        "--search-steps",
        type=int,
        default=20,
        help="Short validation steps while searching family winners.",
    )
    parser.add_argument(
        "--final-steps",
        type=int,
        default=100,
        help="Final benchmark steps for each winning family run.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=64,
        help="Fixed effective batch size for all runs.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="torchrun nproc_per_node value.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Dataloader workers; defaults to 0 for benchmark stability.",
    )
    parser.add_argument(
        "--comm-init-timeout-seconds",
        type=int,
        default=1800,
        help="Distributed init timeout (seconds) passed to torchtitan.",
    )
    parser.add_argument(
        "--comm-train-timeout-seconds",
        type=int,
        default=300,
        help="Distributed post-step-1 train timeout (seconds) passed to torchtitan.",
    )
    parser.add_argument(
        "--compile-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile for both search and final runs.",
    )
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path("."),
        help="Path to torchtitan repository root.",
    )
    parser.add_argument(
        "--venv-activate",
        type=Path,
        default=None,
        help="Path to virtualenv activate script (default: ../nanoVLM_main/.venv/bin/activate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--search-run-timeout-seconds",
        type=int,
        default=300,
        help="Hard timeout per search trial; hung trials are force-killed.",
    )
    parser.add_argument(
        "--final-run-timeout-seconds",
        type=int,
        default=1200,
        help="Hard timeout per final benchmark run; hung runs are force-killed.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity for final runs.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional W&B project for final runs.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default="torchtitan-ddp-vs-fsdp",
        help="Prefix for final W&B run names.",
    )
    parser.add_argument(
        "--enable-wandb-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable W&B logging for final winner runs.",
    )
    args = parser.parse_args()
    _validate_args(args)

    torchtitan_root = args.torchtitan_root.resolve()
    activate_path = _resolve_activate_path(
        torchtitan_root=torchtitan_root,
        venv_activate=args.venv_activate,
    )
    output_dir = _resolve_output_dir(
        torchtitan_root=torchtitan_root,
        output_dir=args.output_dir,
    )

    summary: dict[str, Any] = {
        "benchmark": "nanovlm_ddp_vs_fsdp_benchmark",
        "created_at": datetime.now().isoformat(),
        "configs": {},
        "families": args.families,
        "ac_modes": args.ac_modes,
        "search_steps": args.search_steps,
        "final_steps": args.final_steps,
        "global_batch_size": args.global_batch_size,
        "nproc_per_node": args.nproc_per_node,
        "dataloader_num_workers": args.dataloader_num_workers,
        "compile_enable": args.compile_enable,
        "enable_wandb_final": args.enable_wandb_final,
        "output_dir": str(output_dir),
    }

    for config_name in args.configs:
        print(f"\n[ddp-fsdp-bench] config={config_name}")
        search_runs: list[dict[str, Any]] = []
        family_winners: dict[str, RunOutcome | None] = {}
        final_runs: dict[str, dict[str, Any]] = {}

        for family in args.families:
            family_attempts: list[RunOutcome] = []
            for ac_mode in args.ac_modes:
                print(
                    f"[ddp-fsdp-bench] searching family={family}, ac_mode={ac_mode}"
                )
                attempts = _search_family_candidates(
                    activate_path=activate_path,
                    torchtitan_root=torchtitan_root,
                    output_dir=output_dir,
                    config_name=config_name,
                    family=family,
                    ac_mode=ac_mode,
                    global_batch_size=args.global_batch_size,
                    search_steps=args.search_steps,
                    nproc_per_node=args.nproc_per_node,
                    dataloader_num_workers=args.dataloader_num_workers,
                    comm_init_timeout_seconds=args.comm_init_timeout_seconds,
                    comm_train_timeout_seconds=args.comm_train_timeout_seconds,
                    search_run_timeout_seconds=args.search_run_timeout_seconds,
                    compile_enable=args.compile_enable,
                )
                family_attempts.extend(attempts)

            for attempt in family_attempts:
                row = asdict(attempt)
                row["successful"] = _is_successful(attempt)
                search_runs.append(row)

            winner = _pick_family_winner(family_attempts)
            family_winners[family] = winner
            if winner is None:
                print(
                    f"[ddp-fsdp-bench] no successful search candidate for family={family}"
                )
                continue

            print(
                "[ddp-fsdp-bench] selected winner "
                f"family={family}, ac_mode={winner.ac_mode}, "
                f"local_batch_size={winner.local_batch_size}, "
                f"grad_accum={winner.gradient_accumulation_steps}"
            )
            run_name = (
                f"{args.wandb_run_prefix}-"
                f"{_safe_name(config_name)}-{family}-{winner.ac_mode}-"
                f"lb{winner.local_batch_size}-ga{winner.gradient_accumulation_steps}"
            )
            final_outcome = _run_benchmark_case(
                activate_path=activate_path,
                torchtitan_root=torchtitan_root,
                output_dir=output_dir,
                config_name=config_name,
                family=family,
                ac_mode=winner.ac_mode,
                local_batch_size=winner.local_batch_size,
                global_batch_size=args.global_batch_size,
                steps=args.final_steps,
                phase_name=f"final{args.final_steps}",
                nproc_per_node=args.nproc_per_node,
                dataloader_num_workers=args.dataloader_num_workers,
                comm_init_timeout_seconds=args.comm_init_timeout_seconds,
                comm_train_timeout_seconds=args.comm_train_timeout_seconds,
                run_timeout_seconds=args.final_run_timeout_seconds,
                compile_enable=args.compile_enable,
                wandb_enable=args.enable_wandb_final,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                wandb_run_name=run_name,
            )
            final_runs[family] = asdict(final_outcome)

        ddp_final = _successful_outcome_from_dict(final_runs.get("ddp"))
        fsdp_final = _successful_outcome_from_dict(final_runs.get("fsdp"))

        summary["configs"][config_name] = {
            "search_runs": search_runs,
            "family_winners": {
                family: asdict(winner) if winner is not None else None
                for family, winner in family_winners.items()
            },
            "final_runs": final_runs,
            "verdict": _verdict(ddp=ddp_final, fsdp=fsdp_final),
        }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_summary_md(summary), encoding="utf-8")
    print(f"\n[ddp-fsdp-bench] summary_json={summary_json}")
    print(f"[ddp-fsdp-bench] summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
