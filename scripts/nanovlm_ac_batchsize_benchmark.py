#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import statistics
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


@dataclass(frozen=True, slots=True)
class RunOutcome:
    config_name: str
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


def _divisors_desc(value: int) -> list[int]:
    divisors = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    return sorted(divisors, reverse=True)


def _build_command(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    config_name: str,
    ac_mode: str,
    steps: int,
    global_batch_size: int,
    local_batch_size: int,
    nproc_per_node: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    compile_enable: bool,
) -> str:
    compile_flag = "--compile.enable" if compile_enable else "--compile.no-enable"
    return (
        f"source {activate_path} && "
        f"cd {torchtitan_root} && "
        f"torchrun --standalone --max-restarts=0 --nproc_per_node={nproc_per_node} -m torchtitan.train "
        "--module nanoVLM "
        f"--config {config_name} "
        f"--training.steps {steps} "
        f"--training.global-batch-size {global_batch_size} "
        f"--training.local-batch-size {local_batch_size} "
        f"--comm.init-timeout-seconds {comm_init_timeout_seconds} "
        f"--comm.train-timeout-seconds {comm_train_timeout_seconds} "
        "--metrics.log_freq 1 "
        "--metrics.no-enable-wandb "
        "--checkpoint.no-enable "
        f"{compile_flag} "
        f"--activation-checkpoint.mode {ac_mode}"
    )


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
                    f"[ac-bench] run timed out after {max_run_seconds}s; "
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


def _summarize_log(log_text: str) -> tuple[int, float | None, float | None]:
    parsed = parse_torchtitan_log(log_text)
    max_step = parsed.losses.max_step
    final_loss = parsed.losses.values_by_step.get(max_step) if max_step > 0 else None
    median_tps = parsed.throughput.median_excluding_first_step()
    return max_step, final_loss, median_tps


def _run_benchmark_case(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    output_dir: Path,
    config_name: str,
    ac_mode: str,
    local_batch_size: int,
    global_batch_size: int,
    steps: int,
    phase_name: str,
    nproc_per_node: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    run_timeout_seconds: int,
    compile_enable: bool,
) -> RunOutcome:
    safe_config = config_name.replace("/", "_")
    log_path = output_dir / f"{safe_config}.{ac_mode}.{phase_name}.log"
    mem_trace_path = output_dir / f"{safe_config}.{ac_mode}.{phase_name}.mem.csv"
    command = _build_command(
        activate_path=activate_path,
        torchtitan_root=torchtitan_root,
        config_name=config_name,
        ac_mode=ac_mode,
        steps=steps,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        nproc_per_node=nproc_per_node,
        comm_init_timeout_seconds=comm_init_timeout_seconds,
        comm_train_timeout_seconds=comm_train_timeout_seconds,
        compile_enable=compile_enable,
    )
    return_code, timed_out, elapsed, peak_mem, peak_mem_total, log_text = _run_once(
        command=command,
        workdir=torchtitan_root,
        log_path=log_path,
        mem_trace_path=mem_trace_path,
        max_run_seconds=run_timeout_seconds,
    )
    max_step, final_loss, median_tps = _summarize_log(log_text)
    return RunOutcome(
        config_name=config_name,
        ac_mode=ac_mode,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=global_batch_size
        // (local_batch_size * nproc_per_node),
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
    )


def _search_max_local_batch(
    *,
    activate_path: Path,
    torchtitan_root: Path,
    output_dir: Path,
    config_name: str,
    ac_mode: str,
    global_batch_size: int,
    search_steps: int,
    nproc_per_node: int,
    comm_init_timeout_seconds: int,
    comm_train_timeout_seconds: int,
    search_run_timeout_seconds: int,
) -> tuple[int, list[RunOutcome]]:
    attempts: list[RunOutcome] = []
    local_candidates = [
        candidate
        for candidate in _divisors_desc(global_batch_size)
        if global_batch_size % (candidate * nproc_per_node) == 0
    ]
    if not local_candidates:
        raise ValueError(
            "No valid local batch candidates for given global batch size and nproc_per_node."
        )

    for local_batch_size in local_candidates:
        outcome = _run_benchmark_case(
            activate_path=activate_path,
            torchtitan_root=torchtitan_root,
            output_dir=output_dir,
            config_name=config_name,
            ac_mode=ac_mode,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            steps=search_steps,
            phase_name=f"search{search_steps}",
            nproc_per_node=nproc_per_node,
            comm_init_timeout_seconds=comm_init_timeout_seconds,
            comm_train_timeout_seconds=comm_train_timeout_seconds,
            run_timeout_seconds=search_run_timeout_seconds,
            compile_enable=False,
        )
        attempts.append(outcome)
        if (
            outcome.return_code == 0
            and not outcome.oom_detected
            and outcome.max_step >= search_steps
        ):
            return local_batch_size, attempts
    raise RuntimeError(
        f"No feasible local batch found for config={config_name}, ac_mode={ac_mode}."
    )


def _float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _render_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# AC Max-Batch Benchmark Summary")
    lines.append("")
    lines.append(f"- Steps: `{summary['steps']}`")
    lines.append(f"- Search steps: `{summary['search_steps']}`")
    lines.append(f"- Global batch size: `{summary['global_batch_size']}`")
    lines.append(f"- nproc_per_node: `{summary['nproc_per_node']}`")
    lines.append(
        f"- comm_init_timeout_seconds: `{summary['comm_init_timeout_seconds']}`"
    )
    lines.append(
        f"- comm_train_timeout_seconds: `{summary['comm_train_timeout_seconds']}`"
    )
    lines.append(
        f"- search_run_timeout_seconds: `{summary['search_run_timeout_seconds']}`"
    )
    lines.append(
        f"- final_run_timeout_seconds: `{summary['final_run_timeout_seconds']}`"
    )
    lines.append(f"- Output dir: `{summary['output_dir']}`")
    lines.append("")

    for config_name, config_summary in summary["configs"].items():
        lines.append(f"## Config: `{config_name}`")
        lines.append("")
        lines.append("| Mode | Status | Max local batch | Grad accum | Elapsed (s) | Peak VRAM (MiB, any GPU) | Peak VRAM total (MiB) | Median TPS excl step1 | Final loss | Max step |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        mode_errors = config_summary.get("mode_errors", {})
        for mode in summary["modes"]:
            run = config_summary["final_runs"].get(mode)
            if run is None:
                status = mode_errors.get(mode, "no_final_run")
                lines.append(
                    f"| `{mode}` | `{status}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            median_tps = run["median_tps_excl_step1"]
            final_loss = run["final_loss"]
            lines.append(
                f"| `{mode}` | `ok` | {run['local_batch_size']} | {run['gradient_accumulation_steps']} "
                f"| {run['elapsed_sec']:.2f} | {run['peak_mem_mib']} "
                f"| {run['peak_mem_total_mib']} "
                f"| {median_tps:.2f} | {final_loss:.5f} | {run['max_step']} |"
            )
        lines.append("")
        delta = config_summary.get("delta_full_minus_none")
        if delta is None:
            lines.append("Delta (`full - none`): `n/a`")
        else:
            lines.append("Delta (`full - none`):")
            lines.append(f"- elapsed_sec: `{delta['elapsed_sec']:.2f}`")
            lines.append(f"- peak_mem_mib: `{delta['peak_mem_mib']}`")
            lines.append(f"- peak_mem_total_mib: `{delta['peak_mem_total_mib']}`")
            lines.append(
                f"- median_tps_excl_step1: `{delta['median_tps_excl_step1']:.2f}`"
            )
            lines.append(f"- final_loss: `{delta['final_loss']:.5f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find the highest feasible local batch size for activation-checkpoint "
            "modes (none/full) with fixed global batch size, then run 100-step "
            "TorchTitan benchmarks."
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
        "--steps",
        type=int,
        default=100,
        help="Final benchmark steps per mode.",
    )
    parser.add_argument(
        "--search-steps",
        type=int,
        default=20,
        help="Short validation steps while searching max local batch.",
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
        default=1,
        help="torchrun nproc_per_node value.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "full"],
        choices=["none", "full"],
        help="Activation checkpoint modes to compare.",
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
        "--torchtitan-root",
        type=Path,
        default=Path("."),
        help="Path to torchtitan repository root.",
    )
    parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("../nanoVLM_main/.venv/bin/activate"),
        help="Path to shared virtualenv activate script.",
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
        default=1200,
        help="Hard timeout per search trial; hung trials are force-killed.",
    )
    parser.add_argument(
        "--final-run-timeout-seconds",
        type=int,
        default=7200,
        help="Hard timeout per final benchmark run; hung runs are force-killed.",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.search_steps <= 0:
        raise ValueError("--search-steps must be > 0")
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be > 0")
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be > 0")
    if args.comm_init_timeout_seconds <= 0:
        raise ValueError("--comm-init-timeout-seconds must be > 0")
    if args.comm_train_timeout_seconds <= 0:
        raise ValueError("--comm-train-timeout-seconds must be > 0")
    if args.search_run_timeout_seconds <= 0:
        raise ValueError("--search-run-timeout-seconds must be > 0")
    if args.final_run_timeout_seconds <= 0:
        raise ValueError("--final-run-timeout-seconds must be > 0")
    if not args.modes:
        raise ValueError("--modes must include at least one mode.")

    torchtitan_root = args.torchtitan_root.resolve()
    activate_path = args.venv_activate.resolve()
    if not activate_path.exists():
        raise FileNotFoundError(f"Missing venv activate script: {activate_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            torchtitan_root
            / "outputs"
            / "ac_benchmarks"
            / f"max-batch-ac-compare-{timestamp}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark": "nanovlm_ac_batchsize_benchmark",
        "created_at": datetime.now().isoformat(),
        "steps": args.steps,
        "search_steps": args.search_steps,
        "global_batch_size": args.global_batch_size,
        "nproc_per_node": args.nproc_per_node,
        "comm_init_timeout_seconds": args.comm_init_timeout_seconds,
        "comm_train_timeout_seconds": args.comm_train_timeout_seconds,
        "search_run_timeout_seconds": args.search_run_timeout_seconds,
        "final_run_timeout_seconds": args.final_run_timeout_seconds,
        "modes": args.modes,
        "output_dir": str(output_dir),
        "configs": {},
    }

    for config_name in args.configs:
        print(f"\n[ac-bench] config={config_name}")
        config_summary: dict[str, Any] = {
            "search_runs": {},
            "final_runs": {},
            "mode_errors": {},
            "delta_full_minus_none": None,
        }

        for mode in args.modes:
            print(f"[ac-bench] searching max local batch for mode={mode}")
            try:
                max_local_batch, search_attempts = _search_max_local_batch(
                    activate_path=activate_path,
                    torchtitan_root=torchtitan_root,
                    output_dir=output_dir,
                    config_name=config_name,
                    ac_mode=mode,
                    global_batch_size=args.global_batch_size,
                    search_steps=args.search_steps,
                    nproc_per_node=args.nproc_per_node,
                    comm_init_timeout_seconds=args.comm_init_timeout_seconds,
                    comm_train_timeout_seconds=args.comm_train_timeout_seconds,
                    search_run_timeout_seconds=args.search_run_timeout_seconds,
                )
                config_summary["search_runs"][mode] = [
                    asdict(attempt) for attempt in search_attempts
                ]
                print(
                    "[ac-bench] selected local batch "
                    f"{max_local_batch} for config={config_name}, mode={mode}"
                )
            except Exception as exc:
                config_summary["search_runs"][mode] = []
                config_summary["mode_errors"][mode] = f"search_failed: {exc}"
                print(
                    f"[ac-bench] mode={mode} skipped for config={config_name}: "
                    f"{config_summary['mode_errors'][mode]}"
                )
                continue

            try:
                final_outcome = _run_benchmark_case(
                    activate_path=activate_path,
                    torchtitan_root=torchtitan_root,
                    output_dir=output_dir,
                    config_name=config_name,
                    ac_mode=mode,
                    local_batch_size=max_local_batch,
                    global_batch_size=args.global_batch_size,
                    steps=args.steps,
                    phase_name=f"final{args.steps}",
                    nproc_per_node=args.nproc_per_node,
                    comm_init_timeout_seconds=args.comm_init_timeout_seconds,
                    comm_train_timeout_seconds=args.comm_train_timeout_seconds,
                    run_timeout_seconds=args.final_run_timeout_seconds,
                    compile_enable=True,
                )
                if final_outcome.return_code != 0:
                    raise RuntimeError("final run returned non-zero exit code")
                if final_outcome.max_step < args.steps:
                    raise RuntimeError(
                        f"final run reached step {final_outcome.max_step}, expected {args.steps}"
                    )
                config_summary["final_runs"][mode] = asdict(final_outcome)
            except Exception as exc:
                config_summary["mode_errors"][mode] = f"final_failed: {exc}"
                print(
                    f"[ac-bench] mode={mode} final failed for config={config_name}: "
                    f"{config_summary['mode_errors'][mode]}"
                )

        if "none" in config_summary["final_runs"] and "full" in config_summary["final_runs"]:
            none_run = config_summary["final_runs"]["none"]
            full_run = config_summary["final_runs"]["full"]

            none_median_tps = _float(none_run["median_tps_excl_step1"]) or 0.0
            full_median_tps = _float(full_run["median_tps_excl_step1"]) or 0.0
            none_final_loss = _float(none_run["final_loss"]) or 0.0
            full_final_loss = _float(full_run["final_loss"]) or 0.0
            none_peak_mem = none_run["peak_mem_mib"] or 0
            full_peak_mem = full_run["peak_mem_mib"] or 0
            none_peak_mem_total = none_run["peak_mem_total_mib"] or 0
            full_peak_mem_total = full_run["peak_mem_total_mib"] or 0

            config_summary["delta_full_minus_none"] = {
                "elapsed_sec": float(full_run["elapsed_sec"]) - float(none_run["elapsed_sec"]),
                "peak_mem_mib": int(full_peak_mem) - int(none_peak_mem),
                "peak_mem_total_mib": int(full_peak_mem_total)
                - int(none_peak_mem_total),
                "median_tps_excl_step1": float(full_median_tps) - float(none_median_tps),
                "final_loss": float(full_final_loss) - float(none_final_loss),
            }
        summary["configs"][config_name] = config_summary

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_summary_md(summary), encoding="utf-8")
    print(f"\n[ac-bench] summary_json={summary_json}")
    print(f"[ac-bench] summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
