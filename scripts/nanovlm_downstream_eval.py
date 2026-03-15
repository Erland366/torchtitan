#!/usr/bin/env python
"""Manual downstream evaluation entrypoint for TorchTitan nanoVLM.

Flow:
1) Resolve checkpoint format (HF vs DCP).
2) Convert DCP -> HF when needed.
3) Run the requested lmms-eval backend.
4) Optionally try a secondary TorchTitan fallback backend.
5) Save deterministic JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run downstream lmms-eval for TorchTitan nanoVLM checkpoints."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Checkpoint folder path (HF folder or TorchTitan DCP folder).",
    )
    parser.add_argument(
        "--checkpoint_format",
        type=str,
        choices=["auto", "hf", "dcp"],
        default="auto",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nanoVLM",
        help="Model module name for DCP->HF conversion.",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="230m_vanilla",
        help="Model flavor for DCP->HF conversion.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        default=None,
        help="HF assets path for conversion (optional for nanoVLM).",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument(
        "--model_backend",
        type=str,
        default="torchtitan_nanovlm",
        help="Primary lmms-eval model backend. Default is the local TorchTitan nanoVLM backend.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Additional model args appended to the primary backend model_args.",
    )
    parser.add_argument(
        "--fallback_backend",
        type=str,
        choices=["none", "torchtitan_plugin"],
        default="none",
    )
    parser.add_argument(
        "--fallback_model_args",
        type=str,
        default="",
        help="Additional model args appended for the fallback backend.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval_results/torchtitan"),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--keep_converted",
        action="store_true",
        help="Keep converted HF folder for DCP checkpoints.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _is_hf_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "model.safetensors").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return any(path.glob("*.safetensors"))


def _is_dcp_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / ".metadata").exists():
        return True
    return any(child.is_file() and "__" in child.name for child in path.iterdir())


def _detect_checkpoint_format(path: Path) -> str:
    if _is_hf_checkpoint(path):
        return "hf"
    if _is_dcp_checkpoint(path):
        return "dcp"
    raise ValueError(
        f"Unable to detect checkpoint format for '{path}'. "
        "Expected HF safetensors files or TorchTitan DCP metadata."
    )


def _merge_model_args(*parts: str) -> str:
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        tokens = [token.strip() for token in part.split(",") if token.strip()]
        merged.extend(tokens)
    return ",".join(merged)


def _convert_dcp_to_hf(
    *,
    dcp_path: Path,
    out_path: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: Path | None,
    export_dtype: str,
) -> dict[str, Any]:
    convert_script = _repo_root() / "scripts" / "checkpoint_conversion" / "convert_to_hf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing conversion script: {convert_script}")

    cmd = [
        sys.executable,
        str(convert_script),
        str(dcp_path),
        str(out_path),
        "--model_name",
        model_name,
        "--model_flavor",
        model_flavor,
        "--export_dtype",
        export_dtype,
    ]
    if hf_assets_path is not None:
        cmd.extend(["--hf_assets_path", str(hf_assets_path)])

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_sec": time.time() - started,
        "cmd": cmd,
    }


@dataclass
class EvalAttemptResult:
    backend: str
    model_name: str
    model_args: str
    ok: bool
    error: str | None
    results: dict[str, Any] | None
    duration_sec: float


def _resolve_eval_backend(
    *,
    backend: str,
    eval_model_path: Path,
    extra_model_args: str,
) -> tuple[str, str | None]:
    if backend == "torchtitan_nanovlm":
        return _merge_model_args(f"model={eval_model_path}", extra_model_args), "torchtitan.eval"
    return _merge_model_args(f"pretrained={eval_model_path}", extra_model_args), None


def _run_lmms_simple_eval(
    *,
    model_name: str,
    model_args: str,
    tasks: str,
    batch_size: int,
    device: str,
    limit: int | None,
    num_fewshot: int | None,
    verbosity: str,
    lmms_plugins: str | None = None,
) -> EvalAttemptResult:
    started = time.time()
    run_tmp = Path(tempfile.mkdtemp(prefix="tt_lmms_eval_"))
    result_path = run_tmp / "result.json"
    try:
        inline = r"""
import argparse
import json
from lmms_eval import evaluator
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import get_datetime_str

p = argparse.ArgumentParser()
p.add_argument("--model_name", required=True)
p.add_argument("--model_args", required=True)
p.add_argument("--tasks", required=True)
p.add_argument("--batch_size", type=int, required=True)
p.add_argument("--device", required=True)
p.add_argument("--limit", default="")
p.add_argument("--num_fewshot", default="")
p.add_argument("--verbosity", required=True)
p.add_argument("--result_path", required=True)
ns = p.parse_args()

task_manager = TaskManager(verbosity=ns.verbosity, include_path=None, model_name=ns.model_name)
task_list = [token.strip() for token in ns.tasks.split(",") if token.strip()]
task_names = task_manager.match_tasks(task_list)
missing = [task for task in task_list if task not in task_names and "*" not in task]
if missing:
    raise ValueError(f"Tasks not found: {', '.join(missing)}")

limit = None if ns.limit == "" else int(ns.limit)
num_fewshot = None if ns.num_fewshot == "" else int(ns.num_fewshot)
results = evaluator.simple_evaluate(
    model=ns.model_name,
    model_args=ns.model_args,
    tasks=task_names,
    num_fewshot=num_fewshot,
    batch_size=ns.batch_size,
    device=ns.device,
    limit=limit,
    task_manager=task_manager,
    log_samples=False,
    verbosity=ns.verbosity,
    datetime_str=get_datetime_str(),
)
if results is None:
    raise RuntimeError("lmms-eval returned None results.")
with open(ns.result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, default=str)
"""

        cmd = [
            sys.executable,
            "-c",
            inline,
            "--model_name",
            model_name,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--device",
            device,
            "--limit",
            "" if limit is None else str(limit),
            "--num_fewshot",
            "" if num_fewshot is None else str(num_fewshot),
            "--verbosity",
            verbosity,
            "--result_path",
            str(result_path),
        ]

        env = os.environ.copy()
        if lmms_plugins:
            env["LMMS_EVAL_PLUGINS"] = lmms_plugins
        repo_root = _repo_root()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
        )

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            error = (
                f"returncode={proc.returncode}\n"
                f"stdout={proc.stdout[-3000:]}\n"
                f"stderr={proc.stderr[-3000:]}"
            )
            return EvalAttemptResult(
                backend=model_name,
                model_name=model_name,
                model_args=model_args,
                ok=False,
                error=error,
                results=None,
                duration_sec=time.time() - started,
            )

        with result_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
        return EvalAttemptResult(
            backend=model_name,
            model_name=model_name,
            model_args=model_args,
            ok=True,
            error=None,
            results=results,
            duration_sec=time.time() - started,
        )
    except Exception as exc:
        return EvalAttemptResult(
            backend=model_name,
            model_name=model_name,
            model_args=model_args,
            ok=False,
            error=str(exc),
            results=None,
            duration_sec=time.time() - started,
        )
    finally:
        shutil.rmtree(run_tmp, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    args = _parse_args()
    ckpt_path = args.checkpoint_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    run_name = args.run_name or f"torchtitan-eval-{_now_utc_str()}"
    run_dir = (args.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "run_name": run_name,
        "created_utc": _now_utc_str(),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_format_requested": args.checkpoint_format,
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": args.device,
        "model_backend_requested": args.model_backend,
        "fallback_backend": args.fallback_backend,
    }

    fmt = args.checkpoint_format
    if fmt == "auto":
        fmt = _detect_checkpoint_format(ckpt_path)
    metadata["checkpoint_format_resolved"] = fmt

    eval_model_path = ckpt_path
    conversion_temp_dir: Path | None = None
    if fmt == "dcp":
        conversion_temp_dir = Path(
            tempfile.mkdtemp(prefix="tt_eval_hf_", dir=str(run_dir))
        )
        conversion = _convert_dcp_to_hf(
            dcp_path=ckpt_path,
            out_path=conversion_temp_dir,
            model_name=args.model_name,
            model_flavor=args.model_flavor,
            hf_assets_path=args.hf_assets_path,
            export_dtype=args.export_dtype,
        )
        metadata["conversion"] = conversion
        if conversion["returncode"] != 0:
            metadata["status"] = "failed"
            _write_json(run_dir / "metadata.json", metadata)
            raise RuntimeError(
                "DCP->HF conversion failed.\n"
                f"STDOUT:\n{conversion['stdout']}\nSTDERR:\n{conversion['stderr']}"
            )
        eval_model_path = conversion_temp_dir

    primary_model_args, primary_plugins = _resolve_eval_backend(
        backend=args.model_backend,
        eval_model_path=eval_model_path,
        extra_model_args=args.model_args,
    )
    primary_attempt = _run_lmms_simple_eval(
        model_name=args.model_backend,
        model_args=primary_model_args,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        verbosity=args.verbosity,
        lmms_plugins=primary_plugins,
    )
    metadata["primary_attempt"] = {
        "backend": primary_attempt.backend,
        "ok": primary_attempt.ok,
        "error": primary_attempt.error,
        "duration_sec": primary_attempt.duration_sec,
        "model_args": primary_attempt.model_args,
    }

    chosen_attempt = primary_attempt
    fallback_attempt: EvalAttemptResult | None = None
    if (
        not primary_attempt.ok
        and args.fallback_backend == "torchtitan_plugin"
        and args.model_backend != "torchtitan_nanovlm"
    ):
        fallback_model_args = _merge_model_args(
            f"model={eval_model_path}",
            args.fallback_model_args,
        )
        fallback_attempt = _run_lmms_simple_eval(
            model_name="torchtitan_nanovlm",
            model_args=fallback_model_args,
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
            verbosity=args.verbosity,
            lmms_plugins="torchtitan.eval",
        )
        metadata["fallback_attempt"] = {
            "backend": fallback_attempt.backend,
            "ok": fallback_attempt.ok,
            "error": fallback_attempt.error,
            "duration_sec": fallback_attempt.duration_sec,
            "model_args": fallback_attempt.model_args,
        }
        if fallback_attempt.ok:
            chosen_attempt = fallback_attempt

    if not chosen_attempt.ok:
        metadata["status"] = "failed"
        _write_json(run_dir / "metadata.json", metadata)
        raise RuntimeError(
            "Evaluation failed.\n"
            f"primary_error={primary_attempt.error}\n"
            f"fallback_error={fallback_attempt.error if fallback_attempt else None}"
        )

    results = chosen_attempt.results or {}
    per_task = results.get("results", {})
    summary = {
        "run_name": run_name,
        "backend_used": chosen_attempt.backend,
        "num_tasks": len(per_task),
        "tasks": sorted(per_task.keys()),
    }
    metadata["status"] = "ok"
    metadata["backend_used"] = chosen_attempt.backend
    metadata["eval_model_path"] = str(eval_model_path)
    metadata["run_dir"] = str(run_dir)

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "per_task.json", per_task)
    _write_json(run_dir / "metadata.json", metadata)

    if conversion_temp_dir is not None and not args.keep_converted:
        shutil.rmtree(conversion_temp_dir, ignore_errors=True)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
