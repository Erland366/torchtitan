import importlib.util
import json
import sys
from pathlib import Path


def _load_eval_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "nanovlm_downstream_eval.py"
    )
    module_name = "nanovlm_downstream_eval_test_module"
    spec = importlib.util.spec_from_file_location(
        module_name, script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_torchtitan_backend_and_no_fallback(monkeypatch):
    module = _load_eval_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nanovlm_downstream_eval.py",
            "--checkpoint_path",
            "/tmp/checkpoint",
            "--tasks",
            "mmstar",
        ],
    )

    args = module._parse_args()

    assert args.model_backend == "torchtitan_nanovlm"
    assert args.fallback_backend == "none"


def test_main_runs_single_torchtitan_attempt_by_default(tmp_path, monkeypatch):
    module = _load_eval_module()

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_path / "model.safetensors").write_text("", encoding="utf-8")

    calls = []

    def fake_eval(**kwargs):
        calls.append(kwargs)
        return module.EvalAttemptResult(
            backend=kwargs["model_name"],
            model_name=kwargs["model_name"],
            model_args=kwargs["model_args"],
            ok=True,
            error=None,
            results={"results": {"mmstar": {"average,none": 0.5}}},
            duration_sec=1.0,
        )

    monkeypatch.setattr(module, "_run_lmms_simple_eval", fake_eval)
    monkeypatch.setattr(module, "_now_utc_str", lambda: "20260315-150000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nanovlm_downstream_eval.py",
            "--checkpoint_path",
            str(checkpoint_path),
            "--checkpoint_format",
            "hf",
            "--tasks",
            "mmstar",
            "--output_dir",
            str(tmp_path / "eval_out"),
            "--run_name",
            "default-backend-test",
        ],
    )

    assert module.main() == 0
    assert len(calls) == 1
    assert calls[0]["model_name"] == "torchtitan_nanovlm"
    assert calls[0]["model_args"] == f"model={checkpoint_path.resolve()}"
    assert calls[0]["lmms_plugins"] == "torchtitan.eval"

    metadata = json.loads(
        (
            tmp_path
            / "eval_out"
            / "default-backend-test"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["model_backend_requested"] == "torchtitan_nanovlm"
    assert metadata["fallback_backend"] == "none"
    assert metadata["backend_used"] == "torchtitan_nanovlm"
    assert metadata["primary_attempt"]["backend"] == "torchtitan_nanovlm"
    assert "fallback_attempt" not in metadata
