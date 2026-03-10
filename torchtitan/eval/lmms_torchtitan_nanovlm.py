"""Thin lmms-eval model class for TorchTitan nanoVLM fallback eval.

This module deliberately reuses nanoVLM_main's existing lmms wrapper behavior
to minimize duplicated model-specific prompt/media logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

def _load_nanovlm_wrapper_class():
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    nanovlm_main_root = repo_root / "nanoVLM_main"
    wrapper_path = nanovlm_main_root / "eval" / "lmms_eval_wrapper.py"

    if not wrapper_path.exists():
        raise FileNotFoundError(
            f"Cannot find nanoVLM_main wrapper at: {wrapper_path}"
        )

    # The wrapper imports `models.*` and other project-local modules from
    # nanoVLM_main. Ensure that root is importable before loading the file.
    root_str = str(nanovlm_main_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    spec = importlib.util.spec_from_file_location(
        "_torchtitan_nanovlm_main_lmms_wrapper",
        str(wrapper_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NanoVLMWrapper


NanoVLMWrapperBase = _load_nanovlm_wrapper_class()


class TorchTitanNanoVLM(NanoVLMWrapperBase):
    """Fallback lmms model entry for TorchTitan downstream eval."""
