"""Training configuration registry for nanoVLM.

Config functions are discovered by torchtitan's ConfigManager via:
    --module nanoVLM --config nanovlm_small_debug_momh

All config functions live in the configs/ subpackage, grouped by experiment
category. This module re-exports them so ConfigManager's getattr lookup works.
"""

from .configs import *  # noqa: F401,F403
