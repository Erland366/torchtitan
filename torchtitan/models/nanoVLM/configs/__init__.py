"""Training configuration subpackage for nanoVLM.

Each submodule groups config functions by experiment category (debug, paper, etc.).
All config functions are re-exported here so that config_registry.py can expose
them via a single wildcard import.
"""

from .debug import *  # noqa: F401,F403
from .paper import *  # noqa: F401,F403
