"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# DUMMY_WEIGHTS=1 (benchmark mode): intercept safetensors.safe_open so every
# weight tensor becomes a pinned dummy view. Must run BEFORE any
# diffusers/transformers from_pretrained is imported. Runs in both the main
# process AND spawn subprocesses (StageDiffusionProc, DiffusionWorker), because
# each re-imports vllm_omni at startup. We load dummy_weights.py standalone and
# register it under sys.modules['diffusionflow.patches.dummy_weights'] so the
# diffusers fork's `from diffusionflow.patches.dummy_weights import ...` branch
# resolves without triggering diffusionflow/__init__.py (which pulls heavy
# deps not present in the vllm_omni env).
import os as _os_dw  # noqa: E402
if _os_dw.environ.get("DUMMY_WEIGHTS") == "1":
    _repo_dw = _os_dw.environ.get("DUMMY_WEIGHTS_REPO_ROOT")
    if _repo_dw:
        try:
            import importlib.util as _ilu_dw  # noqa: E402
            import sys as _sys_dw  # noqa: E402
            import types as _types_dw  # noqa: E402

            if "diffusionflow" not in _sys_dw.modules:
                _pkg_dw = _types_dw.ModuleType("diffusionflow")
                _pkg_dw.__path__ = [_os_dw.path.join(_repo_dw, "diffusionflow")]
                _sys_dw.modules["diffusionflow"] = _pkg_dw
            if "diffusionflow.patches" not in _sys_dw.modules:
                _sub_dw = _types_dw.ModuleType("diffusionflow.patches")
                _sub_dw.__path__ = [_os_dw.path.join(_repo_dw, "diffusionflow", "patches")]
                _sys_dw.modules["diffusionflow.patches"] = _sub_dw

            _dw_path = _os_dw.path.join(_repo_dw, "diffusionflow", "patches", "dummy_weights.py")
            _spec_dw = _ilu_dw.spec_from_file_location("diffusionflow.patches.dummy_weights", _dw_path)
            _mod_dw = _ilu_dw.module_from_spec(_spec_dw)
            _sys_dw.modules["diffusionflow.patches.dummy_weights"] = _mod_dw
            _spec_dw.loader.exec_module(_mod_dw)
            _mod_dw.install_fake_safe_open()
        except Exception as _e_dw:
            print(f"[vllm_omni] DUMMY_WEIGHTS=1 but failed to load dummy_weights: {_e_dw}", flush=True)
    else:
        print("[vllm_omni] DUMMY_WEIGHTS=1 but DUMMY_WEIGHTS_REPO_ROOT not set", flush=True)

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402

from .config import OmniModelConfig
from .entrypoints import AsyncOmni, Omni

from .version import __version__, __version_tuple__  # isort:skip


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
