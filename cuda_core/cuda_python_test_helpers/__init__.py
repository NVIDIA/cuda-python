# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Re-export helpers from the repository root `cuda_python_test_helpers` without
# causing a circular import by loading the root module by file path.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[2]
_helpers_path = _root / "cuda_python_test_helpers" / "__init__.py"

_spec = spec_from_file_location("_cuda_python_test_helpers_root", _helpers_path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to locate top-level cuda_python_test_helpers module")

_module = module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

# Re-export symbols
_detect_wsl = getattr(_module, "_detect_wsl")
IS_WSL = getattr(_module, "IS_WSL")
skip_on_wsl = getattr(_module, "skip_on_wsl")
supports_ipc_mempool = getattr(_module, "supports_ipc_mempool")

__all__ = [
    "_detect_wsl",
    "IS_WSL",
    "skip_on_wsl",
    "supports_ipc_mempool",
]
