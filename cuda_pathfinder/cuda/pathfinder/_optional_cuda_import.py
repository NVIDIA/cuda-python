# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError


def _optional_cuda_import(
    fully_qualified_modname: str,
    *,
    probe_function: Callable[[ModuleType], object] | None = None,
) -> ModuleType | None:
    """Import an optional CUDA module without masking unrelated import bugs.

    Returns:
        The imported module if available and the optional probe succeeds,
        otherwise ``None`` when the requested module is unavailable.

    Raises:
        ModuleNotFoundError: If the import fails because a dependency of the
            target module is missing (instead of the target module itself).
        Exception: Any exception raised by ``probe_function`` except
            :class:`DynamicLibNotFoundError`, which is treated as "unavailable".
    """
    try:
        module = importlib.import_module(fully_qualified_modname)
    except ModuleNotFoundError as err:
        if err.name != fully_qualified_modname:
            raise
        return None

    if probe_function is not None:
        try:
            probe_function(module)
        except DynamicLibNotFoundError:
            return None

    return module
