# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError


def _target_is_unavailable(missing_modname: str | None, fully_qualified_modname: str) -> bool:
    """Report whether a ModuleNotFoundError means "the target is not installed".

    ``ModuleNotFoundError.name`` is the module that could not be found, which
    for ``import a.b.c`` is the *outermost* missing name: if package ``a.b`` is
    not installed at all, ``name`` is ``"a.b"``, not ``"a.b.c"``. A missing
    ancestor makes the target just as unavailable as a missing leaf, so both
    count. Anything else is a broken dependency of the target and must not be
    swallowed.
    """
    if missing_modname is None:
        return False
    return fully_qualified_modname == missing_modname or fully_qualified_modname.startswith(missing_modname + ".")


def _optional_cuda_import(
    fully_qualified_modname: str,
    *,
    probe_function: Callable[[ModuleType], object] | None = None,
) -> ModuleType | None:
    """Import an optional CUDA module without masking unrelated import bugs.

    Returns:
        The imported module if available and the optional probe succeeds,
        otherwise ``None`` when the requested module — or a package containing
        it — is not installed.

    Raises:
        ModuleNotFoundError: If the import fails because a dependency of the
            target module is missing (instead of the target module itself).
        Exception: Any exception raised by ``probe_function`` except
            :class:`DynamicLibNotFoundError`, which is treated as "unavailable".
    """
    try:
        module = importlib.import_module(fully_qualified_modname)
    except ModuleNotFoundError as err:
        if not _target_is_unavailable(err.name, fully_qualified_modname):
            raise
        return None

    if probe_function is not None:
        try:
            probe_function(module)
        except DynamicLibNotFoundError:
            return None

    return module
