# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor


class DynamicLibNotFoundError(RuntimeError):
    pass


class DynamicLibNotAvailableError(DynamicLibNotFoundError):
    pass


class DynamicLibUnknownError(DynamicLibNotFoundError):
    pass


@dataclass
class LoadedDL:
    abs_path: str | None
    was_already_loaded_from_elsewhere: bool
    _handle_uint: int  # Platform-agnostic unsigned pointer value
    found_via: str  # "CUDA_PATH" covers both CUDA_PATH and CUDA_HOME env vars


def load_dependencies(desc: LibDescriptor, load_func: Callable[[str], LoadedDL]) -> None:
    """Load required dependencies, then best-effort runtime dependencies.

    A plain ``DynamicLibNotFoundError`` from an optional dependency is
    suppressed. More specific contract errors and failures while loading a
    dependency that was found remain errors.
    """
    for dep in desc.dependencies:
        load_func(dep)
    for dep in desc.optional_dependencies:
        try:
            load_func(dep)
        except DynamicLibNotFoundError as exc:
            # Both public contract errors inherit DynamicLibNotFoundError, but
            # neither an unknown descriptor nor platform incompatibility means
            # that an optional runtime component is simply absent.
            if type(exc) is not DynamicLibNotFoundError:
                raise
