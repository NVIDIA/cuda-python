# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    found_via: str


def load_dependencies(desc: LibDescriptor, load_func: Callable[[str], LoadedDL]) -> None:
    for dep in desc.dependencies:
        load_func(dep)
