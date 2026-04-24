# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Protocol, TypeVar, cast

from cuda.pathfinder._binaries.find_nvidia_binary_utility import (
    find_nvidia_binary_utility as _find_nvidia_binary_utility,
)
from cuda.pathfinder._compatibility_guard_rails import (
    CompatibilityGuardRails,
    CompatibilityInsufficientMetadataError,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)
from cuda.pathfinder._headers.find_nvidia_headers import (
    LocatedHeaderDir,
)
from cuda.pathfinder._headers.find_nvidia_headers import (
    find_nvidia_header_directory as _find_nvidia_header_directory_impl,
)
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as _locate_nvidia_header_directory,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    LocatedBitcodeLib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    find_bitcode_lib as _find_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    locate_bitcode_lib as _locate_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    LocatedStaticLib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    find_static_lib as _find_static_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    locate_static_lib as _locate_static_lib,
)

_T = TypeVar("_T")
_COMPATIBILITY_GUARD_RAILS_ENV_VAR = "CUDA_PATHFINDER_COMPATIBILITY_GUARD_RAILS"
_COMPATIBILITY_GUARD_RAILS_MODES = ("off", "best_effort", "strict")


class _ProcessWideGuardRailsApi(Protocol):
    def load_nvidia_dynamic_lib(self, libname: str) -> LoadedDL: ...

    def locate_nvidia_header_directory(self, libname: str) -> LocatedHeaderDir | None: ...

    def find_nvidia_header_directory(self, libname: str) -> str | None: ...

    def locate_static_lib(self, name: str) -> LocatedStaticLib: ...

    def find_static_lib(self, name: str) -> str: ...

    def locate_bitcode_lib(self, name: str) -> LocatedBitcodeLib: ...

    def find_bitcode_lib(self, name: str) -> str: ...

    def find_nvidia_binary_utility(self, utility_name: str) -> str | None: ...


class _PublicPathfinderModule(Protocol):
    process_wide_compatibility_guard_rails: object


process_wide_compatibility_guard_rails: CompatibilityGuardRails = CompatibilityGuardRails()


def _compatibility_guard_rails_mode() -> str:
    value = os.environ.get(_COMPATIBILITY_GUARD_RAILS_ENV_VAR)
    if not value:
        return "strict"
    if value in _COMPATIBILITY_GUARD_RAILS_MODES:
        return value
    allowed_values = ", ".join(repr(mode) for mode in _COMPATIBILITY_GUARD_RAILS_MODES)
    raise RuntimeError(
        f"Invalid {_COMPATIBILITY_GUARD_RAILS_ENV_VAR}={value!r}. "
        f"Allowed values: {allowed_values}. Unset or empty defaults to 'strict'."
    )


def _public_module() -> _PublicPathfinderModule | None:
    public_module = sys.modules.get("cuda.pathfinder")
    if public_module is None:
        return None
    return cast(_PublicPathfinderModule, public_module)


def _current_process_wide_compatibility_guard_rails() -> _ProcessWideGuardRailsApi:
    public_module = _public_module()
    if public_module is None:
        return cast(_ProcessWideGuardRailsApi, process_wide_compatibility_guard_rails)
    return cast(_ProcessWideGuardRailsApi, public_module.process_wide_compatibility_guard_rails)


def _reset_process_wide_compatibility_guard_rails() -> None:
    current = _current_process_wide_compatibility_guard_rails()
    if isinstance(current, CompatibilityGuardRails):
        current._reset_for_testing()
        return
    public_module = _public_module()
    if public_module is None:
        global process_wide_compatibility_guard_rails
        process_wide_compatibility_guard_rails = CompatibilityGuardRails()
        return
    public_module.process_wide_compatibility_guard_rails = CompatibilityGuardRails()


def _try_process_wide_guard_rails_then_fallback(guard_rails_call: Callable[[], _T], raw_call: Callable[[], _T]) -> _T:
    mode = _compatibility_guard_rails_mode()
    if mode == "off":
        return raw_call()
    try:
        return guard_rails_call()
    except CompatibilityInsufficientMetadataError:
        if mode == "best_effort":
            return raw_call()
        raise


def _cache_clear_with_process_state_reset(cache_clear: Callable[[], object]) -> Callable[[], None]:
    def clear() -> None:
        cache_clear()
        _reset_process_wide_compatibility_guard_rails()

    return clear


def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load a CUDA dynamic library via the process-wide compatibility guard rails."""
    return _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().load_nvidia_dynamic_lib(libname),
        lambda: _load_nvidia_dynamic_lib(libname),
    )


def locate_nvidia_header_directory(libname: str) -> LocatedHeaderDir | None:
    """Locate a CUDA header directory via the process-wide compatibility guard rails."""
    return _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().locate_nvidia_header_directory(libname),
        lambda: _locate_nvidia_header_directory(libname),
    )


def find_nvidia_header_directory(libname: str) -> str | None:
    """Locate a CUDA header directory and return its path string."""
    abs_path = _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().find_nvidia_header_directory(libname),
        lambda: _find_nvidia_header_directory_impl(libname),
    )
    assert abs_path is None or isinstance(abs_path, str)
    return abs_path


def locate_static_lib(name: str) -> LocatedStaticLib:
    """Locate a CUDA static library via the process-wide compatibility guard rails."""
    return _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().locate_static_lib(name),
        lambda: _locate_static_lib(name),
    )


def find_static_lib(name: str) -> str:
    """Locate a CUDA static library and return its path string."""
    abs_path = _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().find_static_lib(name),
        lambda: _find_static_lib(name),
    )
    assert isinstance(abs_path, str)
    return abs_path


def locate_bitcode_lib(name: str) -> LocatedBitcodeLib:
    """Locate a CUDA bitcode library via the process-wide compatibility guard rails."""
    return _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().locate_bitcode_lib(name),
        lambda: _locate_bitcode_lib(name),
    )


def find_bitcode_lib(name: str) -> str:
    """Locate a CUDA bitcode library and return its path string."""
    abs_path = _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().find_bitcode_lib(name),
        lambda: _find_bitcode_lib(name),
    )
    assert isinstance(abs_path, str)
    return abs_path


def find_nvidia_binary_utility(utility_name: str) -> str | None:
    """Locate a CUDA binary utility via the process-wide compatibility guard rails."""
    abs_path = _try_process_wide_guard_rails_then_fallback(
        lambda: _current_process_wide_compatibility_guard_rails().find_nvidia_binary_utility(utility_name),
        lambda: _find_nvidia_binary_utility(utility_name),
    )
    assert abs_path is None or isinstance(abs_path, str)
    return abs_path


load_nvidia_dynamic_lib.cache_clear = _cache_clear_with_process_state_reset(  # type: ignore[attr-defined]
    _load_nvidia_dynamic_lib.cache_clear
)
locate_nvidia_header_directory.cache_clear = _cache_clear_with_process_state_reset(  # type: ignore[attr-defined]
    _locate_nvidia_header_directory.cache_clear
)
find_nvidia_binary_utility.cache_clear = _cache_clear_with_process_state_reset(  # type: ignore[attr-defined]
    _find_nvidia_binary_utility.cache_clear
)
