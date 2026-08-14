# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 wrapper for cuda.core's scikit-build-core backend.

The wrapper keeps CUDA-major-dependent build requirements out of the static
``build-system.requires`` list. Compilation and wheel assembly are delegated to
scikit-build-core.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import scikit_build_core.build as _build_backend

build_sdist = _build_backend.build_sdist
get_requires_for_build_sdist = _build_backend.get_requires_for_build_sdist

_ConfigSettings = Mapping[str, str | list[str] | bool] | None
_MISSING = object()
_CUDA_ROOT_SETTING = "cmake.define.CUDA_CORE_CUDA_ROOT"
_CUDA_ROOT_SETTING_ALIAS = "skbuild.cmake.define.CUDA_CORE_CUDA_ROOT"
_CUDA_MAJOR_SETTING = "cmake.define.CUDA_CORE_BUILD_MAJOR"
_CUDA_MAJOR_SETTING_ALIAS = "skbuild.cmake.define.CUDA_CORE_BUILD_MAJOR"
_BUILD_TYPE_SETTINGS = ("cmake.build-type", "skbuild.cmake.build-type")


def _get_cuda_path() -> str:
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        return cuda_path
    raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")


def _cuda_major_from_headers(cuda_path: str) -> str:
    cuda_h = os.path.join(cuda_path, "include", "cuda.h")
    try:
        with open(cuda_h, encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                if match:
                    # CUDA_VERSION is e.g. 12020 for 12.2.
                    return str(int(match.group(1)) // 1000)
    except OSError:
        pass

    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Set CUDA_CORE_BUILD_MAJOR, or ensure CUDA_PATH or CUDA_HOME points "
        "to a valid CUDA installation with include/cuda.h."
    )


def _determine_cuda_major_version(cuda_path: str | None = None) -> str:
    """Determine the CUDA major used for build requirements and Cython."""
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is None:
        cuda_major = _cuda_major_from_headers(cuda_path or _get_cuda_path())

    if not re.fullmatch(r"\d+", cuda_major):
        raise RuntimeError(f"CUDA_CORE_BUILD_MAJOR must be an integer, got {cuda_major!r}")

    print("CUDA MAJOR VERSION:", cuda_major)
    return cuda_major


def _setting_value(config_settings: _ConfigSettings, *names: str) -> str | bool | None:
    if config_settings is None:
        return None
    for name in names:
        value = config_settings.get(name)
        if isinstance(value, list):
            if not value:
                continue
            value = value[-1]
        if value is not None:
            return value
    return None


def _configured_cuda_major(config_settings: _ConfigSettings) -> str:
    # Match scikit-build-core's precedence when both accepted spellings are
    # supplied: the explicitly prefixed spelling wins.
    cuda_major = _setting_value(config_settings, _CUDA_MAJOR_SETTING_ALIAS, _CUDA_MAJOR_SETTING)
    if cuda_major is None:
        configured_root = _setting_value(config_settings, _CUDA_ROOT_SETTING_ALIAS, _CUDA_ROOT_SETTING)
        return _determine_cuda_major_version(os.fspath(configured_root) if configured_root is not None else None)

    cuda_major = str(cuda_major)
    if not re.fullmatch(r"\d+", cuda_major):
        raise RuntimeError(f"CUDA_CORE_BUILD_MAJOR must be an integer, got {cuda_major!r}")
    return cuda_major


def _parse_bool(value: str | bool, *, setting: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n"}:
        return False
    raise ValueError(f"{setting} must be a boolean value, got {value!r}")


def _translate_config_settings(config_settings: _ConfigSettings) -> dict[str, Any]:
    settings = dict(config_settings or {})
    debug = settings.pop("debug", _MISSING)

    if debug is not _MISSING and any(key in settings for key in _BUILD_TYPE_SETTINGS):
        raise ValueError("debug and cmake.build-type cannot both be specified")

    if debug is not _MISSING:
        if isinstance(debug, list):
            if not debug:
                raise ValueError("debug must have a value")
            debug = debug[-1]
        debug_enabled = _parse_bool(debug, setting="debug")
        if debug_enabled and sys.platform == "win32":
            raise RuntimeError("Debuggable builds are not supported on Windows.")
        settings["cmake.build-type"] = "Debug" if debug_enabled else "Release"

    return settings


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.prepare_metadata_for_build_wheel(metadata_directory, settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.prepare_metadata_for_build_editable(metadata_directory, settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.build_wheel(wheel_directory, settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.build_editable(wheel_directory, settings, metadata_directory)


def _get_cuda_bindings_require(config_settings: _ConfigSettings = None) -> list[str]:
    cuda_major = _configured_cuda_major(config_settings)
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_wheel(config_settings=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.get_requires_for_build_wheel(settings) + _get_cuda_bindings_require(settings)


def get_requires_for_build_editable(config_settings=None):
    settings = _translate_config_settings(config_settings)
    return _build_backend.get_requires_for_build_editable(settings) + _get_cuda_bindings_require(settings)
