# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from local_helpers import (
    have_distribution,
    locate_real_cuda_toolkit_version_from_cuda_h,
)

import cuda.pathfinder._compatibility_guard_rails as compatibility_module
from cuda.pathfinder import LoadedDL, LocatedBitcodeLib, LocatedStaticLib
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _resolve_system_loaded_abs_path_in_subprocess
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as locate_nvidia_header_directory_raw,
)
from cuda.pathfinder._utils import driver_info
from cuda.pathfinder._utils.driver_info import DriverCudaVersion, DriverReleaseVersion
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.toolkit_info import read_cuda_header_version

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_COMPATIBILITY_GUARD_RAILS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")
COMPATIBILITY_GUARD_RAILS_ENV_VAR = "CUDA_PATHFINDER_COMPATIBILITY_GUARD_RAILS"
DRIVER_COMPATIBILITY_ENV_VAR = "CUDA_PATHFINDER_DRIVER_COMPATIBILITY"
process_wide_module = importlib.import_module("cuda.pathfinder._process_wide_compatibility_guard_rails")


@pytest.fixture(autouse=True)
def _default_process_wide_guard_rails_mode(monkeypatch):
    monkeypatch.delenv(COMPATIBILITY_GUARD_RAILS_ENV_VAR, raising=False)
    monkeypatch.delenv(DRIVER_COMPATIBILITY_ENV_VAR, raising=False)


@pytest.fixture
def clear_real_host_probe_caches():
    have_distribution.cache_clear()
    locate_real_cuda_toolkit_version_from_cuda_h.cache_clear()
    locate_nvidia_header_directory_raw.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    get_cuda_path_or_home.cache_clear()
    read_cuda_header_version.cache_clear()
    driver_info._load_nvidia_dynamic_lib.cache_clear()
    driver_info.query_driver_cuda_version.cache_clear()
    driver_info.query_driver_release_version.cache_clear()
    yield
    have_distribution.cache_clear()
    locate_real_cuda_toolkit_version_from_cuda_h.cache_clear()
    locate_nvidia_header_directory_raw.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    get_cuda_path_or_home.cache_clear()
    read_cuda_header_version.cache_clear()
    driver_info._load_nvidia_dynamic_lib.cache_clear()
    driver_info.query_driver_cuda_version.cache_clear()
    driver_info.query_driver_release_version.cache_clear()


def _write_cuda_h(
    ctk_root: Path,
    toolkit_version: str,
    *,
    include_dir_parts: tuple[str, ...] = ("targets", "x86_64-linux", "include"),
) -> None:
    parts = toolkit_version.split(".")
    if len(parts) < 2:
        raise AssertionError(f"Expected at least major.minor in toolkit version, got {toolkit_version!r}")
    encoded = int(parts[0]) * 1000 + int(parts[1]) * 10
    cuda_h_path = ctk_root.joinpath(*include_dir_parts, "cuda.h")
    cuda_h_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_h_path.write_text(
        f"#ifndef CUDA_H\n#define CUDA_H\n#define CUDA_VERSION {encoded}\n#endif\n",
        encoding="utf-8",
    )


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def _loaded_dl(abs_path: str, *, found_via: str = "CUDA_PATH") -> LoadedDL:
    return LoadedDL(
        abs_path=abs_path,
        was_already_loaded_from_elsewhere=False,
        _handle_uint=1,
        found_via=found_via,
    )


def _patch_dynamic_lib_loader(monkeypatch, **loaded_by_libname: LoadedDL) -> None:
    def fake_load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
        loaded = loaded_by_libname.get(libname)
        if loaded is None:
            raise AssertionError(f"Unexpected libname: {libname!r}")
        return loaded

    monkeypatch.setattr(compatibility_module, "_load_nvidia_dynamic_lib", fake_load_nvidia_dynamic_lib)


def _located_static_lib(name: str, abs_path: str) -> LocatedStaticLib:
    return LocatedStaticLib(
        name=name,
        abs_path=abs_path,
        filename=os.path.basename(abs_path),
        found_via="CUDA_PATH",
    )


def _located_bitcode_lib(name: str, abs_path: str) -> LocatedBitcodeLib:
    return LocatedBitcodeLib(
        name=name,
        abs_path=abs_path,
        filename=os.path.basename(abs_path),
        found_via="CUDA_PATH",
    )


def _driver_cuda_version(encoded: int) -> DriverCudaVersion:
    return DriverCudaVersion.from_encoded(encoded)


def _driver_release_version(text: str) -> DriverReleaseVersion:
    return DriverReleaseVersion.from_text(text)


class _FakeDistribution:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        root: Path,
        files: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.files = tuple(Path(file) for file in files)
        self.requires = list(requires)
        self._root = root

    def locate_file(self, file: Path) -> Path:
        return self._root / file


def _assert_real_ctk_backed_path(path: str) -> None:
    norm_path = os.path.normpath(os.path.abspath(path))
    if "site-packages" in Path(norm_path).parts:
        return
    current = Path(norm_path)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "include" / "cuda.h").is_file():
            return
        if any(path.is_file() for path in (candidate / "targets").glob("*/include/cuda.h")):
            return
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        ctk_root = os.environ.get(env_var)
        if not ctk_root:
            continue
        norm_ctk_root = os.path.normpath(os.path.abspath(ctk_root))
        if os.path.commonpath((norm_path, norm_ctk_root)) == norm_ctk_root:
            return
    raise AssertionError(
        "Expected a site-packages path, a path under a CTK root with cuda.h, "
        f"or a path under CUDA_PATH/CUDA_HOME, got {path!r}"
    )


class _DelegatingProcessWideGuardRails:
    def __init__(self, method_name: str, return_value: object) -> None:
        self._method_name = method_name
        self._return_value = return_value
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str):
        if name != self._method_name:
            raise AttributeError(name)

        def delegated(*args: object) -> object:
            self.calls.append((name, args))
            return self._return_value

        return delegated
