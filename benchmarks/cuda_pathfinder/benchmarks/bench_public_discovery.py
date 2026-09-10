# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import site
import tempfile
import time
from pathlib import Path

from cuda.pathfinder import find_nvidia_binary_utility, find_nvidia_header_directory, locate_static_lib
from cuda.pathfinder._headers.find_nvidia_headers import locate_nvidia_header_directory
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_cached
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_TEMP_DIR = tempfile.TemporaryDirectory()
_ROOT = Path(_TEMP_DIR.name)
_SITE_ROOTS = tuple(_ROOT / f"site-root-{index}" for index in range(3))
for root in _SITE_ROOTS:
    root.mkdir()

_PACKAGE_ROOT = _SITE_ROOTS[-1] / "nvidia"
_HEADER_DIR = _PACKAGE_ROOT / "cuda_runtime" / "include"
_HEADER_DIR.mkdir(parents=True)
(_HEADER_DIR / "cuda_runtime.h").touch()

_BINARY_DIR = _PACKAGE_ROOT / "cuda_nvcc" / "bin"
_BINARY_DIR.mkdir(parents=True)
_BINARY_NAME = "nvdisasm.exe" if IS_WINDOWS else "nvdisasm"
_BINARY_PATH = _BINARY_DIR / _BINARY_NAME
_BINARY_PATH.touch()
if not IS_WINDOWS:
    _BINARY_PATH.chmod(0o755)

_STATIC_LIB_DIR = _PACKAGE_ROOT / "cuda_runtime" / "lib"
if IS_WINDOWS:
    _STATIC_LIB_DIR /= "x64"
_STATIC_LIB_DIR.mkdir(parents=True)
_STATIC_LIB_NAME = "cudadevrt.lib" if IS_WINDOWS else "libcudadevrt.a"
(_STATIC_LIB_DIR / _STATIC_LIB_NAME).touch()

_ORIGINAL_GETSITEPACKAGES = site.getsitepackages
_ORIGINAL_ENABLE_USER_SITE = site.ENABLE_USER_SITE
_ACTIVE_SITE_ROOTS: tuple[Path, ...] = _SITE_ROOTS


def _getsitepackages() -> list[str]:
    return [str(root) for root in _ACTIVE_SITE_ROOTS]


site.getsitepackages = _getsitepackages
site.ENABLE_USER_SITE = False


def _clear_discovery_caches() -> None:
    find_sub_dirs_cached.cache_clear()
    find_nvidia_binary_utility.cache_clear()
    locate_nvidia_header_directory.cache_clear()


def _cleanup() -> None:
    site.getsitepackages = _ORIGINAL_GETSITEPACKAGES
    site.ENABLE_USER_SITE = _ORIGINAL_ENABLE_USER_SITE
    _TEMP_DIR.cleanup()


def bench_header_cold_1_root(loops: int) -> float:
    global _ACTIVE_SITE_ROOTS
    _ACTIVE_SITE_ROOTS = _SITE_ROOTS[-1:]
    _fn = find_nvidia_header_directory
    _clear = _clear_discovery_caches

    t0 = time.perf_counter()
    for _ in range(loops):
        _clear()
        _fn("cudart")
    return time.perf_counter() - t0


def bench_header_cold_3_roots(loops: int) -> float:
    global _ACTIVE_SITE_ROOTS
    _ACTIVE_SITE_ROOTS = _SITE_ROOTS
    _fn = find_nvidia_header_directory
    _clear = _clear_discovery_caches

    t0 = time.perf_counter()
    for _ in range(loops):
        _clear()
        _fn("cudart")
    return time.perf_counter() - t0


def bench_binary_cold_1_root(loops: int) -> float:
    global _ACTIVE_SITE_ROOTS
    _ACTIVE_SITE_ROOTS = _SITE_ROOTS[-1:]
    _fn = find_nvidia_binary_utility
    _clear = _clear_discovery_caches

    t0 = time.perf_counter()
    for _ in range(loops):
        _clear()
        _fn("nvdisasm")
    return time.perf_counter() - t0


def bench_binary_cold_3_roots(loops: int) -> float:
    global _ACTIVE_SITE_ROOTS
    _ACTIVE_SITE_ROOTS = _SITE_ROOTS
    _fn = find_nvidia_binary_utility
    _clear = _clear_discovery_caches

    t0 = time.perf_counter()
    for _ in range(loops):
        _clear()
        _fn("nvdisasm")
    return time.perf_counter() - t0


def bench_static_lib_cold_3_roots(loops: int) -> float:
    global _ACTIVE_SITE_ROOTS
    _ACTIVE_SITE_ROOTS = _SITE_ROOTS
    _fn = locate_static_lib
    _clear = find_sub_dirs_cached.cache_clear

    t0 = time.perf_counter()
    for _ in range(loops):
        _clear()
        _fn("cudadevrt")
    return time.perf_counter() - t0


_clear_discovery_caches()
assert find_nvidia_header_directory("cudart") == os.path.normpath(os.path.abspath(_HEADER_DIR))
assert find_nvidia_binary_utility("nvdisasm") == os.path.abspath(_BINARY_PATH)
assert locate_static_lib("cudadevrt").abs_path == os.path.join(os.path.abspath(_STATIC_LIB_DIR), _STATIC_LIB_NAME)
_clear_discovery_caches()

atexit.register(_cleanup)
