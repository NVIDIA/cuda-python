# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import tempfile
import time
from pathlib import Path

from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs, find_sub_dirs_cached, find_sub_dirs_no_cache

_WILDCARD_CHILD_COUNT = 50
_SUB_DIRS = ("nvidia", "cuda_runtime", "lib")
_MISSING_SUB_DIRS = ("nvidia", "not_installed", "lib")

_TEMP_DIR = tempfile.TemporaryDirectory()
_ROOT = Path(_TEMP_DIR.name)
_PARENTS: tuple[str, ...] = tuple(str(_ROOT / f"environment-{index}") for index in range(3))

for parent in _PARENTS:
    (Path(parent) / Path(*_SUB_DIRS)).mkdir(parents=True)

_WILDCARD_PARENT = _ROOT / "wildcard-environment"
for index in range(_WILDCARD_CHILD_COUNT):
    (_WILDCARD_PARENT / "nvidia" / f"package-{index}" / "lib").mkdir(parents=True)

_WILDCARD_PARENTS = (str(_WILDCARD_PARENT),)
_WILDCARD_SUB_DIRS = ("nvidia", "*", "lib")

find_sub_dirs_cached.cache_clear()
find_sub_dirs(_PARENTS, _SUB_DIRS)


def bench_exact_hit_1_parent(loops: int) -> float:
    _fn = find_sub_dirs_no_cache
    _parents = _PARENTS[:1]
    _sub_dirs = _SUB_DIRS

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_parents, _sub_dirs)
    return time.perf_counter() - t0


def bench_exact_hit_3_parents(loops: int) -> float:
    _fn = find_sub_dirs_no_cache
    _parents = _PARENTS
    _sub_dirs = _SUB_DIRS

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_parents, _sub_dirs)
    return time.perf_counter() - t0


def bench_exact_miss_3_parents(loops: int) -> float:
    _fn = find_sub_dirs_no_cache
    _parents = _PARENTS
    _sub_dirs = _MISSING_SUB_DIRS

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_parents, _sub_dirs)
    return time.perf_counter() - t0


def bench_wildcard_hit_50_children(loops: int) -> float:
    _fn = find_sub_dirs_no_cache
    _parents = _WILDCARD_PARENTS
    _sub_dirs = _WILDCARD_SUB_DIRS

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_parents, _sub_dirs)
    return time.perf_counter() - t0


def bench_cached_exact_hit(loops: int) -> float:
    _fn = find_sub_dirs
    _parents = _PARENTS
    _sub_dirs = _SUB_DIRS
    _fn(_parents, _sub_dirs)

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_parents, _sub_dirs)
    return time.perf_counter() - t0


atexit.register(_TEMP_DIR.cleanup)
