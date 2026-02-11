# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composable search steps and cascade runner."""

from __future__ import annotations

import os

import pytest

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.search_steps import (
    EARLY_FIND_STEPS,
    LATE_FIND_STEPS,
    FindResult,
    SearchContext,
    find_in_conda,
    find_in_cuda_home,
    find_in_site_packages,
    run_find_steps,
)

_MOD = "cuda.pathfinder._dynamic_libs.search_steps"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_desc(name: str = "cudart", **overrides) -> LibDescriptor:
    defaults = dict(
        name=name,
        strategy="ctk",
        linux_sonames=("libcudart.so",),
        windows_dlls=("cudart64_12.dll",),
        site_packages_linux=(os.path.join("nvidia", "cuda_runtime", "lib"),),
        site_packages_windows=(os.path.join("nvidia", "cuda_runtime", "bin"),),
    )
    defaults.update(overrides)
    return LibDescriptor(**defaults)


def _ctx(desc: LibDescriptor | None = None) -> SearchContext:
    return SearchContext(desc or _make_desc())


# ---------------------------------------------------------------------------
# SearchContext
# ---------------------------------------------------------------------------


class TestSearchContext:
    def test_libname_delegates_to_descriptor(self):
        ctx = _ctx(_make_desc(name="nvrtc"))
        assert ctx.libname == "nvrtc"

    def test_lib_searched_for_linux(self, mocker):
        mocker.patch(f"{_MOD}.IS_WINDOWS", False)
        ctx = _ctx(_make_desc(name="cublas"))
        assert ctx.lib_searched_for == "libcublas.so"

    def test_lib_searched_for_windows(self, mocker):
        mocker.patch(f"{_MOD}.IS_WINDOWS", True)
        ctx = _ctx(_make_desc(name="cublas"))
        assert ctx.lib_searched_for == "cublas*.dll"

    def test_raise_not_found_includes_messages(self):
        ctx = _ctx()
        ctx.error_messages.append("No such file: libcudart.so*")
        ctx.attachments.append('  listdir("/some/dir"):')
        with pytest.raises(DynamicLibNotFoundError, match="No such file"):
            ctx.raise_not_found()

    def test_raise_not_found_empty_messages(self):
        ctx = _ctx()
        with pytest.raises(DynamicLibNotFoundError):
            ctx.raise_not_found()


# ---------------------------------------------------------------------------
# find_in_site_packages
# ---------------------------------------------------------------------------


class TestFindInSitePackages:
    def test_returns_none_when_no_rel_dirs(self):
        desc = _make_desc(site_packages_linux=(), site_packages_windows=())
        result = find_in_site_packages(_ctx(desc))
        assert result is None

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        lib_dir.mkdir(parents=True)
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", False)
        mocker.patch(
            f"{_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(lib_dir)],
        )

        desc = _make_desc(
            site_packages_linux=(os.path.join("nvidia", "cuda_runtime", "lib"),),
        )
        result = find_in_site_packages(_ctx(desc))
        assert result is not None
        assert result.abs_path == str(so_file)
        assert result.found_via == "site-packages"

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
        bin_dir.mkdir(parents=True)
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", True)
        mocker.patch(
            f"{_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(bin_dir)],
        )
        mocker.patch(f"{_MOD}.is_suppressed_dll_file", return_value=False)

        desc = _make_desc(
            name="cudart",
            site_packages_windows=(os.path.join("nvidia", "cuda_runtime", "bin"),),
        )
        result = find_in_site_packages(_ctx(desc))
        assert result is not None
        assert result.abs_path == str(dll)
        assert result.found_via == "site-packages"

    def test_not_found_appends_error(self, mocker, tmp_path):
        empty_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        empty_dir.mkdir(parents=True)

        mocker.patch(f"{_MOD}.IS_WINDOWS", False)
        mocker.patch(
            f"{_MOD}.find_sub_dirs_all_sitepackages",
            return_value=[str(empty_dir)],
        )

        ctx = _ctx()
        result = find_in_site_packages(ctx)
        assert result is None
        assert any("No such file" in m for m in ctx.error_messages)


# ---------------------------------------------------------------------------
# find_in_conda
# ---------------------------------------------------------------------------


class TestFindInConda:
    def test_returns_none_without_conda_prefix(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        assert find_in_conda(_ctx()) is None

    def test_returns_none_with_empty_conda_prefix(self, mocker):
        mocker.patch.dict(os.environ, {"CONDA_PREFIX": ""})
        assert find_in_conda(_ctx()) is None

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", False)
        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})

        result = find_in_conda(_ctx())
        assert result is not None
        assert result.abs_path == str(so_file)
        assert result.found_via == "conda"

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "Library" / "bin"
        bin_dir.mkdir(parents=True)
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", True)
        mocker.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)})
        mocker.patch(f"{_MOD}.is_suppressed_dll_file", return_value=False)

        result = find_in_conda(_ctx())
        assert result is not None
        assert result.abs_path == str(dll)
        assert result.found_via == "conda"


# ---------------------------------------------------------------------------
# find_in_cuda_home
# ---------------------------------------------------------------------------


class TestFindInCudaHome:
    def test_returns_none_without_env_var(self, mocker):
        mocker.patch(f"{_MOD}.get_cuda_home_or_path", return_value=None)
        assert find_in_cuda_home(_ctx()) is None

    def test_found_linux(self, mocker, tmp_path):
        lib_dir = tmp_path / "lib64"
        lib_dir.mkdir()
        so_file = lib_dir / "libcudart.so"
        so_file.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", False)
        mocker.patch(f"{_MOD}.get_cuda_home_or_path", return_value=str(tmp_path))

        result = find_in_cuda_home(_ctx())
        assert result is not None
        assert result.abs_path == str(so_file)
        assert result.found_via == "CUDA_HOME"

    def test_found_windows(self, mocker, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        dll = bin_dir / "cudart64_12.dll"
        dll.touch()

        mocker.patch(f"{_MOD}.IS_WINDOWS", True)
        mocker.patch(f"{_MOD}.get_cuda_home_or_path", return_value=str(tmp_path))
        mocker.patch(f"{_MOD}.is_suppressed_dll_file", return_value=False)

        result = find_in_cuda_home(_ctx())
        assert result is not None
        assert result.abs_path == str(dll)
        assert result.found_via == "CUDA_HOME"


# ---------------------------------------------------------------------------
# run_find_steps
# ---------------------------------------------------------------------------


class TestRunFindSteps:
    def test_returns_first_hit(self):
        hit = FindResult("/path/to/lib.so", "step-a")

        def step_a(ctx):
            return hit

        def step_b(ctx):
            raise AssertionError("step_b should not be called")

        result = run_find_steps(_ctx(), (step_a, step_b))
        assert result is hit

    def test_returns_none_when_all_miss(self):
        result = run_find_steps(_ctx(), (lambda ctx: None, lambda ctx: None))
        assert result is None

    def test_empty_steps(self):
        assert run_find_steps(_ctx(), ()) is None

    def test_skips_nones_returns_later_hit(self):
        hit = FindResult("/later/lib.so", "step-c")
        result = run_find_steps(_ctx(), (lambda ctx: None, lambda ctx: hit))
        assert result is hit


# ---------------------------------------------------------------------------
# Step tuple sanity checks
# ---------------------------------------------------------------------------


class TestStepTuples:
    def test_early_find_steps_contains_expected(self):
        assert find_in_site_packages in EARLY_FIND_STEPS
        assert find_in_conda in EARLY_FIND_STEPS

    def test_late_find_steps_contains_expected(self):
        assert find_in_cuda_home in LATE_FIND_STEPS

    def test_early_and_late_are_disjoint(self):
        assert not set(EARLY_FIND_STEPS) & set(LATE_FIND_STEPS)
