# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_mod
from cuda.pathfinder._dynamic_libs import search_steps as steps_mod
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _load_lib_no_cache,
    _resolve_system_loaded_abs_path_in_subprocess,
)
from cuda.pathfinder._dynamic_libs.search_steps import EARLY_FIND_STEPS
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_STEPS_MODULE = "cuda.pathfinder._dynamic_libs.search_steps"


@pytest.fixture(autouse=True)
def _clear_canary_subprocess_probe_cache():
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    yield
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def _create_cupti_in_ctk(ctk_root):
    """Create a fake cupti lib in extras/CUPTI/lib64."""
    if IS_WINDOWS:
        cupti_dir = ctk_root / "extras" / "CUPTI" / "lib64"
        cupti_dir.mkdir(parents=True, exist_ok=True)
        cupti_lib = cupti_dir / "cupti64_2025.4.1.dll"
    else:
        cupti_dir = ctk_root / "extras" / "CUPTI" / "lib64"
        cupti_dir.mkdir(parents=True, exist_ok=True)
        cupti_lib = cupti_dir / "libcupti.so.13"
        # Create symlink like real CTK installations
        cupti_symlink = cupti_dir / "libcupti.so"
        cupti_symlink.symlink_to("libcupti.so.13")
    cupti_lib.write_bytes(b"fake")
    return cupti_lib


# ---------------------------------------------------------------------------
# Conda tests
# Note: Site-packages and CTK are covered by real CI tests.
# Mock tests focus on Conda (not covered by real CI) and error paths.
# ---------------------------------------------------------------------------


def test_cupti_found_in_conda(tmp_path, mocker, monkeypatch):
    """Test finding cupti in conda environment."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create conda structure
    conda_prefix = tmp_path / "conda_env"
    conda_lib_dir = conda_prefix / "lib"
    conda_lib_dir.mkdir(parents=True)
    cupti_lib = conda_lib_dir / "libcupti.so.13"
    cupti_lib.write_bytes(b"fake")

    # Mock conda discovery
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))

    # Disable site-packages search
    def _run_find_steps_without_site_packages(ctx, steps):
        if steps is EARLY_FIND_STEPS:
            # Skip site-packages, only run conda
            from cuda.pathfinder._dynamic_libs.search_steps import find_in_conda

            result = find_in_conda(ctx)
            return result
        return steps_mod.run_find_steps(ctx, steps)

    mocker.patch(f"{_MODULE}.run_find_steps", side_effect=_run_find_steps_without_site_packages)
    mocker.patch.object(load_mod.LOADER, "check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch.object(
        load_mod.LOADER,
        "load_with_abs_path",
        side_effect=lambda _desc, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "conda"
    assert result.abs_path == str(cupti_lib)


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


def test_cupti_not_found_raises_error(mocker):
    """Test that DynamicLibNotFoundError is raised when cupti is not found."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Mock all search paths to return None
    def _run_find_steps_disabled(ctx, steps):
        return None

    mocker.patch(f"{_MODULE}.run_find_steps", side_effect=_run_find_steps_disabled)
    mocker.patch.object(load_mod.LOADER, "check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=None,
    )

    with pytest.raises(DynamicLibNotFoundError):
        _load_lib_no_cache("cupti")


# ---------------------------------------------------------------------------
# Search order tests (Conda-specific, since Conda is not covered by real CI)
# ---------------------------------------------------------------------------


def test_cupti_search_order_conda_before_cuda_home(tmp_path, mocker, monkeypatch):
    """Test that conda is searched before CUDA_HOME (CTK).

    This test is important because Conda is not covered by real CI tests,
    so we need to verify the search order between Conda and CTK.
    """
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create both conda and CUDA_HOME structures
    conda_prefix = tmp_path / "conda_env"
    conda_lib_dir = conda_prefix / "lib"
    conda_lib_dir.mkdir(parents=True)
    conda_cupti_lib = conda_lib_dir / "libcupti.so.13"
    conda_cupti_lib.write_bytes(b"fake")

    ctk_root = tmp_path / "cuda-13.1"
    _create_cupti_in_ctk(ctk_root)

    # Mock discovery - disable site-packages, enable conda
    def _run_find_steps_without_site_packages(ctx, steps):
        if steps is EARLY_FIND_STEPS:
            # Skip site-packages, only run conda
            from cuda.pathfinder._dynamic_libs.search_steps import find_in_conda

            result = find_in_conda(ctx)
            return result
        return steps_mod.run_find_steps(ctx, steps)

    mocker.patch(f"{_MODULE}.run_find_steps", side_effect=_run_find_steps_without_site_packages)
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    mocker.patch.object(load_mod.LOADER, "check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch.object(load_mod.LOADER, "load_with_system_search", return_value=None)
    mocker.patch(f"{_STEPS_MODULE}.get_cuda_home_or_path", return_value=str(ctk_root))
    mocker.patch.object(
        load_mod.LOADER,
        "load_with_abs_path",
        side_effect=lambda _desc, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "conda"
    assert result.abs_path == str(conda_cupti_lib)
