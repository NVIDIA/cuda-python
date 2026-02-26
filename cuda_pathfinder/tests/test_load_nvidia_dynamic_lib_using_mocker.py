# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import site

import pytest

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _load_lib_no_cache,
    _resolve_system_loaded_abs_path_in_subprocess,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_FIND_MODULE = "cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib"


@pytest.fixture(autouse=True)
def _clear_canary_subprocess_probe_cache():
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    yield
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def _create_cudart_in_ctk(ctk_root):
    """Create a fake cudart lib in the platform-appropriate CTK subdirectory."""
    if IS_WINDOWS:
        cudart_dir = ctk_root / "bin" / "x64"
        cudart_dir.mkdir(parents=True, exist_ok=True)
        cudart_lib = cudart_dir / "cudart64_13.dll"
    else:
        cudart_dir = ctk_root / "lib64"
        cudart_dir.mkdir(parents=True, exist_ok=True)
        cudart_lib = cudart_dir / "libcudart.so.13"
    cudart_lib.write_bytes(b"fake")
    return cudart_lib


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


def _fake_canary_path(ctk_root):
    """Return a fake canary lib path that would resolve to the given CTK root."""
    if IS_WINDOWS:
        return str(ctk_root / "bin" / "x64" / "cudart64_13.dll")
    return str(ctk_root / "lib64" / "libcudart.so.13")


# ---------------------------------------------------------------------------
# Site-packages tests
# ---------------------------------------------------------------------------


def test_cupti_found_in_site_packages_cuda12(tmp_path, mocker, monkeypatch):
    """Test finding cupti in site-packages for CUDA 12."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create site-packages structure for CUDA 12
    site_packages = tmp_path / "site-packages"
    cupti_dir = site_packages / "nvidia" / "cuda_cupti" / "lib"
    cupti_dir.mkdir(parents=True)
    cupti_lib = cupti_dir / "libcupti.so.12"
    cupti_lib.write_bytes(b"fake")

    # Mock site-packages discovery
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "site-packages"
    assert result.abs_path == str(cupti_lib)


def test_cupti_found_in_site_packages_cuda13(tmp_path, mocker, monkeypatch):
    """Test finding cupti in site-packages for CUDA 13."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create site-packages structure for CUDA 13
    site_packages = tmp_path / "site-packages"
    cupti_dir = site_packages / "nvidia" / "cu13" / "lib"
    cupti_dir.mkdir(parents=True)
    cupti_lib = cupti_dir / "libcupti.so.13"
    cupti_lib.write_bytes(b"fake")

    # Mock site-packages discovery
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "site-packages"
    assert result.abs_path == str(cupti_lib)


# ---------------------------------------------------------------------------
# Conda tests
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
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "conda"
    assert result.abs_path == str(cupti_lib)


# ---------------------------------------------------------------------------
# CTK via CUDA_HOME tests
# ---------------------------------------------------------------------------


def test_cupti_found_via_cuda_home(tmp_path, mocker):
    """Test finding cupti via CUDA_HOME pointing to CTK installation."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    ctk_root = tmp_path / "cuda-13.1"
    cupti_lib = _create_cupti_in_ctk(ctk_root)

    # Mock CUDA_HOME discovery
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=str(ctk_root))
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "CUDA_HOME"
    # The finder resolves symlinks, so it finds libcupti.so (the symlink) not libcupti.so.13
    assert result.abs_path == str(cupti_lib.parent / "libcupti.so")


# ---------------------------------------------------------------------------
# CTK via canary probe tests
# ---------------------------------------------------------------------------


def test_cupti_found_via_canary_probe(tmp_path, mocker):
    """Test finding cupti via CTK root canary probe."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    ctk_root = tmp_path / "cuda-13.1"
    _create_cudart_in_ctk(ctk_root)
    cupti_lib = _create_cupti_in_ctk(ctk_root)

    # Mock canary probe discovery
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(ctk_root),
    )
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "system-ctk-root"
    # The finder resolves symlinks, so it finds libcupti.so (the symlink) not libcupti.so.13
    assert result.abs_path == str(cupti_lib.parent / "libcupti.so")


def test_cupti_not_found_raises_error(mocker):
    """Test that DynamicLibNotFoundError is raised when cupti is not found."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError

    # Mock all search paths to return None
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=None,
    )

    with pytest.raises(DynamicLibNotFoundError):
        _load_lib_no_cache("cupti")


# ---------------------------------------------------------------------------
# Search order tests
# ---------------------------------------------------------------------------


def test_cupti_search_order_site_packages_before_conda(tmp_path, mocker, monkeypatch):
    """Test that site-packages is searched before conda."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create both site-packages and conda structures
    site_packages = tmp_path / "site-packages"
    site_cupti_dir = site_packages / "nvidia" / "cu13" / "lib"
    site_cupti_dir.mkdir(parents=True)
    site_cupti_lib = site_cupti_dir / "libcupti.so.13"
    site_cupti_lib.write_bytes(b"fake")

    conda_prefix = tmp_path / "conda_env"
    conda_lib_dir = conda_prefix / "lib"
    conda_lib_dir.mkdir(parents=True)
    conda_cupti_lib = conda_lib_dir / "libcupti.so.13"
    conda_cupti_lib.write_bytes(b"fake")

    # Mock discovery
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "site-packages"
    assert result.abs_path == str(site_cupti_lib)


def test_cupti_search_order_conda_before_cuda_home(tmp_path, mocker, monkeypatch):
    """Test that conda is searched before CUDA_HOME."""
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

    # Mock discovery
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=str(ctk_root))
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "conda"
    assert result.abs_path == str(conda_cupti_lib)


def test_cupti_search_order_cuda_home_before_canary(tmp_path, mocker):
    """Test that CUDA_HOME is searched before canary probe."""
    if IS_WINDOWS:
        pytest.skip("Windows support for cupti not yet implemented")

    # Create two CTK roots: one for CUDA_HOME, one for canary
    cuda_home_root = tmp_path / "cuda-home"
    cuda_home_cupti = _create_cupti_in_ctk(cuda_home_root)

    canary_root = tmp_path / "cuda-system"
    _create_cudart_in_ctk(canary_root)
    _create_cupti_in_ctk(canary_root)

    # Mock discovery
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    mocker.patch(f"{_MODULE}.load_dependencies")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=str(cuda_home_root))
    canary_mock = mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(canary_root),
    )
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("cupti")
    assert result.found_via == "CUDA_HOME"
    # The finder resolves symlinks, so it finds libcupti.so (the symlink) not libcupti.so.13
    assert result.abs_path == str(cuda_home_cupti.parent / "libcupti.so")
    # Canary should not have been called
    canary_mock.assert_not_called()
