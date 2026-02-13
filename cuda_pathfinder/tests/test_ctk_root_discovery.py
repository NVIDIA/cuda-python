# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import (
    _derive_ctk_root_linux,
    _derive_ctk_root_windows,
    _FindNvidiaDynamicLib,
    derive_ctk_root,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _load_lib_no_cache,
    _try_ctk_root_canary,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_FIND_MODULE = "cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib"


# ---------------------------------------------------------------------------
# Platform-aware test helpers
# ---------------------------------------------------------------------------


def _create_nvvm_in_ctk(ctk_root):
    """Create a fake nvvm lib in the platform-appropriate CTK subdirectory."""
    if IS_WINDOWS:
        nvvm_dir = ctk_root / "nvvm" / "bin"
        nvvm_dir.mkdir(parents=True)
        nvvm_lib = nvvm_dir / "nvvm64.dll"
    else:
        nvvm_dir = ctk_root / "nvvm" / "lib64"
        nvvm_dir.mkdir(parents=True)
        nvvm_lib = nvvm_dir / "libnvvm.so"
    nvvm_lib.write_bytes(b"fake")
    return nvvm_lib


def _create_cudart_in_ctk(ctk_root):
    """Create a fake cudart lib in the platform-appropriate CTK subdirectory."""
    if IS_WINDOWS:
        lib_dir = ctk_root / "bin"
        lib_dir.mkdir(parents=True)
        lib_file = lib_dir / "cudart64_12.dll"
    else:
        lib_dir = ctk_root / "lib64"
        lib_dir.mkdir(parents=True)
        lib_file = lib_dir / "libcudart.so"
    lib_file.write_bytes(b"fake")
    return lib_file


def _fake_canary_path(ctk_root):
    """Return the path a system-loaded canary lib would resolve to."""
    if IS_WINDOWS:
        return str(ctk_root / "bin" / "cudart64_13.dll")
    return str(ctk_root / "lib64" / "libcudart.so.13")


# ---------------------------------------------------------------------------
# derive_ctk_root
# ---------------------------------------------------------------------------


def test_derive_ctk_root_linux_lib64():
    assert _derive_ctk_root_linux("/usr/local/cuda-13/lib64/libcudart.so.13") == "/usr/local/cuda-13"


def test_derive_ctk_root_linux_lib():
    assert _derive_ctk_root_linux("/opt/cuda/lib/libcudart.so.12") == "/opt/cuda"


def test_derive_ctk_root_linux_unrecognized():
    assert _derive_ctk_root_linux("/some/weird/path/libcudart.so.13") is None


def test_derive_ctk_root_linux_root_level():
    assert _derive_ctk_root_linux("/lib64/libcudart.so.13") == "/"


def test_derive_ctk_root_windows_ctk13():
    path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cudart64_13.dll"
    assert _derive_ctk_root_windows(path) == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"


def test_derive_ctk_root_windows_ctk12():
    path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll"
    assert _derive_ctk_root_windows(path) == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"


def test_derive_ctk_root_windows_unrecognized():
    assert _derive_ctk_root_windows(r"C:\weird\cudart64_13.dll") is None


def test_derive_ctk_root_windows_case_insensitive_bin():
    assert _derive_ctk_root_windows(r"C:\CUDA\Bin\cudart64_12.dll") == r"C:\CUDA"


def test_derive_ctk_root_windows_case_insensitive_x64():
    assert _derive_ctk_root_windows(r"C:\CUDA\BIN\X64\cudart64_13.dll") == r"C:\CUDA"


def test_derive_ctk_root_dispatches_to_linux(mocker):
    mocker.patch(f"{_FIND_MODULE}.IS_WINDOWS", False)
    assert derive_ctk_root("/usr/local/cuda/lib64/libcudart.so.13") == "/usr/local/cuda"


def test_derive_ctk_root_dispatches_to_windows(mocker):
    mocker.patch(f"{_FIND_MODULE}.IS_WINDOWS", True)
    assert derive_ctk_root(r"C:\CUDA\v13\bin\cudart64_13.dll") == r"C:\CUDA\v13"


# ---------------------------------------------------------------------------
# _FindNvidiaDynamicLib.try_via_ctk_root
# ---------------------------------------------------------------------------


def test_try_via_ctk_root_finds_nvvm(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    nvvm_lib = _create_nvvm_in_ctk(ctk_root)

    assert _FindNvidiaDynamicLib("nvvm").try_via_ctk_root(str(ctk_root)) == str(nvvm_lib)


def test_try_via_ctk_root_returns_none_when_dir_missing(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    ctk_root.mkdir()

    assert _FindNvidiaDynamicLib("nvvm").try_via_ctk_root(str(ctk_root)) is None


def test_try_via_ctk_root_regular_lib(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    cudart_lib = _create_cudart_in_ctk(ctk_root)

    assert _FindNvidiaDynamicLib("cudart").try_via_ctk_root(str(ctk_root)) == str(cudart_lib)


# ---------------------------------------------------------------------------
# _try_ctk_root_canary
# ---------------------------------------------------------------------------


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def test_canary_finds_nvvm(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    _create_cudart_in_ctk(ctk_root)
    nvvm_lib = _create_nvvm_in_ctk(ctk_root)

    probe = mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(ctk_root),
    )
    parent_system_loader = mocker.patch(f"{_MODULE}.load_with_system_search")

    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) == str(nvvm_lib)
    probe.assert_called_once_with("cudart")
    parent_system_loader.assert_not_called()


def test_canary_returns_none_when_subprocess_probe_fails(mocker):
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_returns_none_when_ctk_root_unrecognized(mocker):
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value="/weird/path/libcudart.so.13",
    )
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_returns_none_when_nvvm_not_in_ctk_root(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    # Create only the canary lib dir, not nvvm
    _create_cudart_in_ctk(ctk_root)

    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(ctk_root),
    )
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_skips_when_abs_path_none(mocker):
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", return_value=None)
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


# ---------------------------------------------------------------------------
# _load_lib_no_cache search-order
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolate_load_cascade(mocker):
    """Disable the search steps that run before system-search in _load_lib_no_cache.

    This lets the ordering tests focus on system-search, CUDA_HOME, and the
    canary probe without needing a real site-packages or conda environment.
    """
    # No wheels installed
    mocker.patch.object(_FindNvidiaDynamicLib, "try_site_packages", return_value=None)
    # No conda env
    mocker.patch.object(_FindNvidiaDynamicLib, "try_with_conda_prefix", return_value=None)
    # Lib not already loaded by another component
    mocker.patch(f"{_MODULE}.check_if_already_loaded_from_elsewhere", return_value=None)
    # Skip transitive dependency loading
    mocker.patch(f"{_MODULE}.load_dependencies")


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_cuda_home_takes_priority_over_canary(tmp_path, mocker):
    # Two competing CTK roots: one from CUDA_HOME, one the canary would find.
    cuda_home_root = tmp_path / "cuda-home"
    nvvm_home_lib = _create_nvvm_in_ctk(cuda_home_root)

    canary_root = tmp_path / "cuda-system"
    _create_cudart_in_ctk(canary_root)
    _create_nvvm_in_ctk(canary_root)

    canary_mock = mocker.MagicMock(return_value=_fake_canary_path(canary_root))

    # System search finds nothing for nvvm.
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    # Canary subprocess probe would find cudart if consulted.
    mocker.patch(f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess", side_effect=canary_mock)
    # CUDA_HOME points to a separate root that also has nvvm
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=str(cuda_home_root))
    # Capture the final load call
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("nvvm")

    # CUDA_HOME must win; the canary should never have been consulted
    assert result.found_via == "CUDA_HOME"
    assert result.abs_path == str(nvvm_home_lib)
    canary_mock.assert_not_called()


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_canary_fires_only_after_all_earlier_steps_fail(tmp_path, mocker):
    canary_root = tmp_path / "cuda-system"
    _create_cudart_in_ctk(canary_root)
    nvvm_lib = _create_nvvm_in_ctk(canary_root)

    # System search: nvvm not on linker path.
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    # Canary subprocess probe finds cudart under a system CTK root.
    mocker.patch(
        f"{_MODULE}._resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_canary_path(canary_root),
    )
    # No CUDA_HOME set
    mocker.patch(f"{_FIND_MODULE}.get_cuda_home_or_path", return_value=None)
    # Capture the final load call
    mocker.patch(
        f"{_MODULE}.load_with_abs_path",
        side_effect=lambda _libname, path, via: _make_loaded_dl(path, via),
    )

    result = _load_lib_no_cache("nvvm")

    assert result.found_via == "system-ctk-root"
    assert result.abs_path == str(nvvm_lib)
