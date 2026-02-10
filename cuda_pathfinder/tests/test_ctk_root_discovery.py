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

_MODULE = "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib"
_FIND_MODULE = "cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib"


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
    nvvm_dir = ctk_root / "nvvm" / "lib64"
    nvvm_dir.mkdir(parents=True)
    nvvm_so = nvvm_dir / "libnvvm.so"
    nvvm_so.write_bytes(b"fake")

    assert _FindNvidiaDynamicLib("nvvm").try_via_ctk_root(str(ctk_root)) == str(nvvm_so)


def test_try_via_ctk_root_returns_none_when_dir_missing(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    ctk_root.mkdir()

    assert _FindNvidiaDynamicLib("nvvm").try_via_ctk_root(str(ctk_root)) is None


def test_try_via_ctk_root_regular_lib(tmp_path):
    ctk_root = tmp_path / "cuda-13"
    lib_dir = ctk_root / "lib64"
    lib_dir.mkdir(parents=True)
    cudart_so = lib_dir / "libcudart.so"
    cudart_so.write_bytes(b"fake")

    assert _FindNvidiaDynamicLib("cudart").try_via_ctk_root(str(ctk_root)) == str(cudart_so)


# ---------------------------------------------------------------------------
# _try_ctk_root_canary
# ---------------------------------------------------------------------------


def _make_loaded_dl(path, found_via):
    return LoadedDL(path, False, 0xDEAD, found_via)


def test_canary_finds_nvvm(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    (ctk_root / "lib64").mkdir(parents=True)
    nvvm_dir = ctk_root / "nvvm" / "lib64"
    nvvm_dir.mkdir(parents=True)
    nvvm_so = nvvm_dir / "libnvvm.so"
    nvvm_so.write_bytes(b"fake")

    canary = _make_loaded_dl(str(ctk_root / "lib64" / "libcudart.so.13"), "system-search")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=canary)

    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) == str(nvvm_so)


def test_canary_returns_none_when_system_search_fails(mocker):
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=None)
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_returns_none_when_ctk_root_unrecognized(mocker):
    canary = _make_loaded_dl("/weird/path/libcudart.so.13", "system-search")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=canary)
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_returns_none_when_nvvm_not_in_ctk_root(tmp_path, mocker):
    ctk_root = tmp_path / "cuda-13"
    (ctk_root / "lib64").mkdir(parents=True)

    canary = _make_loaded_dl(str(ctk_root / "lib64" / "libcudart.so.13"), "system-search")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=canary)
    assert _try_ctk_root_canary(_FindNvidiaDynamicLib("nvvm")) is None


def test_canary_skips_when_abs_path_none(mocker):
    canary = _make_loaded_dl(None, "system-search")
    mocker.patch(f"{_MODULE}.load_with_system_search", return_value=canary)
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
    nvvm_home = cuda_home_root / "nvvm" / "lib64"
    nvvm_home.mkdir(parents=True)
    nvvm_home_so = nvvm_home / "libnvvm.so"
    nvvm_home_so.write_bytes(b"home")

    canary_root = tmp_path / "cuda-system"
    (canary_root / "lib64").mkdir(parents=True)
    nvvm_canary = canary_root / "nvvm" / "lib64"
    nvvm_canary.mkdir(parents=True)
    (nvvm_canary / "libnvvm.so").write_bytes(b"canary")

    canary_mock = mocker.MagicMock(
        return_value=_make_loaded_dl(str(canary_root / "lib64" / "libcudart.so.13"), "system-search")
    )

    # System search finds nothing for nvvm; canary would find cudart
    mocker.patch(
        f"{_MODULE}.load_with_system_search",
        side_effect=lambda name: None if name == "nvvm" else canary_mock(name),
    )
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
    assert result.abs_path == str(nvvm_home_so)
    canary_mock.assert_not_called()


@pytest.mark.usefixtures("_isolate_load_cascade")
def test_canary_fires_only_after_all_earlier_steps_fail(tmp_path, mocker):
    canary_root = tmp_path / "cuda-system"
    (canary_root / "lib64").mkdir(parents=True)
    nvvm_dir = canary_root / "nvvm" / "lib64"
    nvvm_dir.mkdir(parents=True)
    nvvm_so = nvvm_dir / "libnvvm.so"
    nvvm_so.write_bytes(b"canary")

    canary_result = _make_loaded_dl(str(canary_root / "lib64" / "libcudart.so.13"), "system-search")

    # System search: nvvm not on linker path, but cudart (canary) is
    mocker.patch(
        f"{_MODULE}.load_with_system_search",
        side_effect=lambda name: canary_result if name == "cudart" else None,
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
    assert result.abs_path == str(nvvm_so)
