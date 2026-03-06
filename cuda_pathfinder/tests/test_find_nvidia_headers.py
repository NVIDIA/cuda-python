# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Currently these installations are only manually tested:

# ../toolshed/conda_create_for_pathfinder_testing.*

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
# sudo apt install libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13

import functools
import glob
import importlib.metadata
import os
import re
from pathlib import Path

import pytest

import cuda.pathfinder._headers.find_nvidia_headers as find_nvidia_headers_module
from cuda.pathfinder import LocatedHeaderDir, find_nvidia_header_directory, locate_nvidia_header_directory
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _resolve_system_loaded_abs_path_in_subprocess,
)
from cuda.pathfinder._headers.supported_nvidia_headers import (
    SUPPORTED_HEADERS_CTK,
    SUPPORTED_HEADERS_CTK_ALL,
    SUPPORTED_HEADERS_NON_CTK,
    SUPPORTED_HEADERS_NON_CTK_ALL,
    SUPPORTED_INSTALL_DIRS_NON_CTK,
    SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_HEADERS_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES = {
    "cusparseLt": r"^nvidia-cusparselt-.*$",
    "cutensor": r"^cutensor-.*$",
    "nvshmem": r"^nvidia-nvshmem-.*$",
}


def test_unknown_libname():
    with pytest.raises(RuntimeError, match=r"^UNKNOWN libname='unknown-libname'$"):
        find_nvidia_header_directory("unknown-libname")


def _located_hdr_dir_asserts(located_hdr_dir):
    assert isinstance(located_hdr_dir, LocatedHeaderDir)
    assert located_hdr_dir.found_via in (
        "site-packages",
        "conda",
        "CUDA_HOME",
        "system-ctk-root",
        "supported_install_dir",
    )


def test_non_ctk_importlib_metadata_distributions_names():
    # Ensure the dict keys above stay in sync with supported_nvidia_headers
    assert sorted(NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES) == sorted(SUPPORTED_HEADERS_NON_CTK_ALL)


@functools.cache
def have_distribution_for(libname: str) -> bool:
    pattern = re.compile(NON_CTK_IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES[libname])
    return any(
        pattern.match(dist.metadata["Name"]) for dist in importlib.metadata.distributions() if "Name" in dist.metadata
    )


@pytest.fixture
def clear_locate_nvidia_header_cache():
    locate_nvidia_header_directory.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()
    yield
    locate_nvidia_header_directory.cache_clear()
    _resolve_system_loaded_abs_path_in_subprocess.cache_clear()


def _create_ctk_header(ctk_root: Path, libname: str) -> str:
    """Create a fake CTK header file and return its directory."""
    header_basename = SUPPORTED_HEADERS_CTK[libname]
    if libname == "nvvm":
        header_dir = ctk_root / "nvvm" / "include"
    elif libname == "cccl":
        header_dir = ctk_root / "include" / "cccl"
    else:
        header_dir = ctk_root / "include"
    header_path = header_dir / header_basename
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.touch()
    return str(header_dir)


def _fake_cudart_canary_abs_path(ctk_root: Path) -> str:
    if IS_WINDOWS:
        return str(ctk_root / "bin" / "x64" / "cudart64_13.dll")
    return str(ctk_root / "lib64" / "libcudart.so.13")


@pytest.mark.parametrize("libname", SUPPORTED_HEADERS_NON_CTK.keys())
def test_locate_non_ctk_headers(info_summary_append, libname):
    hdr_dir = find_nvidia_header_directory(libname)
    located_hdr_dir = locate_nvidia_header_directory(libname)
    assert hdr_dir is None if not located_hdr_dir else hdr_dir == located_hdr_dir.abs_path

    info_summary_append(f"{hdr_dir=!r}")
    if hdr_dir:
        _located_hdr_dir_asserts(located_hdr_dir)
        assert os.path.isdir(hdr_dir)
        assert os.path.isfile(os.path.join(hdr_dir, SUPPORTED_HEADERS_NON_CTK[libname]))
    if have_distribution_for(libname):
        assert hdr_dir is not None
        hdr_dir_parts = hdr_dir.split(os.path.sep)
        assert "site-packages" in hdr_dir_parts
    elif STRICTNESS == "all_must_work":
        assert hdr_dir is not None
        if conda_prefix := os.environ.get("CONDA_PREFIX"):
            assert hdr_dir.startswith(conda_prefix)
        else:
            inst_dirs = SUPPORTED_INSTALL_DIRS_NON_CTK.get(libname)
            if inst_dirs is not None:
                for inst_dir in inst_dirs:
                    globbed = glob.glob(inst_dir)
                    if hdr_dir in globbed:
                        break
                else:
                    raise RuntimeError(f"{hdr_dir=} does not match any {inst_dirs=}")


def test_supported_headers_site_packages_ctk_consistency():
    assert tuple(sorted(SUPPORTED_HEADERS_CTK_ALL)) == tuple(sorted(SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK.keys()))


@pytest.mark.parametrize("libname", SUPPORTED_HEADERS_CTK.keys())
def test_locate_ctk_headers(info_summary_append, libname):
    hdr_dir = find_nvidia_header_directory(libname)
    located_hdr_dir = locate_nvidia_header_directory(libname)
    assert hdr_dir is None if not located_hdr_dir else hdr_dir == located_hdr_dir.abs_path

    info_summary_append(f"{hdr_dir=!r}")
    if hdr_dir:
        _located_hdr_dir_asserts(located_hdr_dir)
        assert os.path.isdir(hdr_dir)
        h_filename = SUPPORTED_HEADERS_CTK[libname]
        assert os.path.isfile(os.path.join(hdr_dir, h_filename))
    if STRICTNESS == "all_must_work":
        assert hdr_dir is not None


@pytest.mark.usefixtures("clear_locate_nvidia_header_cache")
def test_locate_ctk_headers_uses_canary_fallback_when_cuda_home_unset(tmp_path, monkeypatch, mocker):
    ctk_root = tmp_path / "cuda-system"
    expected_hdr_dir = _create_ctk_header(ctk_root, "cudart")

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    mocker.patch.object(find_nvidia_headers_module, "find_sub_dirs_all_sitepackages", return_value=[])
    probe = mocker.patch.object(
        find_nvidia_headers_module,
        "_resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_cudart_canary_abs_path(ctk_root),
    )

    located_hdr_dir = locate_nvidia_header_directory("cudart")

    assert located_hdr_dir is not None
    assert located_hdr_dir.abs_path == expected_hdr_dir
    assert located_hdr_dir.found_via == "system-ctk-root"
    probe.assert_called_once_with("cudart")


@pytest.mark.usefixtures("clear_locate_nvidia_header_cache")
def test_locate_ctk_headers_cuda_home_takes_priority_over_canary(tmp_path, monkeypatch, mocker):
    cuda_home = tmp_path / "cuda-home"
    expected_hdr_dir = _create_ctk_header(cuda_home, "cudart")
    canary_root = tmp_path / "cuda-system"
    _create_ctk_header(canary_root, "cudart")

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    mocker.patch.object(find_nvidia_headers_module, "find_sub_dirs_all_sitepackages", return_value=[])
    probe = mocker.patch.object(
        find_nvidia_headers_module,
        "_resolve_system_loaded_abs_path_in_subprocess",
        return_value=_fake_cudart_canary_abs_path(canary_root),
    )

    located_hdr_dir = locate_nvidia_header_directory("cudart")

    assert located_hdr_dir is not None
    assert located_hdr_dir.abs_path == expected_hdr_dir
    assert located_hdr_dir.found_via == "CUDA_HOME"
    probe.assert_not_called()


@pytest.mark.usefixtures("clear_locate_nvidia_header_cache")
def test_locate_ctk_headers_canary_miss_paths_are_non_fatal(monkeypatch, mocker):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    mocker.patch.object(find_nvidia_headers_module, "find_sub_dirs_all_sitepackages", return_value=[])
    mocker.patch.object(
        find_nvidia_headers_module,
        "_resolve_system_loaded_abs_path_in_subprocess",
        return_value=None,
    )

    assert locate_nvidia_header_directory("cudart") is None
    assert find_nvidia_header_directory("cudart") is None


@pytest.mark.usefixtures("clear_locate_nvidia_header_cache")
def test_locate_ctk_headers_canary_probe_errors_are_not_masked(monkeypatch, mocker):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    mocker.patch.object(find_nvidia_headers_module, "find_sub_dirs_all_sitepackages", return_value=[])
    mocker.patch.object(
        find_nvidia_headers_module,
        "_resolve_system_loaded_abs_path_in_subprocess",
        side_effect=RuntimeError("canary probe failed"),
    )

    with pytest.raises(RuntimeError, match="canary probe failed"):
        locate_nvidia_header_directory("cudart")
    with pytest.raises(RuntimeError, match="canary probe failed"):
        find_nvidia_header_directory("cudart")
