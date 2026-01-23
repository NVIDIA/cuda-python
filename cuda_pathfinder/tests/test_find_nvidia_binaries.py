# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_binary_utility
from cuda.pathfinder._binaries import find_nvidia_binary_utility as binary_finder_module
from cuda.pathfinder._binaries.find_nvidia_binary_utility import UnsupportedBinaryError
from cuda.pathfinder._binaries.supported_nvidia_binaries import (
    SITE_PACKAGES_BINDIRS,
    SUPPORTED_BINARIES,
    SUPPORTED_BINARIES_ALL,
)


def test_unknown_utility_name():
    with pytest.raises(UnsupportedBinaryError, match=r"'unknown-utility' is not supported"):
        find_nvidia_binary_utility("unknown-utility")


@pytest.mark.parametrize("utility_name", SUPPORTED_BINARIES)
def test_find_binary_utilities(info_summary_append, utility_name):
    bin_path = find_nvidia_binary_utility(utility_name)
    info_summary_append(f"{bin_path=!r}")

    if bin_path is not None:
        assert os.path.isfile(bin_path)


def test_supported_binaries_consistency():
    assert set(SUPPORTED_BINARIES).issubset(SUPPORTED_BINARIES_ALL)
    assert set(SITE_PACKAGES_BINDIRS).issubset(SUPPORTED_BINARIES_ALL)


@pytest.fixture
def clear_find_binary_cache():
    find_nvidia_binary_utility.cache_clear()
    yield
    find_nvidia_binary_utility.cache_clear()


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_search_path_includes_site_packages_conda_cuda(monkeypatch):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")
    site_dir = os.path.join("site-packages", "cuda_nvcc", "bin")
    captured = {}

    def fake_find_sub_dirs_all_sitepackages(sub_dirs):
        captured["sub_dirs"] = tuple(sub_dirs)
        return [site_dir]

    def fake_which(name, path=None, **_kwargs):
        captured["name"] = name
        captured["path"] = path
        return os.path.join(os.sep, "resolved", "nvcc")

    monkeypatch.setattr(binary_finder_module, "IS_WINDOWS", False)
    monkeypatch.setattr(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    monkeypatch.setattr(binary_finder_module, "find_sub_dirs_all_sitepackages", fake_find_sub_dirs_all_sitepackages)
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    monkeypatch.setattr(binary_finder_module, "get_cuda_home_or_path", lambda: cuda_home)
    monkeypatch.setattr(binary_finder_module.shutil, "which", fake_which)

    result = find_nvidia_binary_utility("nvcc")

    assert result == os.path.join(os.sep, "resolved", "nvcc")
    assert captured["name"] == "nvcc"
    assert captured["sub_dirs"] == tuple(site_key.split(os.sep))
    expected_dirs = [
        site_dir,
        os.path.join(conda_prefix, "bin"),
        os.path.join(cuda_home, "bin"),
    ]
    assert captured["path"] == os.pathsep.join(expected_dirs)


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_windows_extension_and_search_dirs(monkeypatch):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")
    site_dir = os.path.join("site-packages", "cuda_nvcc", "bin")
    captured = {}

    def fake_find_sub_dirs_all_sitepackages(sub_dirs):
        captured["sub_dirs"] = tuple(sub_dirs)
        return [site_dir]

    def fake_which(name, path=None, **_kwargs):
        captured["name"] = name
        captured["path"] = path
        return os.path.join(os.sep, "resolved", "nvcc.exe")

    monkeypatch.setattr(binary_finder_module, "IS_WINDOWS", True)
    monkeypatch.setattr(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    monkeypatch.setattr(binary_finder_module, "find_sub_dirs_all_sitepackages", fake_find_sub_dirs_all_sitepackages)
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    monkeypatch.setattr(binary_finder_module, "get_cuda_home_or_path", lambda: cuda_home)
    monkeypatch.setattr(binary_finder_module.shutil, "which", fake_which)

    result = find_nvidia_binary_utility("nvcc")

    assert result == os.path.join(os.sep, "resolved", "nvcc.exe")
    assert captured["name"] == "nvcc.exe"
    assert captured["sub_dirs"] == tuple(site_key.split(os.sep))
    expected_dirs = [
        site_dir,
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "bin"),
        os.path.join(cuda_home, "bin", "x64"),
        os.path.join(cuda_home, "bin", "x86_64"),
        os.path.join(cuda_home, "bin"),
    ]
    assert captured["path"] == os.pathsep.join(expected_dirs)


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_caching_per_utility():
    """Verify that different utilities have independent cache entries."""
    nvdisasm1 = find_nvidia_binary_utility("nvdisasm")
    nvcc1 = find_nvidia_binary_utility("nvcc")
    nvdisasm2 = find_nvidia_binary_utility("nvdisasm")
    nvcc2 = find_nvidia_binary_utility("nvcc")

    # Same utility should return cached result
    assert nvdisasm1 is nvdisasm2
    assert nvcc1 is nvcc2

    # Different utilities should have different results (unless at least one of
    # them is None)
    if nvdisasm1 is not None and nvcc1 is not None:
        assert nvdisasm1 != nvcc1
