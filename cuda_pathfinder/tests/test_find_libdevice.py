# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_libdevice
from cuda.pathfinder._static_libs import find_libdevice as find_libdevice_module

FILENAME = "libdevice.10.bc"

SITE_PACKAGES_REL_DIR_CUDA12 = "nvidia/cuda_nvcc/nvvm/libdevice"
SITE_PACKAGES_REL_DIR_CUDA13 = "nvidia/cuda_nvvm/nvvm/libdevice"


@pytest.fixture
def clear_find_libdevice_cache():
    find_libdevice.cache_clear()
    yield
    find_libdevice.cache_clear()


def _make_libdevice_file(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, FILENAME)
    with open(file_path, "wb"):
        pass
    return file_path


@pytest.mark.parametrize("rel_dir", [SITE_PACKAGES_REL_DIR_CUDA12, SITE_PACKAGES_REL_DIR_CUDA13])
@pytest.mark.usefixtures("clear_find_libdevice_cache")
def test_find_libdevice_via_site_packages(monkeypatch, mocker, tmp_path, rel_dir):
    libdevice_dir = tmp_path.joinpath(*rel_dir.split("/"))
    expected_path = str(_make_libdevice_file(str(libdevice_dir)))

    mocker.patch.object(
        find_libdevice_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[str(libdevice_dir)],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_libdevice()

    assert result == expected_path
    assert os.path.isfile(result)


# same for cu12/cu13
@pytest.mark.usefixtures("clear_find_libdevice_cache")
def test_find_libdevice_via_conda(monkeypatch, mocker, tmp_path):
    rel_path = os.path.join("nvvm", "libdevice")
    libdevice_dir = tmp_path / rel_path
    expected_path = str(_make_libdevice_file(str(libdevice_dir)))

    mocker.patch.object(find_libdevice_module, "IS_WINDOWS", False)
    mocker.patch.object(
        find_libdevice_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[],
    )
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_libdevice()

    assert result == expected_path
    assert os.path.isfile(result)


@pytest.mark.usefixtures("clear_find_libdevice_cache")
def test_find_libdevice_via_cuda_home(monkeypatch, mocker, tmp_path):
    rel_path = os.path.join("nvvm", "libdevice")
    libdevice_dir = tmp_path / rel_path
    expected_path = str(_make_libdevice_file(str(libdevice_dir)))

    mocker.patch.object(
        find_libdevice_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_libdevice()

    assert result == expected_path
    assert os.path.isfile(result)
