# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import cuda.pathfinder._static_libs.find_bitcode_lib as find_bitcode_lib_module

FILENAME = "libdevice.10.bc"

SITE_PACKAGES_REL_DIR_CUDA12 = "nvidia/cuda_nvcc/nvvm/libdevice"
SITE_PACKAGES_REL_DIR_CUDA13 = "nvidia/cuda_nvvm/nvvm/libdevice"


@pytest.fixture
def clear_find_bitcode_lib_cache():
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    yield
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()


def _make_bitcode_lib_file(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, FILENAME)
    with open(file_path, "wb"):
        pass
    return file_path


@pytest.mark.parametrize("rel_dir", [SITE_PACKAGES_REL_DIR_CUDA12, SITE_PACKAGES_REL_DIR_CUDA13])
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_via_site_packages(monkeypatch, mocker, tmp_path, rel_dir):
    bitcode_lib_dir = tmp_path.joinpath(*rel_dir.split("/"))
    expected_path = str(_make_bitcode_lib_file(str(bitcode_lib_dir)))

    mocker.patch.object(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[str(bitcode_lib_dir)],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_bitcode_lib_module.locate_bitcode_lib("device")

    assert result is not None
    assert result.abs_path == expected_path
    assert result.name == "device"
    assert result.filename == FILENAME
    assert os.path.isfile(result.abs_path)


# same for cu12/cu13
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_via_conda(monkeypatch, mocker, tmp_path):
    rel_path = os.path.join("nvvm", "libdevice")
    bitcode_lib_dir = tmp_path / rel_path
    expected_path = str(_make_bitcode_lib_file(str(bitcode_lib_dir)))

    mocker.patch.object(find_bitcode_lib_module, "IS_WINDOWS", False)
    mocker.patch.object(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[],
    )
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_bitcode_lib_module.locate_bitcode_lib("device")

    assert result is not None
    assert result.abs_path == expected_path
    assert os.path.isfile(result.abs_path)


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_via_cuda_home(monkeypatch, mocker, tmp_path):
    rel_path = os.path.join("nvvm", "libdevice")
    bitcode_lib_dir = tmp_path / rel_path
    expected_path = str(_make_bitcode_lib_file(str(bitcode_lib_dir)))

    mocker.patch.object(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_bitcode_lib_module.locate_bitcode_lib("device")

    assert result is not None
    assert result.abs_path == expected_path
    assert os.path.isfile(result.abs_path)


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_returns_path(monkeypatch, mocker, tmp_path):
    rel_path = os.path.join("nvvm", "libdevice")
    bitcode_lib_dir = tmp_path / rel_path
    expected_path = str(_make_bitcode_lib_file(str(bitcode_lib_dir)))

    mocker.patch.object(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        return_value=[str(bitcode_lib_dir)],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = find_bitcode_lib_module.find_bitcode_lib("device")

    assert result == expected_path
    assert isinstance(result, str)


def test_find_bitcode_lib_invalid_name():
    with pytest.raises(ValueError, match="Unknown bitcode library"):
        find_bitcode_lib_module.locate_bitcode_lib("invalid")
