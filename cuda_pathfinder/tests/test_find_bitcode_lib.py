# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

import cuda.pathfinder._static_libs.find_bitcode_lib as find_bitcode_lib_module
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    SUPPORTED_BITCODE_LIBS,
    BitcodeLibNotFoundError,
    find_bitcode_lib,
    locate_bitcode_lib,
)
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_BITCODE_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def _bitcode_lib_info(libname: str):
    return find_bitcode_lib_module._SUPPORTED_BITCODE_LIBS_INFO[libname]


def _bitcode_lib_filename(libname: str) -> str:
    return _bitcode_lib_info(libname)["filename"]


@pytest.fixture
def clear_find_bitcode_lib_cache():
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    get_cuda_path_or_home.cache_clear()
    yield
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    get_cuda_path_or_home.cache_clear()


def _make_bitcode_lib_file(dir_path: Path, libname: str = "device") -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / _bitcode_lib_filename(libname)
    file_path.touch()
    return str(file_path)


def _bitcode_lib_dir_under(anchor_dir: Path, libname: str = "device") -> Path:
    return anchor_dir / _bitcode_lib_info(libname)["rel_path"]


def _site_packages_bitcode_lib_dir_under(anchor_dir: Path, libname: str) -> Path:
    rel_dir = _bitcode_lib_info(libname)["site_packages_dirs"][0]
    return anchor_dir.joinpath(*rel_dir.split("/"))


def _conda_anchor(conda_prefix: Path) -> Path:
    if find_bitcode_lib_module.IS_WINDOWS:
        return conda_prefix / "Library"
    return conda_prefix


def _located_bitcode_lib_asserts(located_bitcode_lib):
    """Common assertions for a located bitcode library."""
    assert located_bitcode_lib is not None
    assert isinstance(located_bitcode_lib.name, str)
    assert isinstance(located_bitcode_lib.abs_path, str)
    assert isinstance(located_bitcode_lib.filename, str)
    assert isinstance(located_bitcode_lib.found_via, str)
    assert located_bitcode_lib.found_via in ("site-packages", "conda", "CUDA_PATH")
    assert os.path.isfile(located_bitcode_lib.abs_path)


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_BITCODE_LIBS)
def test_locate_bitcode_lib(info_summary_append, libname):
    try:
        located_lib = locate_bitcode_lib(libname)
        lib_path = find_bitcode_lib(libname)
    except BitcodeLibNotFoundError:
        if STRICTNESS == "all_must_work":
            raise
        info_summary_append(f"{libname}: not found")
        return

    info_summary_append(f"{lib_path=!r}")
    _located_bitcode_lib_asserts(located_lib)
    assert os.path.isfile(lib_path)
    assert lib_path == located_lib.abs_path
    expected_filename = located_lib.filename
    assert os.path.basename(lib_path) == expected_filename


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_BITCODE_LIBS)
def test_locate_bitcode_lib_search_order(monkeypatch, tmp_path, libname):
    site_packages_lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", libname)
    site_packages_path = _make_bitcode_lib_file(site_packages_lib_dir, libname)

    conda_prefix = tmp_path / "conda-prefix"
    conda_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(_conda_anchor(conda_prefix), libname), libname)

    cuda_home = tmp_path / "cuda-home"
    cuda_home_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(cuda_home, libname), libname)

    site_packages_sub_dirs = tuple(
        tuple(rel_dir.split("/")) for rel_dir in _bitcode_lib_info(libname)["site_packages_dirs"]
    )

    def find_expected_sub_dir(sub_dir):
        assert sub_dir in site_packages_sub_dirs
        if sub_dir == site_packages_sub_dirs[0]:
            return [str(site_packages_lib_dir)]
        return []

    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        find_expected_sub_dir,
    )
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    located_lib = locate_bitcode_lib(libname)
    assert located_lib.abs_path == site_packages_path
    assert located_lib.found_via == "site-packages"
    os.remove(site_packages_path)

    located_lib = locate_bitcode_lib(libname)
    assert located_lib.abs_path == conda_path
    assert located_lib.found_via == "conda"
    os.remove(conda_path)

    located_lib = locate_bitcode_lib(libname)
    assert located_lib.abs_path == cuda_home_path
    assert located_lib.found_via == "CUDA_PATH"


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_not_found_error_includes_cuda_home_directory_listing(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda-home"
    lib_dir = _bitcode_lib_dir_under(cuda_home)
    lib_dir.mkdir(parents=True, exist_ok=True)
    extra_file = lib_dir / "README.txt"
    extra_file.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(BitcodeLibNotFoundError, match=r'Failure finding "libdevice\.10\.bc"') as exc_info:
        find_bitcode_lib("device")

    message = str(exc_info.value)
    expected_missing_file = os.path.join(str(lib_dir), _bitcode_lib_filename("device"))
    assert f"No such file: {expected_missing_file}" in message
    assert f'listdir("{lib_dir}"):' in message
    assert "README.txt" in message


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_not_found_error_without_cuda_home(monkeypatch):
    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(
        BitcodeLibNotFoundError,
        match=r'Failure finding "libdevice\.10\.bc": CUDA_HOME/CUDA_PATH not set',
    ):
        find_bitcode_lib("device")


def test_find_bitcode_lib_invalid_name():
    with pytest.raises(ValueError, match="Unknown bitcode library"):
        find_bitcode_lib_module.locate_bitcode_lib("invalid")
