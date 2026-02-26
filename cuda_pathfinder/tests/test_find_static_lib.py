# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

import cuda.pathfinder._static_libs.find_static_lib as find_static_lib_module
from cuda.pathfinder._static_libs.find_static_lib import (
    SUPPORTED_STATIC_LIBS,
    StaticLibNotFoundError,
    find_static_lib,
    locate_static_lib,
)
from cuda.pathfinder._utils.platform_aware import quote_for_shell

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_STATIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

CUDADEVRT_INFO = find_static_lib_module._SUPPORTED_STATIC_LIBS_INFO["cudadevrt"]


@pytest.fixture
def clear_find_static_lib_cache():
    find_static_lib_module.find_static_lib.cache_clear()
    yield
    find_static_lib_module.find_static_lib.cache_clear()


def _make_static_lib_file(dir_path: Path, filename: str) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename
    file_path.touch()
    return str(file_path)


def _conda_anchor(conda_prefix: Path) -> Path:
    if find_static_lib_module.IS_WINDOWS:
        return conda_prefix / "Library"
    return conda_prefix


def _located_static_lib_asserts(located_static_lib):
    """Common assertions for a located static library."""
    assert located_static_lib is not None
    assert isinstance(located_static_lib.name, str)
    assert isinstance(located_static_lib.abs_path, str)
    assert isinstance(located_static_lib.filename, str)
    assert isinstance(located_static_lib.found_via, str)
    assert located_static_lib.found_via in ("site-packages", "conda", "CUDA_HOME")
    assert os.path.isfile(located_static_lib.abs_path)


@pytest.mark.usefixtures("clear_find_static_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_STATIC_LIBS)
def test_locate_static_lib(info_summary_append, libname):
    try:
        located_lib = locate_static_lib(libname)
        lib_path = find_static_lib(libname)
    except StaticLibNotFoundError:
        if STRICTNESS == "all_must_work":
            raise
        info_summary_append(f"{libname}: not found")
        return

    info_summary_append(f"abs_path={quote_for_shell(lib_path)}")
    _located_static_lib_asserts(located_lib)
    assert os.path.isfile(lib_path)
    assert lib_path == located_lib.abs_path
    expected_filename = located_lib.filename
    assert os.path.basename(lib_path) == expected_filename


@pytest.mark.usefixtures("clear_find_static_lib_cache")
def test_locate_static_lib_search_order(monkeypatch, tmp_path):
    filename = CUDADEVRT_INFO["filename"]
    conda_rel_path = CUDADEVRT_INFO["conda_rel_path"]

    site_pkg_rel = CUDADEVRT_INFO["site_packages_dirs"][0]
    site_packages_lib_dir = tmp_path / "site-packages" / Path(site_pkg_rel.replace("/", os.sep))
    site_packages_path = _make_static_lib_file(site_packages_lib_dir, filename)

    conda_prefix = tmp_path / "conda-prefix"
    conda_lib_dir = _conda_anchor(conda_prefix) / Path(conda_rel_path)
    conda_path = _make_static_lib_file(conda_lib_dir, filename)

    cuda_home = tmp_path / "cuda-home"
    ctk_rel_path = CUDADEVRT_INFO["ctk_rel_paths"][0]
    cuda_home_lib_dir = cuda_home / Path(ctk_rel_path)
    cuda_home_path = _make_static_lib_file(cuda_home_lib_dir, filename)

    monkeypatch.setattr(
        find_static_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [str(site_packages_lib_dir)],
    )
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    located_lib = locate_static_lib("cudadevrt")
    assert located_lib.abs_path == site_packages_path
    assert located_lib.found_via == "site-packages"
    os.remove(site_packages_path)

    located_lib = locate_static_lib("cudadevrt")
    assert located_lib.abs_path == conda_path
    assert located_lib.found_via == "conda"
    os.remove(conda_path)

    located_lib = locate_static_lib("cudadevrt")
    assert located_lib.abs_path == cuda_home_path
    assert located_lib.found_via == "CUDA_HOME"


@pytest.mark.usefixtures("clear_find_static_lib_cache")
def test_find_static_lib_not_found_error_includes_cuda_home_directory_listing(monkeypatch, tmp_path):
    filename = CUDADEVRT_INFO["filename"]
    ctk_rel_path = CUDADEVRT_INFO["ctk_rel_paths"][0]

    cuda_home = tmp_path / "cuda-home"
    lib_dir = cuda_home / Path(ctk_rel_path)
    lib_dir.mkdir(parents=True, exist_ok=True)
    extra_file = lib_dir / "README.txt"
    extra_file.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        find_static_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(StaticLibNotFoundError, match=rf'Failure finding "{filename}"') as exc_info:
        find_static_lib("cudadevrt")

    message = str(exc_info.value)
    expected_missing_file = os.path.join(str(lib_dir), filename)
    assert f"No such file: {expected_missing_file}" in message
    assert f'listdir("{lib_dir}"):' in message
    assert "README.txt" in message


@pytest.mark.usefixtures("clear_find_static_lib_cache")
def test_find_static_lib_not_found_error_without_cuda_home(monkeypatch):
    filename = CUDADEVRT_INFO["filename"]

    monkeypatch.setattr(
        find_static_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(
        StaticLibNotFoundError,
        match=rf'Failure finding "{filename}": CUDA_HOME/CUDA_PATH not set',
    ):
        find_static_lib("cudadevrt")


def test_find_static_lib_invalid_name():
    with pytest.raises(ValueError, match="Unknown static library"):
        find_static_lib_module.locate_static_lib("invalid")
