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
    return _bitcode_lib_info(libname)["filenames"][None]


@pytest.fixture
def clear_find_bitcode_lib_cache():
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    get_cuda_path_or_home.cache_clear()
    yield
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    get_cuda_path_or_home.cache_clear()


def _make_bitcode_lib_file(file_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()
    return str(file_path)


def _bitcode_lib_dir_under(anchor_dir: Path, libname: str) -> Path:
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
    assert Path(located_bitcode_lib.abs_path).is_file()


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
    assert Path(lib_path).is_file()
    assert lib_path == located_lib.abs_path
    assert located_lib.filename == _bitcode_lib_filename(libname)


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_BITCODE_LIBS)
def test_locate_bitcode_lib_search_order(monkeypatch, tmp_path, libname):
    filename = _bitcode_lib_filename(libname)
    site_packages_lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", libname)
    site_packages_path = _make_bitcode_lib_file(site_packages_lib_dir / filename)

    conda_prefix = tmp_path / "conda-prefix"
    conda_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(_conda_anchor(conda_prefix), libname) / filename)

    cuda_home = tmp_path / "cuda-home"
    cuda_home_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(cuda_home, libname) / filename)

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
@pytest.mark.parametrize("source", ("site-packages", "conda", "CUDA_PATH"))
@pytest.mark.agent_authored(model="gpt-6")
def test_locate_bitcode_lib_returns_resolved_path_for_relative_root(monkeypatch, tmp_path, source):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    info = _bitcode_lib_info("device")
    root = Path("relative-root")
    if source == "site-packages":
        lib_dir = root / Path(info["site_packages_dirs"][0])
        monkeypatch.setattr(find_bitcode_lib_module, "find_sub_dirs_all_sitepackages", lambda _sub_dir: [str(lib_dir)])
    else:
        monkeypatch.setattr(find_bitcode_lib_module, "find_sub_dirs_all_sitepackages", lambda _sub_dir: [])
        if source == "conda":
            monkeypatch.setenv("CONDA_PREFIX", str(root))
            lib_dir = _conda_anchor(root) / Path(info["rel_path"])
        else:
            monkeypatch.setenv("CUDA_HOME", str(root))
            lib_dir = root / Path(info["rel_path"])

    expected = (tmp_path / lib_dir / _bitcode_lib_filename("device")).resolve()
    expected.parent.mkdir(parents=True)
    expected.touch()
    located = locate_bitcode_lib("device")
    assert located.found_via == source
    assert located.abs_path == str(expected)
    assert find_bitcode_lib("device") == str(expected)

    monkeypatch.chdir(tmp_path.parent)
    assert Path(located.abs_path).is_file()
    assert Path(find_bitcode_lib("device")).is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink parent traversal")
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-6")
def test_locate_bitcode_lib_preserves_symlink_parent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", "alias/..")
    monkeypatch.setattr(find_bitcode_lib_module, "find_sub_dirs_all_sitepackages", lambda _sub_dir: [])

    real_root = tmp_path / "real"
    (real_root / "child").mkdir(parents=True)
    (tmp_path / "alias").symlink_to(real_root / "child", target_is_directory=True)
    info = _bitcode_lib_info("device")
    expected = real_root / info["rel_path"] / _bitcode_lib_filename("device")
    expected.parent.mkdir(parents=True)
    expected.touch()

    located = locate_bitcode_lib("device")
    assert located.abs_path == str(expected.resolve())
    cached = find_bitcode_lib("device")
    monkeypatch.chdir(tmp_path.parent)
    assert cached == str(expected.resolve())
    assert find_bitcode_lib("device") == cached


@pytest.mark.skipif(os.name == "nt", reason="Symlink creation requires privileges on Windows")
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-6")
def test_locate_bitcode_lib_resolves_symlinked_file(monkeypatch, tmp_path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(find_bitcode_lib_module, "find_sub_dirs_all_sitepackages", lambda _sub_dir: [])
    root = tmp_path / "toolkit"
    monkeypatch.setenv("CUDA_HOME", str(root))
    info = _bitcode_lib_info("device")
    target = tmp_path / "actual-library"
    target.touch()
    filename = _bitcode_lib_filename("device")
    candidate = root / info["rel_path"] / filename
    candidate.parent.mkdir(parents=True)
    candidate.symlink_to(target)

    located = locate_bitcode_lib("device")
    assert located.abs_path == str(target.resolve())
    assert located.filename == filename
    assert find_bitcode_lib("device") == located.abs_path


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_find_bitcode_lib_not_found_error_includes_cuda_home_directory_listing(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda-home"
    lib_dir = _bitcode_lib_dir_under(cuda_home, "device")
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
    expected_missing_file = lib_dir / _bitcode_lib_filename("device")
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


@pytest.fixture
def sm_bitcode_layout(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    cuda_home = tmp_path / "cuda-home"
    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda sub_dir: [str(site_packages.joinpath(*sub_dir))],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    return site_packages, cuda_home


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize(
    "libname", [name for name in SUPPORTED_BITCODE_LIBS if name in ("nccl_device", "nvshmem_device")]
)
def test_sm_arch_selection_and_cache(sm_bitcode_layout, libname):
    site_packages, _ = sm_bitcode_layout
    site_dir = _site_packages_bitcode_lib_dir_under(site_packages, libname)
    future_sm = _make_bitcode_lib_file(site_dir / f"lib{libname}_sm_999a.bc")
    sm100 = _make_bitcode_lib_file(site_dir / f"lib{libname}_sm_100.bc")
    default = _make_bitcode_lib_file(site_dir / _bitcode_lib_filename(libname))

    assert find_bitcode_lib(libname, sm_arch="999a") == future_sm
    assert find_bitcode_lib(libname, sm_arch="100") == sm100
    assert find_bitcode_lib(libname) == default


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_BITCODE_LIBS)
def test_sm_arch_does_not_use_default_file(sm_bitcode_layout, libname):
    site_packages, _ = sm_bitcode_layout
    site_dir = _site_packages_bitcode_lib_dir_under(site_packages, libname)
    default = _make_bitcode_lib_file(site_dir / _bitcode_lib_filename(libname))
    assert find_bitcode_lib(libname, sm_arch=None) == default
    with pytest.raises(BitcodeLibNotFoundError, match="sm_arch='100'"):
        find_bitcode_lib(libname, sm_arch="100")


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
def test_exact_sm_path_takes_precedence_over_wildcard(monkeypatch, sm_bitcode_layout):
    site_packages, cuda_home = sm_bitcode_layout
    site_dir = _site_packages_bitcode_lib_dir_under(site_packages, "device")
    config = dict(_bitcode_lib_info("device"))
    config["filenames"] = {"100a": "{sm_arch}/device.bc", "*": "generic.bc"}
    monkeypatch.setitem(find_bitcode_lib_module._SUPPORTED_BITCODE_LIBS_INFO, "device", config)
    generic = _make_bitcode_lib_file(site_dir / "generic.bc")
    exact_dir = _bitcode_lib_dir_under(cuda_home, "device") / "100a"
    exact = _make_bitcode_lib_file(exact_dir / "device.bc")

    located = locate_bitcode_lib("device", sm_arch="100a")
    assert located.abs_path == exact
    assert located.filename == "device.bc"
    assert locate_bitcode_lib("device", sm_arch="100").abs_path == generic
    Path(exact).unlink()
    with pytest.raises(BitcodeLibNotFoundError, match=r"100a/device\.bc") as exc_info:
        locate_bitcode_lib("device", sm_arch="100a")
    assert f'listdir("{exact_dir}"):' in str(exc_info.value)
    with pytest.raises(BitcodeLibNotFoundError, match="no default file configured"):
        locate_bitcode_lib("device")


@pytest.mark.parametrize("sm_arch", ["", 100, "sm_100", "*", "100/../90", r"100\..\90"])
def test_invalid_sm_arch(sm_arch):
    with pytest.raises(ValueError, match="sm_arch must be a string"):
        locate_bitcode_lib("device", sm_arch=sm_arch)
