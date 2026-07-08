# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import os
import re
from pathlib import Path

import pytest

import cuda.pathfinder._static_libs.find_bitcode_lib as find_bitcode_lib_module
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    SUPPORTED_BITCODE_LIBS,
    BitcodeLibNotFoundError,
    find_bitcode_lib,
    find_bitcode_lib_by_name,
    locate_bitcode_lib,
    locate_bitcode_lib_by_name,
)
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_BITCODE_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")

_NVSHMEM_DISTRIBUTION_PATTERN = re.compile(r"^nvidia-nvshmem-cu(?:12|13)$", re.IGNORECASE)
_NVSHMEM_LIB_DIR_PARTS = ("nvidia", "nvshmem", "lib")
_NVSHMEM_LEGACY_FILENAME = "libnvshmem_device.bc"


def _bitcode_lib_info(libname: str):
    return find_bitcode_lib_module._SUPPORTED_BITCODE_LIBS_INFO[libname]


def _bitcode_lib_filename(libname: str) -> str:
    return _bitcode_lib_info(libname)["filename"]


@pytest.fixture
def clear_find_bitcode_lib_cache():
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    find_bitcode_lib_module.find_bitcode_lib_by_name.cache_clear()
    get_cuda_path_or_home.cache_clear()
    yield
    find_bitcode_lib_module.find_bitcode_lib.cache_clear()
    find_bitcode_lib_module.find_bitcode_lib_by_name.cache_clear()
    get_cuda_path_or_home.cache_clear()


def _make_bitcode_lib_file(dir_path: Path, filename: str) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename
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


def _installed_nvshmem_distributions():
    return tuple(
        dist
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata and _NVSHMEM_DISTRIBUTION_PATTERN.fullmatch(dist.metadata["Name"])
    )


def _installed_nvshmem_bitcode_paths(distributions):
    paths = {
        Path(dist.locate_file(file))
        for dist in distributions
        for file in (dist.files or ())
        if file.parts[-4:-1] == _NVSHMEM_LIB_DIR_PARTS
        and file.name.startswith("libnvshmem_device")
        and file.suffix == ".bc"
    }
    return tuple(sorted(paths))


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
@pytest.mark.parametrize("libname", tuple(name for name in SUPPORTED_BITCODE_LIBS if name != "nvshmem_device"))
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


@pytest.mark.skipif(
    find_bitcode_lib_module.IS_WINDOWS,
    reason="NVSHMEM wheel test dependencies are not installed on Windows",
)
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-5")
def test_locate_installed_nvshmem_bitcode(info_summary_append):
    distributions = _installed_nvshmem_distributions()
    if not distributions:
        if STRICTNESS == "all_must_work":
            pytest.fail("No nvidia-nvshmem-cu12/13 distribution is installed")
        info_summary_append("NVSHMEM distribution not installed")
        return

    bitcode_paths = _installed_nvshmem_bitcode_paths(distributions)
    versions = ", ".join(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions)
    info_summary_append(f"{versions}: {[path.name for path in bitcode_paths]}")
    assert bitcode_paths, f"No NVSHMEM device bitcode files found in {versions}"

    for expected_path in bitcode_paths:
        located_lib = locate_bitcode_lib_by_name("nvshmem_device", expected_path.name)
        assert Path(located_lib.abs_path).samefile(expected_path)
        assert located_lib.filename == expected_path.name
        assert located_lib.found_via == "site-packages"
        assert Path(find_bitcode_lib_by_name("nvshmem_device", expected_path.name)).samefile(expected_path)

    legacy_paths = [path for path in bitcode_paths if path.name == _NVSHMEM_LEGACY_FILENAME]
    if legacy_paths:
        assert len(legacy_paths) == 1
        located_lib = locate_bitcode_lib("nvshmem_device")
        assert Path(located_lib.abs_path).samefile(legacy_paths[0])
        assert Path(find_bitcode_lib("nvshmem_device")).samefile(legacy_paths[0])


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.parametrize("libname", SUPPORTED_BITCODE_LIBS)
def test_locate_bitcode_lib_search_order(monkeypatch, tmp_path, libname):
    filename = _bitcode_lib_filename(libname)
    site_packages_lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", libname)
    site_packages_path = _make_bitcode_lib_file(site_packages_lib_dir, filename)

    conda_prefix = tmp_path / "conda-prefix"
    conda_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(_conda_anchor(conda_prefix), libname), filename)

    cuda_home = tmp_path / "cuda-home"
    cuda_home_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(cuda_home, libname), filename)

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


@pytest.mark.skipif(
    find_bitcode_lib_module.IS_WINDOWS,
    reason="NVSHMEM wheel test dependencies are not installed on Windows",
)
@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-5")
def test_nvshmem_legacy_unversioned_bitcode_filename(monkeypatch, tmp_path):
    libname = "nvshmem_device"
    assert _bitcode_lib_filename(libname) == _NVSHMEM_LEGACY_FILENAME

    lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", libname)
    expected_path = _make_bitcode_lib_file(lib_dir, _NVSHMEM_LEGACY_FILENAME)
    expected_sub_dir = tuple(_bitcode_lib_info(libname)["site_packages_dirs"][0].split("/"))

    def find_expected_sub_dir(sub_dir):
        assert sub_dir == expected_sub_dir
        return [str(lib_dir)]

    monkeypatch.setattr(find_bitcode_lib_module, "find_sub_dirs_all_sitepackages", find_expected_sub_dir)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    located_lib = locate_bitcode_lib(libname)
    assert located_lib.abs_path == expected_path
    assert located_lib.filename == _NVSHMEM_LEGACY_FILENAME
    assert located_lib.found_via == "site-packages"
    assert find_bitcode_lib(libname) == expected_path

    located_by_name = locate_bitcode_lib_by_name(libname, _NVSHMEM_LEGACY_FILENAME)
    assert located_by_name.abs_path == expected_path
    assert located_by_name.filename == _NVSHMEM_LEGACY_FILENAME
    assert located_by_name.found_via == "site-packages"
    assert find_bitcode_lib_by_name(libname, _NVSHMEM_LEGACY_FILENAME) == expected_path


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-5")
def test_bitcode_lib_by_name_search_order(monkeypatch, tmp_path):
    libname = "device"
    filename = "libdevice_sm_90.bc"
    site_packages_lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", libname)
    site_packages_path = _make_bitcode_lib_file(site_packages_lib_dir, filename)

    conda_prefix = tmp_path / "conda-prefix"
    conda_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(_conda_anchor(conda_prefix), libname), filename)

    cuda_home = tmp_path / "cuda-home"
    cuda_home_path = _make_bitcode_lib_file(_bitcode_lib_dir_under(cuda_home, libname), filename)

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

    located_lib = locate_bitcode_lib_by_name(libname, filename)
    assert located_lib.abs_path == site_packages_path
    assert located_lib.filename == filename
    assert located_lib.found_via == "site-packages"
    assert find_bitcode_lib_by_name(libname, filename) == site_packages_path
    os.remove(site_packages_path)
    find_bitcode_lib_by_name.cache_clear()

    located_lib = locate_bitcode_lib_by_name(libname, filename)
    assert located_lib.abs_path == conda_path
    assert located_lib.filename == filename
    assert located_lib.found_via == "conda"
    assert find_bitcode_lib_by_name(libname, filename) == conda_path
    os.remove(conda_path)
    find_bitcode_lib_by_name.cache_clear()

    located_lib = locate_bitcode_lib_by_name(libname, filename)
    assert located_lib.abs_path == cuda_home_path
    assert located_lib.filename == filename
    assert located_lib.found_via == "CUDA_PATH"
    assert find_bitcode_lib_by_name(libname, filename) == cuda_home_path


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-5")
def test_find_bitcode_lib_by_name_cache_keeps_filenames_separate(monkeypatch, tmp_path):
    lib_dir = _site_packages_bitcode_lib_dir_under(tmp_path / "site-packages", "device")
    sm80_path = _make_bitcode_lib_file(lib_dir, "libdevice_sm_80.bc")
    sm90_path = _make_bitcode_lib_file(lib_dir, "libdevice_sm_90.bc")

    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [str(lib_dir)],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert find_bitcode_lib_by_name("device", "libdevice_sm_80.bc") == sm80_path
    assert find_bitcode_lib_by_name("device", "libdevice_sm_90.bc") == sm90_path


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
    expected_missing_file = os.path.join(str(lib_dir), _bitcode_lib_filename("device"))
    assert f"No such file: {expected_missing_file}" in message
    assert f'listdir("{lib_dir}"):' in message
    assert "README.txt" in message


@pytest.mark.usefixtures("clear_find_bitcode_lib_cache")
@pytest.mark.agent_authored(model="gpt-5")
def test_find_bitcode_lib_by_name_not_found_error_uses_requested_filename(monkeypatch, tmp_path):
    filename = "libdevice_sm_90.bc"
    cuda_home = tmp_path / "cuda-home"
    lib_dir = _bitcode_lib_dir_under(cuda_home, "device")
    lib_dir.mkdir(parents=True)
    (lib_dir / "libdevice.10.bc").touch()

    monkeypatch.setattr(
        find_bitcode_lib_module,
        "find_sub_dirs_all_sitepackages",
        lambda _sub_dir: [],
    )
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(BitcodeLibNotFoundError, match=rf'Failure finding "{filename}"') as exc_info:
        find_bitcode_lib_by_name("device", filename)

    message = str(exc_info.value)
    assert f"No such file: {lib_dir / filename}" in message
    assert "libdevice.10.bc" in message


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


@pytest.mark.parametrize(
    "filename",
    ("", ".", "..", "../file.bc", "subdir/file.bc", r"subdir\file.bc", r"C:\lib\file.bc", "bad\0name.bc"),
)
@pytest.mark.parametrize("lookup", (find_bitcode_lib_by_name, locate_bitcode_lib_by_name))
@pytest.mark.agent_authored(model="gpt-5")
def test_bitcode_lib_by_name_rejects_paths(lookup, filename):
    with pytest.raises(ValueError, match="without a directory"):
        lookup("device", filename)


@pytest.mark.parametrize("filename", (None, 90))
@pytest.mark.parametrize("lookup", (find_bitcode_lib_by_name, locate_bitcode_lib_by_name))
@pytest.mark.agent_authored(model="gpt-5")
def test_bitcode_lib_by_name_requires_string(lookup, filename):
    with pytest.raises(TypeError, match="filename must be a string"):
        lookup("device", filename)


@pytest.mark.parametrize("lookup", (find_bitcode_lib_by_name, locate_bitcode_lib_by_name))
@pytest.mark.agent_authored(model="gpt-5")
def test_bitcode_lib_by_name_invalid_project(lookup):
    with pytest.raises(ValueError, match="Unknown bitcode library"):
        lookup("not_a_real_lib", "libdevice_sm_90.bc")
