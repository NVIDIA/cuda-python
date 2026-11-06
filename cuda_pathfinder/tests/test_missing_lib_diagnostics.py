# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

import cuda.pathfinder._static_libs.find_bitcode_lib as bitcode_module
import cuda.pathfinder._static_libs.find_static_lib as static_module
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home


@pytest.fixture(autouse=True)
def clear_lookup_caches():
    get_cuda_path_or_home.cache_clear()
    bitcode_module.find_bitcode_lib.cache_clear()
    static_module.find_static_lib.cache_clear()
    yield
    get_cuda_path_or_home.cache_clear()
    bitcode_module.find_bitcode_lib.cache_clear()
    static_module.find_static_lib.cache_clear()


def _missing_lookup(monkeypatch, tmp_path, kind, api):
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    root = tmp_path / "toolkit"
    monkeypatch.setenv("CUDA_HOME", str(root))
    if kind == "bitcode":
        module = bitcode_module
        name = "device"
        info = module._SUPPORTED_BITCODE_LIBS_INFO[name]
        relative_dir = info["rel_path"]
        expected_error = module.BitcodeLibNotFoundError
        lookup = module.find_bitcode_lib if api == "find" else module.locate_bitcode_lib
    else:
        module = static_module
        name = "cudadevrt"
        info = module._SUPPORTED_STATIC_LIBS_INFO[name]
        relative_dir = info["ctk_rel_paths"][0]
        expected_error = module.StaticLibNotFoundError
        lookup = module.find_static_lib if api == "find" else module.locate_static_lib

    monkeypatch.setattr(module, "find_sub_dirs_all_sitepackages", lambda _sub_dir: [])
    directory = root / relative_dir
    directory.mkdir(parents=True)
    return directory, info["filename"], lookup, name, expected_error


@pytest.mark.parametrize("kind", ("bitcode", "static"))
@pytest.mark.parametrize("api", ("find", "locate"))
@pytest.mark.parametrize("error_type", (PermissionError, FileNotFoundError))
@pytest.mark.agent_authored(model="gpt-6")
def test_listing_os_error_preserves_not_found_exception(monkeypatch, tmp_path, kind, api, error_type):
    directory, filename, lookup, name, expected_error = _missing_lookup(monkeypatch, tmp_path, kind, api)
    real_iterdir = Path.iterdir

    def failing_iterdir(path):
        if path == directory:
            raise error_type("simulated listing failure")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    with pytest.raises(expected_error) as exc_info:
        lookup(name)

    message = str(exc_info.value)
    assert str(directory / filename) in message
    assert error_type.__name__ in message
    assert "simulated listing failure" in message


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permission bits")
@pytest.mark.parametrize("kind", ("bitcode", "static"))
@pytest.mark.parametrize("api", ("find", "locate"))
@pytest.mark.agent_authored(model="gpt-6")
def test_unreadable_directory_preserves_not_found_exception(monkeypatch, tmp_path, kind, api):
    directory, filename, lookup, name, expected_error = _missing_lookup(monkeypatch, tmp_path, kind, api)
    directory.chmod(0o111)
    try:
        try:
            next(directory.iterdir(), None)
        except PermissionError:
            pass
        else:
            pytest.skip("the current user can list a directory without read permission")

        with pytest.raises(expected_error) as exc_info:
            lookup(name)

        message = str(exc_info.value)
        assert str(directory / filename) in message
        assert "PermissionError" in message
    finally:
        directory.chmod(0o700)


@pytest.mark.parametrize("kind", ("bitcode", "static"))
@pytest.mark.agent_authored(model="gpt-6")
def test_non_os_listing_error_is_not_suppressed(monkeypatch, tmp_path, kind):
    directory, _filename, lookup, name, _expected_error = _missing_lookup(monkeypatch, tmp_path, kind, "locate")
    real_iterdir = Path.iterdir

    def failing_iterdir(path):
        if path == directory:
            raise RuntimeError("unexpected listing failure")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    with pytest.raises(RuntimeError, match="unexpected listing failure"):
        lookup(name)
