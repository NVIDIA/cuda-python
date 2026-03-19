# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _load_windows_module():
    if not IS_WINDOWS:
        pytest.skip("Windows-specific loader tests")
    from cuda.pathfinder._dynamic_libs import load_dl_windows as mod

    return mod


def _make_loaded_dl(path, found_via="system-search"):
    return LoadedDL(path, False, 0xDEAD, found_via)


def test_find_dll_on_env_path_ignores_current_directory(tmp_path, monkeypatch):
    mod = _load_windows_module()

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    path_dir = tmp_path / "path_dir"
    path_dir.mkdir()

    dll_name = "fakecuda.dll"
    (cwd_dir / dll_name).write_bytes(b"cwd-copy")
    expected = path_dir / dll_name
    expected.write_bytes(b"path-copy")

    monkeypatch.chdir(cwd_dir)
    monkeypatch.setenv("PATH", os.pathsep.join((".", f'"{path_dir}"')))

    assert mod._find_dll_on_env_path(dll_name) == str(expected)


def test_env_path_fallback_uses_load_with_abs_path(tmp_path, monkeypatch, mocker):
    mod = _load_windows_module()
    desc = LIB_DESCRIPTORS["nvrtc"]
    dll_name = desc.windows_dlls[-1]

    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    dll_path = path_dir / dll_name
    dll_path.write_bytes(b"fake-dll")

    monkeypatch.setenv("PATH", str(path_dir))
    expected = _make_loaded_dl(str(dll_path))
    load_with_abs_path = mocker.patch.object(mod, "load_with_abs_path", return_value=expected)

    result = mod._try_load_with_env_path_fallback(desc, dll_name)

    assert result is expected
    load_with_abs_path.assert_called_once_with(desc, str(dll_path), "system-search")


def test_load_with_system_search_prefers_process_dll_search_over_env_path(mocker):
    mod = _load_windows_module()
    desc = LIB_DESCRIPTORS["nvrtc"]
    expected = _make_loaded_dl(r"C:\CUDA\bin\nvrtc64_130_0.dll")

    process_search = mocker.patch.object(mod, "_try_load_with_process_dll_search", return_value=expected)
    env_path = mocker.patch.object(mod, "_try_load_with_env_path_fallback")

    result = mod.load_with_system_search(desc)

    assert result is expected
    process_search.assert_called_once_with(desc, desc.windows_dlls[-1])
    env_path.assert_not_called()


def test_load_with_system_search_skips_env_path_fallback_for_driver_libs(mocker):
    mod = _load_windows_module()
    desc = LIB_DESCRIPTORS["cuda"]

    process_search = mocker.patch.object(mod, "_try_load_with_process_dll_search", return_value=None)
    env_path = mocker.patch.object(mod, "_try_load_with_env_path_fallback")

    result = mod.load_with_system_search(desc)

    assert result is None
    assert process_search.call_count == len(desc.windows_dlls)
    env_path.assert_not_called()
