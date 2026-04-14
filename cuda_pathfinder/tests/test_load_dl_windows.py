# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only tests", allow_module_level=True)

from cuda.pathfinder._dynamic_libs import load_dl_windows
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS


def test_check_if_already_loaded_falls_back_to_enumerated_modules(tmp_path, mocker):
    desc = LIB_DESCRIPTORS["cupti"]
    expected_path = tmp_path / desc.windows_dlls[0]
    handles = (0x111, 0x222)

    mocker.patch.object(load_dl_windows.kernel32, "GetModuleHandleW", return_value=0)
    mocker.patch.object(load_dl_windows, "_iter_loaded_module_handles", return_value=iter(handles))
    mocker.patch.object(
        load_dl_windows,
        "abs_path_for_dynamic_library",
        side_effect=(
            r"C:\Windows\System32\kernel32.dll",
            str(expected_path),
        ),
    )
    add_dll_directory = mocker.patch.object(load_dl_windows, "add_dll_directory")

    result = load_dl_windows.check_if_already_loaded_from_elsewhere(desc, have_abs_path=False)

    assert result is not None
    assert result.abs_path == str(expected_path)
    assert result.was_already_loaded_from_elsewhere is True
    assert result.found_via == "was-already-loaded-from-elsewhere"
    assert result._handle_uint == handles[1]
    add_dll_directory.assert_not_called()


def test_check_if_already_loaded_fallback_preserves_add_dll_directory_side_effect(tmp_path, mocker):
    desc = LIB_DESCRIPTORS["nvrtc"]
    expected_path = tmp_path / desc.windows_dlls[0]

    mocker.patch.object(load_dl_windows.kernel32, "GetModuleHandleW", return_value=0)
    mocker.patch.object(load_dl_windows, "_iter_loaded_module_handles", return_value=iter((0x333,)))
    mocker.patch.object(load_dl_windows, "abs_path_for_dynamic_library", return_value=str(expected_path))
    add_dll_directory = mocker.patch.object(load_dl_windows, "add_dll_directory")

    result = load_dl_windows.check_if_already_loaded_from_elsewhere(desc, have_abs_path=True)

    assert result is not None
    assert result.abs_path == str(expected_path)
    assert result.was_already_loaded_from_elsewhere is True
    add_dll_directory.assert_called_once_with(str(expected_path))
