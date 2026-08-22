# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows dynamic-loader tests", allow_module_level=True)

from cuda.pathfinder._dynamic_libs import load_dl_windows
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS


@pytest.mark.parametrize(
    ("libname", "register_directory"),
    (("cudnn", True), ("cublasLt", False)),
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_already_loaded_library_registers_resolved_directory_by_descriptor_policy(
    mocker, tmp_path, libname, register_directory
):
    desc = LIB_DESCRIPTORS[libname]
    resolved_path = str(tmp_path / desc.windows_dlls[-1])
    handle = 0xBEEF
    mocker.patch.object(load_dl_windows.kernel32, "GetModuleHandleW", return_value=handle)
    mocker.patch.object(load_dl_windows, "abs_path_for_dynamic_library", return_value=resolved_path)
    add_dll_directory = mocker.patch.object(load_dl_windows, "add_dll_directory")

    loaded = load_dl_windows.check_if_already_loaded_from_elsewhere(desc)

    assert loaded is not None
    assert loaded.abs_path == resolved_path
    assert loaded.was_already_loaded_from_elsewhere
    assert loaded._handle_uint == handle
    assert loaded.found_via == "was-already-loaded-from-elsewhere"
    if register_directory:
        add_dll_directory.assert_called_once_with(resolved_path)
    else:
        add_dll_directory.assert_not_called()
