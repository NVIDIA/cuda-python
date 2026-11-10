# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

if sys.platform != "win32":
    pytest.skip("Windows dynamic-loader tests", allow_module_level=True)

from cuda.pathfinder._dynamic_libs import load_dl_windows
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_system_search_finds_dll_exposed_only_through_path(tmp_path, info_summary_append):
    dll_name = f"cuda_pathfinder_probe_{uuid.uuid4().hex}.dll"
    dll_dir = tmp_path / "path_only"
    dll_dir.mkdir()
    dll_path = dll_dir / dll_name
    shutil.copyfile(Path(os.environ["SYSTEMROOT"], "System32", "version.dll"), dll_path)
    probe = """\
import json
import sys
from dataclasses import replace

from cuda.pathfinder._dynamic_libs import load_dl_windows
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS

desc = replace(LIB_DESCRIPTORS["cublasLt"], windows_dlls=(sys.argv[1],))
loaded = load_dl_windows.load_with_system_search(desc)
print(json.dumps(None if loaded is None else {
    "abs_path": loaded.abs_path,
    "found_via": loaded.found_via,
}))
"""

    def run_probe(env):
        result = subprocess.run(  # noqa: S603 - trusted interpreter, test-owned script and DLL filename
            [sys.executable, "-c", probe, dll_name],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=30,
        )
        return json.loads(result.stdout)

    env = os.environ.copy()
    assert run_probe(env) is None
    env["PATH"] = os.pathsep.join((str(dll_dir), env.get("PATH", "")))
    loaded = run_probe(env)

    assert loaded["found_via"] == "system-search"
    assert os.path.samefile(loaded["abs_path"], dll_path)
    info_summary_append(f"PATH-only DLL: abs_path={loaded['abs_path']!r}")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_already_loaded_library_checks_known_dlls_in_preference_order(mocker):
    desc = LIB_DESCRIPTORS["cublasLt"]
    preferred_dll = desc.windows_dlls[0]
    fallback_dll = desc.windows_dlls[-1]
    queried_dlls: list[str] = []
    handle = 0xBEEF

    def get_module_handle(dll_name):
        queried_dlls.append(dll_name)
        return handle if dll_name == fallback_dll else 0

    mocker.patch.object(load_dl_windows.kernel32, "GetModuleHandleW", side_effect=get_module_handle)
    mocker.patch.object(
        load_dl_windows,
        "abs_path_for_dynamic_library",
        return_value=rf"C:\CUDA\bin\{fallback_dll}",
    )

    loaded = load_dl_windows.check_if_already_loaded_from_elsewhere(desc)

    assert loaded is not None
    assert queried_dlls == list(desc.windows_dlls)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_system_search_checks_known_dlls_in_preference_order(mocker):
    desc = LIB_DESCRIPTORS["cublasLt"]
    queried_dlls: list[str] = []

    def load_library(dll_name, _file, _flags):
        queried_dlls.append(dll_name)
        return 0

    mocker.patch.object(load_dl_windows.kernel32, "LoadLibraryExW", side_effect=load_library)

    loaded = load_dl_windows.load_with_system_search(desc)

    assert loaded is None
    assert queried_dlls == list(desc.windows_dlls)


@pytest.mark.parametrize(
    ("libname", "register_directory"),
    (("cudnn", True), ("cublasLt", False)),
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_already_loaded_library_registers_resolved_directory_by_descriptor_policy(
    mocker, tmp_path, libname, register_directory
):
    desc = LIB_DESCRIPTORS[libname]
    resolved_path = str(tmp_path / desc.windows_dlls[0])
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
