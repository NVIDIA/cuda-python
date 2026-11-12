# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.platform != "linux":
    pytest.skip("Linux dynamic-loader tests", allow_module_level=True)

from child_load_nvidia_dynamic_lib_helper import (
    LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD,
    build_child_process_failed_for_libname_message,
    run_load_nvidia_dynamic_lib_in_subprocess,
)

from cuda.pathfinder._dynamic_libs import load_dl_linux
from cuda.pathfinder._dynamic_libs.descriptor_catalog import DescriptorSpec
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    STATUS_NOT_FOUND,
    parse_dynamic_lib_subprocess_payload,
)

_NVRTC_COMPILE_CHILD_SCRIPT = Path(__file__).with_name("child_nvrtc_compile_helper.py")


def _descriptor() -> DescriptorSpec:
    return DescriptorSpec(
        name="probe",
        packaged_with="other",
        linux_sonames=("libprobe.so.13", "libprobe.so.12"),
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_already_loaded_library_checks_declared_sonames_in_preference_order(mocker):
    queried_sonames: list[tuple[str, int]] = []

    def cdll(soname, mode):
        queried_sonames.append((soname, mode))
        raise OSError

    mocker.patch.object(load_dl_linux.ctypes, "CDLL", side_effect=cdll)

    loaded = load_dl_linux.check_if_already_loaded_from_elsewhere(_descriptor())

    assert loaded is None
    assert queried_sonames == [
        ("libprobe.so.13", os.RTLD_NOLOAD),
        ("libprobe.so.12", os.RTLD_NOLOAD),
    ]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_system_search_checks_declared_sonames_in_preference_order(mocker):
    queried_sonames: list[str] = []

    def load_lib(_desc, soname):
        queried_sonames.append(soname)
        raise OSError

    mocker.patch.object(load_dl_linux, "_load_lib", side_effect=load_lib)

    loaded = load_dl_linux.load_with_system_search(_descriptor())

    assert loaded is None
    assert queried_sonames == ["libprobe.so.13", "libprobe.so.12"]


def _make_nvrtc_dir(tmp_path, nvrtc_file, *builtins_files):
    # Mirror real layouts: CTK and Conda name the real file libnvrtc.so.<version>
    # behind a libnvrtc.so.<major> symlink; wheels ship a physical libnvrtc.so.<major>.
    major = nvrtc_file.split(".")[2]
    (tmp_path / nvrtc_file).touch()
    soname_path = tmp_path / f"libnvrtc.so.{major}"
    if nvrtc_file != soname_path.name:
        soname_path.symlink_to(nvrtc_file)
    for name in builtins_files:
        (tmp_path / name).touch()
    return str(soname_path)


@pytest.mark.agent_authored(model="claude-opus-5.5")
@pytest.mark.parametrize(
    ("found_via", "nvrtc_file", "builtins_file", "expect_preload"),
    [
        ("CUDA_PATH", "libnvrtc.so.13.5.37", "libnvrtc-builtins.so.13.5", True),
        ("system-ctk-root", "libnvrtc.so.12.9.86", "libnvrtc-builtins.so.12.9", True),
        ("site-packages", "libnvrtc.so.13", "libnvrtc-builtins.so.13.0", True),
        ("site-packages", "libnvrtc.so.13", "libnvrtc-builtins.so.13.5", False),
        ("conda", "libnvrtc.so.13.5.37", "libnvrtc-builtins.so.13.5", False),
    ],
    ids=["ctk-13.5", "ctk-12.9", "wheel-13.0", "wheel-13.5", "conda-13.5"],
)
def test_preload_nvrtc_builtins_policy(mocker, tmp_path, found_via, nvrtc_file, builtins_file, expect_preload):
    nvrtc_path = _make_nvrtc_dir(tmp_path, nvrtc_file, builtins_file)
    cdll = mocker.patch.object(load_dl_linux.ctypes, "CDLL")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_dl_linux._preload_nvrtc_builtins(nvrtc_path, found_via)

    if expect_preload:
        cdll.assert_called_once_with(str(tmp_path / builtins_file), load_dl_linux.CDLL_MODE)
    else:
        cdll.assert_not_called()


@pytest.mark.agent_authored(model="claude-opus-5.5")
@pytest.mark.parametrize(
    ("found_via", "nvrtc_file", "builtins_files", "expected_warning"),
    [
        ("CUDA_PATH", "libnvrtc.so.13.5.37", (), r"NVRTC builtins sibling not found: .*libnvrtc-builtins\.so\.13\.5$"),
        ("site-packages", "libnvrtc.so.13", (), None),
        ("CUDA_PATH", "libnvrtc.so.13", ("libnvrtc-builtins.so.13.4",), r"Cannot determine the NVRTC version from "),
    ],
    ids=["ctk-missing-sibling", "wheel-missing-sibling", "ctk-unversioned-file"],
)
def test_preload_nvrtc_builtins_skips(mocker, tmp_path, found_via, nvrtc_file, builtins_files, expected_warning):
    nvrtc_path = _make_nvrtc_dir(tmp_path, nvrtc_file, *builtins_files)
    cdll = mocker.patch.object(load_dl_linux.ctypes, "CDLL")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_dl_linux._preload_nvrtc_builtins(nvrtc_path, found_via)

    cdll.assert_not_called()
    messages = [str(w.message) for w in caught]
    if expected_warning is None:
        assert messages == []
    else:
        assert len(messages) == 1
        assert re.search(expected_warning, messages[0]), messages[0]


@pytest.mark.agent_authored(model="claude-opus-5.5")
def test_load_with_abs_path_warns_and_returns_nvrtc_if_builtins_preload_fails(mocker, tmp_path):
    nvrtc_path = _make_nvrtc_dir(tmp_path, "libnvrtc.so.13.5.37", "libnvrtc-builtins.so.13.5")
    mocker.patch.object(load_dl_linux, "_load_lib", return_value=SimpleNamespace(_handle=1234))
    mocker.patch.object(load_dl_linux.ctypes, "CDLL", side_effect=OSError("bad ELF"))

    with pytest.warns(
        RuntimeWarning, match=r"Failed to preload .*libnvrtc-builtins\.so\.13\.5 for .*libnvrtc\.so\.13: bad ELF"
    ):
        loaded = load_dl_linux.load_with_abs_path(LIB_DESCRIPTORS["nvrtc"], nvrtc_path, "CUDA_PATH")

    assert loaded.abs_path == nvrtc_path


@pytest.mark.agent_authored(model="claude-opus-5.5")
def test_nvrtc_copy_without_loader_path_compiles_using_its_own_builtins(tmp_path):
    # Copy NVRTC and its builtins to a directory that is not on any loader
    # search path. Asserting where the builtins were mapped from (not only that
    # compilation succeeded) keeps the test meaningful on hosts where ldconfig
    # also knows a libnvrtc-builtins of the same version.
    timeout = 30
    result = run_load_nvidia_dynamic_lib_in_subprocess("nvrtc", timeout=timeout)
    assert result.returncode == 0, build_child_process_failed_for_libname_message("nvrtc", result)
    payload = parse_dynamic_lib_subprocess_payload(
        result.stdout, libname="nvrtc", error_label="Load subprocess child process"
    )
    if payload.status == STATUS_NOT_FOUND:
        pytest.skip("nvrtc not found")
    assert payload.abs_path is not None
    src_dir, nvrtc_basename = os.path.split(payload.abs_path)
    nvrtc_real_basename = os.path.basename(os.path.realpath(payload.abs_path))
    if not re.fullmatch(r"libnvrtc\.so\.\d+\.\d+.*", nvrtc_real_basename):
        pytest.skip(f"Not a CTK-style NVRTC (no versioned real file): {payload.abs_path}")
    builtins_basenames = [
        name for name in os.listdir(src_dir) if re.fullmatch(r"libnvrtc-builtins\.so\.\d+\.\d+", name)
    ]
    if len(builtins_basenames) != 1:
        pytest.skip(f"Expected exactly one libnvrtc-builtins.so.<major>.<minor> in {src_dir}: {builtins_basenames}")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    shutil.copy(os.path.realpath(payload.abs_path), lib_dir / nvrtc_real_basename)
    (lib_dir / nvrtc_basename).symlink_to(nvrtc_real_basename)
    shutil.copy(os.path.join(src_dir, builtins_basenames[0]), lib_dir / builtins_basenames[0])
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    child = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + internal test helper script
        [sys.executable, str(_NVRTC_COMPILE_CHILD_SCRIPT), str(lib_dir / nvrtc_basename)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD,
        env=env,
    )
    assert child.returncode == 0, build_child_process_failed_for_libname_message("nvrtc", child)

    report = json.loads(child.stdout.splitlines()[-1])
    assert report["compile_status"] == 0, report["log"]
    assert report["builtins_paths"] == [os.path.realpath(lib_dir / builtins_basenames[0])]
