# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

from child_load_nvidia_dynamic_lib_helper import (
    LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD,
    LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE,
    LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE,
    PROCESS_TIMED_OUT,
    run_load_nvidia_dynamic_lib_in_subprocess,
)

from cuda.pathfinder._dynamic_libs import dynamic_lib_subprocess as subprocess_mod
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError

_HELPER_MODULE = "child_load_nvidia_dynamic_lib_helper"


def test_run_load_nvidia_dynamic_lib_in_subprocess_invokes_dedicated_module(mocker):
    result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"status": "ok", "abs_path": "/tmp/libcudart.so.13"}\n',
        stderr="",
    )
    run_mock = mocker.patch(f"{_HELPER_MODULE}.subprocess.run", return_value=result)

    assert run_load_nvidia_dynamic_lib_in_subprocess("cudart", timeout=12.5) is result
    run_mock.assert_called_once_with(
        [
            sys.executable,
            "-m",
            LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE,
            LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE,
            "cudart",
        ],
        capture_output=True,
        text=True,
        timeout=12.5,
        check=False,
        cwd=LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_CWD,
    )


def test_run_load_nvidia_dynamic_lib_in_subprocess_returns_timeout_result(mocker):
    mocker.patch(
        f"{_HELPER_MODULE}.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=3.0),
    )

    result = run_load_nvidia_dynamic_lib_in_subprocess("nvvm", timeout=3.0)

    assert result.args == [
        sys.executable,
        "-m",
        LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODULE,
        LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE,
        "nvvm",
    ]
    assert result.returncode == PROCESS_TIMED_OUT
    assert result.stdout == ""
    assert result.stderr == "Process timed out after 3.0 seconds and was terminated."


def test_probe_load_nvidia_dynamic_lib_and_print_json(mocker, capsys):
    mocker.patch.object(subprocess_mod, "_load_nvidia_dynamic_lib_for_test", return_value="/usr/lib/libcudart.so.13")

    subprocess_mod.probe_dynamic_lib_and_print_json("cudart", LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ok", "abs_path": "/usr/lib/libcudart.so.13"}
    assert captured.err == ""


def test_probe_load_nvidia_dynamic_lib_and_prints_not_found_payload(mocker, capsys):
    mocker.patch.object(
        subprocess_mod,
        "_load_nvidia_dynamic_lib_for_test",
        side_effect=DynamicLibNotFoundError("not found"),
    )

    subprocess_mod.probe_dynamic_lib_and_print_json("cudart", LOAD_NVIDIA_DYNAMIC_LIB_SUBPROCESS_MODE)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "not-found", "abs_path": None}
    assert captured.err == ""
