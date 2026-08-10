# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import sys

import pytest
from cuda_python_test_helpers.arch_check import skip_if_nvml_unsupported

from cuda.bindings import driver
from cuda.core import system
from cuda.core._utils.cuda_utils import handle_return


def test_user_mode_driver_version():
    umd = system.get_user_mode_driver_version()
    assert isinstance(umd, tuple)
    assert len(umd) == 2
    version = handle_return(driver.cuDriverGetVersion())
    expected = (version // 1000, (version % 1000) // 10)
    assert umd == expected, "UMD driver version does not match expected value"


@skip_if_nvml_unsupported
def test_kernel_mode_driver_version():
    kmd = system.get_kernel_mode_driver_version()
    assert isinstance(kmd, tuple)
    assert len(kmd) in (2, 3)
    ver_maj, ver_min, *ver_patch = kmd
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


def test_kernel_mode_driver_version_requires_nvml():
    if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        pytest.skip("NVML is available, cannot test the error path")
    with pytest.raises(RuntimeError, match="requires NVML support"):
        system.get_kernel_mode_driver_version()


@skip_if_nvml_unsupported
def test_nvml_version():
    nvml_version = system.get_nvml_version()
    assert isinstance(nvml_version, tuple)
    assert len(nvml_version) in (3, 4)

    (cuda_ver_maj, ver_maj, ver_min, *ver_patch) = nvml_version
    assert cuda_ver_maj >= 10
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


@skip_if_nvml_unsupported
def test_get_process_name():
    for device in system.Device.get_all_devices():
        x = device.compute_running_processes

    try:
        process_name = system.get_process_name(os.getpid())
    except system.NotFoundError:
        pytest.skip("Process not found")

    assert isinstance(process_name, str)
    assert "python" in process_name


def test_device_count():
    device_count = system.get_num_devices()
    assert isinstance(device_count, int)
    assert device_count >= 0


@skip_if_nvml_unsupported
def test_get_driver_branch():
    driver_branch = system.get_driver_branch()
    assert isinstance(driver_branch, str)
    assert len(driver_branch) > 0


# The NVML-unavailable fallback is decided at import time, so it can only be
# exercised in a fresh interpreter with cuda.bindings.nvml blocked.
_NO_NVML_SCRIPT = """
import sys


class _BlockNvml:
    def find_spec(self, name, path=None, target=None):
        if name == "cuda.bindings.nvml":
            raise ImportError("blocked for testing", name=name)
        return None


sys.meta_path.insert(0, _BlockNvml())

# Used to raise ImportError out of this import: the flag was cleared, but the
# `else` that binds the non-NVML fallbacks belonged to the outer `if`, which
# had already been evaluated, and _nvml_context (imported unconditionally
# right after) imports cuda.bindings.nvml itself.
from cuda.core import system
from cuda.core.system import _system

assert system.CUDA_BINDINGS_NVML_IS_COMPATIBLE is False, "flag must be cleared when nvml is unimportable"
for name in ("driver", "handle_return", "runtime"):
    assert hasattr(_system, name), f"non-NVML fallback {name!r} is not bound"
print("ok")
"""


@pytest.mark.agent_authored(model="claude-opus-5")
def test_system_falls_back_when_nvml_is_unimportable():
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _NO_NVML_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert proc.stdout.strip().endswith("ok")
