# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import os

import pytest

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

from cuda.core import system
from cuda.core._utils.cuda_utils import handle_return

from .conftest import skip_if_nvml_unsupported


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
