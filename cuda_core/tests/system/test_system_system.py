# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

import os

import pytest

try:
    from cuda.bindings import driver, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime

from cuda.core import Device, system
from cuda.core._utils.cuda_utils import handle_return

from .conftest import skip_if_nvml_unsupported


def test_driver_version():
    driver_version = system.get_driver_version()
    version = handle_return(driver.cuDriverGetVersion())
    expected_driver_version = (version // 1000, (version % 1000) // 10)
    assert driver_version == expected_driver_version, "Driver version does not match expected value"


def test_num_devices():
    num_devices = system.get_num_devices()
    expected_num_devices = handle_return(runtime.cudaGetDeviceCount())
    assert num_devices == expected_num_devices, "Number of devices does not match expected value"


def test_devices():
    devices = Device.get_all_devices()
    expected_num_devices = handle_return(runtime.cudaGetDeviceCount())
    expected_devices = tuple(Device(device_id) for device_id in range(expected_num_devices))
    assert len(devices) == len(expected_devices), "Number of devices does not match expected value"
    for device, expected_device in zip(devices, expected_devices):
        assert device.device_id == expected_device.device_id, "Device ID does not match expected value"


def test_cuda_driver_version():
    cuda_driver_version = system.get_driver_version_full()
    assert isinstance(cuda_driver_version, tuple)
    assert len(cuda_driver_version) == 3

    ver_maj, ver_min, ver_patch = cuda_driver_version
    assert ver_maj >= 10
    assert 0 <= ver_min <= 99
    assert 0 <= ver_patch <= 9


@skip_if_nvml_unsupported
def test_gpu_driver_version():
    driver_version = system.get_driver_version(kernel_mode=True)
    assert isinstance(driver_version, tuple)
    assert len(driver_version) in (2, 3)

    (ver_maj, ver_min, *ver_patch) = driver_version
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


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
    try:
        driver_branch = system.get_driver_branch()
    except (system.UnknownError, system.NotSupportedError):
        pytest.skip("Driver branch not supported on this system")
    assert isinstance(driver_branch, str)
    assert len(driver_branch) > 0
    assert driver_branch[0] == "r"
