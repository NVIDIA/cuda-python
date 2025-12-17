# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime

from cuda.core.experimental import Device, system
from cuda.core.experimental._utils.cuda_utils import handle_return


def test_system_singleton():
    system1 = system
    system2 = system
    assert id(system1) == id(system2), "system is not a singleton"


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
