# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

try:
    from cuda.bindings import driver, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime

from cuda.core.experimental import Device
from cuda.core.experimental._utils import handle_return, ComputeCapability

def test_device_set_current(deinit_cuda):
    device = Device()
    device.set_current()
    assert handle_return(driver.cuCtxGetCurrent()) is not None
    
def test_device_repr():
    device = Device(0)
    assert str(device).startswith('<Device 0')

def test_device_alloc(init_cuda):
    device = Device()
    buffer = device.allocate(1024)
    device.sync()
    assert buffer.handle != 0
    assert buffer.size == 1024
    assert buffer.device_id == 0

def test_device_create_stream(init_cuda):
    device = Device()
    stream = device.create_stream()
    assert stream is not None
    assert stream.handle

def test_pci_bus_id():
    device = Device()
    bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, device.device_id))
    assert device.pci_bus_id == bus_id[:12].decode()

def test_uuid():
    device = Device()
    driver_ver = handle_return(driver.cuDriverGetVersion())
    if driver_ver >= 11040:
        uuid = handle_return(driver.cuDeviceGetUuid_v2(device.device_id))
    else:
        uuid = handle_return(driver.cuDeviceGetUuid(device.device_id))
    uuid = uuid.bytes.hex()
    expected_uuid = f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"
    assert device.uuid == expected_uuid

def test_name():
    device = Device()
    name = handle_return(driver.cuDeviceGetName(128, device.device_id))
    name = name.split(b'\0')[0]
    assert device.name == name.decode()

def test_compute_capability():
    device = Device()
    major = handle_return(runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device.device_id))
    minor = handle_return(runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device.device_id))
    expected_cc = ComputeCapability(major, minor)
    assert device.compute_capability == expected_cc
 