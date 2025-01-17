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
import pytest

from cuda.core.experimental import Device
from cuda.core.experimental._utils import ComputeCapability, handle_return


def test_device_set_current(deinit_cuda):
    device = Device()
    device.set_current()
    assert handle_return(driver.cuCtxGetCurrent()) is not None


def test_device_repr():
    device = Device(0)
    assert str(device).startswith("<Device 0")


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
    name = name.split(b"\0")[0]
    assert device.name == name.decode()


def test_compute_capability():
    device = Device()
    major = handle_return(
        runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device.device_id)
    )
    minor = handle_return(
        runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device.device_id)
    )
    expected_cc = ComputeCapability(major, minor)
    assert device.compute_capability == expected_cc


def test_device_property_values():
    device = Device()
    assert device.properties.name == device.name
    assert device.properties.uuid.hex() == device.uuid.replace("-", "")


cuda_base_properties = [
    ("name", str),
    ("uuid", bytes),
    ("total_global_mem", int),
    ("shared_mem_per_block", int),
    ("regs_per_block", int),
    ("warp_size", int),
    ("mem_pitch", int),
    ("max_threads_per_block", int),
    ("max_threads_dim", tuple),
    ("max_grid_size", tuple),
    ("total_const_mem", int),
    ("major", int),
    ("minor", int),
    ("texture_alignment", int),
    ("texture_pitch_alignment", int),
    ("multi_processor_count", int),
    ("integrated", int),
    ("can_map_host_memory", int),
    ("max_texture_1d", int),
    ("max_texture_1d_mipmap", int),
    ("max_texture_2d", tuple),
    ("max_texture_2d_mipmap", tuple),
    ("max_texture_2d_linear", tuple),
    ("max_texture_2d_gather", tuple),
    ("max_texture_3d", tuple),
    ("max_texture_3d_alt", tuple),
    ("max_texture_cubemap", int),
    ("max_texture_1d_layered", tuple),
    ("max_texture_2d_layered", tuple),
    ("max_texture_cubemap_layered", tuple),
    ("max_surface_1d", int),
    ("max_surface_2d", tuple),
    ("max_surface_3d", tuple),
    ("max_surface_1d_layered", tuple),
    ("max_surface_2d_layered", tuple),
    ("max_surface_cubemap", int),
    ("max_surface_cubemap_layered", tuple),
    ("surface_alignment", int),
    ("concurrent_kernels", int),
    ("ecc_enabled", int),
    ("pci_bus_id", int),
    ("pci_device_id", int),
    ("pci_domain_id", int),
    ("tcc_driver", int),
    ("async_engine_count", int),
    ("unified_addressing", int),
    ("memory_bus_width", int),
    ("l2_cache_size", int),
    ("persisting_l2_cache_max_size", int),
    ("max_threads_per_multi_processor", int),
    ("stream_priorities_supported", int),
    ("global_l1_cache_supported", int),
    ("local_l1_cache_supported", int),
    ("shared_mem_per_multiprocessor", int),
    ("regs_per_multiprocessor", int),
    ("managed_memory", int),
    ("is_multi_gpu_board", int),
    ("multi_gpu_board_group_id", int),
    ("pageable_memory_access", int),
    ("concurrent_managed_access", int),
    ("compute_preemption_supported", int),
    ("can_use_host_pointer_for_registered_mem", int),
    ("cooperative_launch", int),
    ("pageable_memory_access_uses_host_page_tables", int),
    ("direct_managed_mem_access_from_host", int),
    ("access_policy_max_window_size", int),
    ("reserved_shared_mem_per_block", int),
    ("host_register_supported", int),
    ("sparse_cuda_array_supported", int),
    ("host_register_read_only_supported", int),
    ("timeline_semaphore_interop_supported", int),
    ("memory_pools_supported", int),
    ("gpu_direct_rdma_supported", int),
    ("gpu_direct_rdma_flush_writes_options", int),
    ("gpu_direct_rdma_writes_ordering", int),
    ("memory_pool_supported_handle_types", int),
    ("deferred_mapping_cuda_array_supported", int),
    ("ipc_event_supported", int),
    ("unified_function_pointers", int),
]

cuda_12_properties = [
    ("host_native_atomic_supported", int),
    ("luid", bytes),
    ("luid_device_node_mask", int),
    ("max_blocks_per_multi_processor", int),
    ("shared_mem_per_block_optin", int),
    ("cluster_launch", int),
]


driver_ver = handle_return(driver.cuDriverGetVersion())
if driver_ver >= 12000:
    cuda_base_properties += cuda_12_properties


@pytest.mark.parametrize("property_name, expected_type", cuda_base_properties)
def test_device_property_types(property_name, expected_type):
    device = Device()
    assert isinstance(getattr(device.properties, property_name), expected_type)
