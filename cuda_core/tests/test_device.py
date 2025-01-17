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


def test_device_property_types():
    device = Device()
    assert isinstance(device.properties.name, str)
    assert isinstance(device.properties.uuid, bytes)
    assert isinstance(device.properties.total_global_mem, int)
    assert isinstance(device.properties.shared_mem_per_block, int)
    assert isinstance(device.properties.regs_per_block, int)
    assert isinstance(device.properties.warp_size, int)
    assert isinstance(device.properties.mem_pitch, int)
    assert isinstance(device.properties.max_threads_per_block, int)
    assert isinstance(device.properties.max_threads_dim, tuple)
    assert isinstance(device.properties.max_grid_size, tuple)
    assert isinstance(device.properties.clock_rate, int)
    assert isinstance(device.properties.total_const_mem, int)
    assert isinstance(device.properties.major, int)
    assert isinstance(device.properties.minor, int)
    assert isinstance(device.properties.texture_alignment, int)
    assert isinstance(device.properties.texture_pitch_alignment, int)
    assert isinstance(device.properties.device_overlap, int)
    assert isinstance(device.properties.multi_processor_count, int)
    assert isinstance(device.properties.kernel_exec_timeout_enabled, int)
    assert isinstance(device.properties.integrated, int)
    assert isinstance(device.properties.can_map_host_memory, int)
    assert isinstance(device.properties.compute_mode, int)
    assert isinstance(device.properties.max_texture_1d, int)
    assert isinstance(device.properties.max_texture_1d_mipmap, int)
    assert isinstance(device.properties.max_texture_1d_linear, int)
    assert isinstance(device.properties.max_texture_2d, tuple)
    assert isinstance(device.properties.max_texture_2d_mipmap, tuple)
    assert isinstance(device.properties.max_texture_2d_linear, tuple)
    assert isinstance(device.properties.max_texture_2d_gather, tuple)
    assert isinstance(device.properties.max_texture_3d, tuple)
    assert isinstance(device.properties.max_texture_3d_alt, tuple)
    assert isinstance(device.properties.max_texture_cubemap, int)
    assert isinstance(device.properties.max_texture_1d_layered, tuple)
    assert isinstance(device.properties.max_texture_2d_layered, tuple)
    assert isinstance(device.properties.max_texture_cubemap_layered, tuple)
    assert isinstance(device.properties.max_surface_1d, int)
    assert isinstance(device.properties.max_surface_2d, tuple)
    assert isinstance(device.properties.max_surface_3d, tuple)
    assert isinstance(device.properties.max_surface_1d_layered, tuple)
    assert isinstance(device.properties.max_surface_2d_layered, tuple)
    assert isinstance(device.properties.max_surface_cubemap, int)
    assert isinstance(device.properties.max_surface_cubemap_layered, tuple)
    assert isinstance(device.properties.surface_alignment, int)
    assert isinstance(device.properties.concurrent_kernels, int)
    assert isinstance(device.properties.ecc_enabled, int)
    assert isinstance(device.properties.pci_bus_id, int)
    assert isinstance(device.properties.pci_device_id, int)
    assert isinstance(device.properties.pci_domain_id, int)
    assert isinstance(device.properties.tcc_driver, int)
    assert isinstance(device.properties.async_engine_count, int)
    assert isinstance(device.properties.unified_addressing, int)
    assert isinstance(device.properties.memory_clock_rate, int)
    assert isinstance(device.properties.memory_bus_width, int)
    assert isinstance(device.properties.l2_cache_size, int)
    assert isinstance(device.properties.persisting_l2_cache_max_size, int)
    assert isinstance(device.properties.max_threads_per_multi_processor, int)
    assert isinstance(device.properties.stream_priorities_supported, int)
    assert isinstance(device.properties.global_l1_cache_supported, int)
    assert isinstance(device.properties.local_l1_cache_supported, int)
    assert isinstance(device.properties.shared_mem_per_multiprocessor, int)
    assert isinstance(device.properties.regs_per_multiprocessor, int)
    assert isinstance(device.properties.managed_memory, int)
    assert isinstance(device.properties.is_multi_gpu_board, int)
    assert isinstance(device.properties.multi_gpu_board_group_id, int)
    assert isinstance(device.properties.single_to_double_precision_perf_ratio, int)
    assert isinstance(device.properties.pageable_memory_access, int)
    assert isinstance(device.properties.concurrent_managed_access, int)
    assert isinstance(device.properties.compute_preemption_supported, int)
    assert isinstance(device.properties.can_use_host_pointer_for_registered_mem, int)
    assert isinstance(device.properties.cooperative_launch, int)
    assert isinstance(device.properties.cooperative_multi_device_launch, int)
    assert isinstance(device.properties.pageable_memory_access_uses_host_page_tables, int)
    assert isinstance(device.properties.direct_managed_mem_access_from_host, int)
    assert isinstance(device.properties.access_policy_max_window_size, int)
    assert isinstance(device.properties.reserved_shared_mem_per_block, int)
    assert isinstance(device.properties.host_register_supported, int)
    assert isinstance(device.properties.sparse_cuda_array_supported, int)
    assert isinstance(device.properties.host_register_read_only_supported, int)
    assert isinstance(device.properties.timeline_semaphore_interop_supported, int)
    assert isinstance(device.properties.memory_pools_supported, int)
    assert isinstance(device.properties.gpu_direct_rdma_supported, int)
    assert isinstance(device.properties.gpu_direct_rdma_flush_writes_options, int)
    assert isinstance(device.properties.gpu_direct_rdma_writes_ordering, int)
    assert isinstance(device.properties.memory_pool_supported_handle_types, int)
    assert isinstance(device.properties.deferred_mapping_cuda_array_supported, int)
    assert isinstance(device.properties.ipc_event_supported, int)
    assert isinstance(device.properties.unified_function_pointers, int)
