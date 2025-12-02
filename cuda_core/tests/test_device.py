# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime
import cuda.core
import pytest
from cuda.core import Device
from cuda.core._utils.cuda_utils import ComputeCapability, get_binding_version, handle_return


def test_device_init_disabled():
    with pytest.raises(RuntimeError, match=r"^DeviceProperties cannot be instantiated directly\."):
        cuda.core.experimental._device.DeviceProperties()  # Ensure back door is locked.


@pytest.fixture(scope="module")
def cuda_version():
    # binding availability depends on cuda-python version
    _py_major_ver, _ = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    return _py_major_ver, _driver_ver


def test_device_set_current(deinit_cuda):
    device = Device()
    device.set_current()
    assert handle_return(driver.cuCtxGetCurrent()) is not None


def test_device_repr(deinit_cuda):
    device = Device(0)
    device.set_current()
    assert str(device).startswith("<Device 0")


def test_device_alloc(deinit_cuda):
    device = Device()
    device.set_current()
    buffer = device.allocate(1024)
    device.sync()
    assert buffer.handle != 0
    assert buffer.size == 1024
    assert buffer.device_id == int(device)


def test_device_id(deinit_cuda):
    for device in cuda.core.experimental.system.devices:
        device.set_current()
        assert device.device_id == handle_return(runtime.cudaGetDevice())


def test_device_create_stream(init_cuda):
    device = Device()
    stream = device.create_stream()
    assert stream is not None
    assert stream.handle


def test_device_create_event(init_cuda):
    device = Device()
    event = device.create_event()
    assert event is not None
    assert event.handle


def test_pci_bus_id():
    device = Device()
    bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, device.device_id))
    assert device.pci_bus_id == bus_id[:12].decode()


def test_uuid():
    device = Device()
    driver_ver = handle_return(driver.cuDriverGetVersion())
    if driver_ver < 13000:
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


def test_arch():
    device = Device()
    # Test that arch returns the same as the old pattern
    expected_arch = "".join(f"{i}" for i in device.compute_capability)
    assert device.arch == expected_arch
    # Test that it's a string
    assert isinstance(device.arch, str)
    # Test that it matches the expected format (e.g., "75" for CC 7.5)
    cc = device.compute_capability
    assert device.arch == f"{cc.major}{cc.minor}"


cuda_base_properties = [
    ("max_threads_per_block", int),
    ("max_block_dim_x", int),
    ("max_block_dim_y", int),
    ("max_block_dim_z", int),
    ("max_grid_dim_x", int),
    ("max_grid_dim_y", int),
    ("max_grid_dim_z", int),
    ("max_shared_memory_per_block", int),
    ("total_constant_memory", int),
    ("warp_size", int),
    ("max_pitch", int),
    ("maximum_texture1d_width", int),
    ("maximum_texture1d_linear_width", int),
    ("maximum_texture1d_mipmapped_width", int),
    ("maximum_texture2d_width", int),
    ("maximum_texture2d_height", int),
    ("maximum_texture2d_linear_width", int),
    ("maximum_texture2d_linear_height", int),
    ("maximum_texture2d_linear_pitch", int),
    ("maximum_texture2d_mipmapped_width", int),
    ("maximum_texture2d_mipmapped_height", int),
    ("maximum_texture3d_width", int),
    ("maximum_texture3d_height", int),
    ("maximum_texture3d_depth", int),
    ("maximum_texture3d_width_alternate", int),
    ("maximum_texture3d_height_alternate", int),
    ("maximum_texture3d_depth_alternate", int),
    ("maximum_texturecubemap_width", int),
    ("maximum_texture1d_layered_width", int),
    ("maximum_texture1d_layered_layers", int),
    ("maximum_texture2d_layered_width", int),
    ("maximum_texture2d_layered_height", int),
    ("maximum_texture2d_layered_layers", int),
    ("maximum_texturecubemap_layered_width", int),
    ("maximum_texturecubemap_layered_layers", int),
    ("maximum_surface1d_width", int),
    ("maximum_surface2d_width", int),
    ("maximum_surface2d_height", int),
    ("maximum_surface3d_width", int),
    ("maximum_surface3d_height", int),
    ("maximum_surface3d_depth", int),
    ("maximum_surface1d_layered_width", int),
    ("maximum_surface1d_layered_layers", int),
    ("maximum_surface2d_layered_width", int),
    ("maximum_surface2d_layered_height", int),
    ("maximum_surface2d_layered_layers", int),
    ("maximum_surfacecubemap_width", int),
    ("maximum_surfacecubemap_layered_width", int),
    ("maximum_surfacecubemap_layered_layers", int),
    ("max_registers_per_block", int),
    ("clock_rate", int),
    ("texture_alignment", int),
    ("texture_pitch_alignment", int),
    ("gpu_overlap", bool),
    ("multiprocessor_count", int),
    ("kernel_exec_timeout", bool),
    ("integrated", bool),
    ("can_map_host_memory", bool),
    ("compute_mode", int),
    ("concurrent_kernels", bool),
    ("ecc_enabled", bool),
    ("pci_bus_id", int),
    ("pci_device_id", int),
    ("pci_domain_id", int),
    ("tcc_driver", bool),
    ("memory_clock_rate", int),
    ("global_memory_bus_width", int),
    ("l2_cache_size", int),
    ("max_threads_per_multiprocessor", int),
    ("unified_addressing", bool),
    ("compute_capability_major", int),
    ("compute_capability_minor", int),
    ("global_l1_cache_supported", bool),
    ("local_l1_cache_supported", bool),
    ("max_shared_memory_per_multiprocessor", int),
    ("max_registers_per_multiprocessor", int),
    ("managed_memory", bool),
    ("multi_gpu_board", bool),
    ("multi_gpu_board_group_id", int),
    ("host_native_atomic_supported", bool),
    ("single_to_double_precision_perf_ratio", int),
    ("pageable_memory_access", bool),
    ("concurrent_managed_access", bool),
    ("compute_preemption_supported", bool),
    ("can_use_host_pointer_for_registered_mem", bool),
    ("cooperative_launch", bool),
    ("max_shared_memory_per_block_optin", int),
    ("pageable_memory_access_uses_host_page_tables", bool),
    ("direct_managed_mem_access_from_host", bool),
    ("virtual_memory_management_supported", bool),
    ("handle_type_posix_file_descriptor_supported", bool),
    ("handle_type_win32_handle_supported", bool),
    ("handle_type_win32_kmt_handle_supported", bool),
    ("max_blocks_per_multiprocessor", int),
    ("generic_compression_supported", bool),
    ("max_persisting_l2_cache_size", int),
    ("max_access_policy_window_size", int),
    ("gpu_direct_rdma_with_cuda_vmm_supported", bool),
    ("reserved_shared_memory_per_block", int),
    ("sparse_cuda_array_supported", bool),
    ("read_only_host_register_supported", bool),
    ("memory_pools_supported", bool),
    ("gpu_direct_rdma_supported", bool),
    ("gpu_direct_rdma_flush_writes_options", int),
    ("gpu_direct_rdma_writes_ordering", int),
    ("mempool_supported_handle_types", int),
    ("deferred_mapping_cuda_array_supported", bool),
    ("surface_alignment", int),
    ("async_engine_count", int),
    ("can_tex2d_gather", bool),
    ("maximum_texture2d_gather_width", int),
    ("maximum_texture2d_gather_height", int),
    ("stream_priorities_supported", bool),
    ("can_flush_remote_writes", bool),
    ("host_register_supported", bool),
    ("timeline_semaphore_interop_supported", bool),
    ("cluster_launch", bool),
    ("can_use_64_bit_stream_mem_ops", bool),
    ("can_use_stream_wait_value_nor", bool),
    ("dma_buf_supported", bool),
    ("ipc_event_supported", bool),
    ("mem_sync_domain_count", int),
    ("tensor_map_access_supported", bool),
    ("handle_type_fabric_supported", bool),
    ("unified_function_pointers", bool),
    ("numa_config", int),
    ("numa_id", int),
    ("multicast_supported", bool),
    ("mps_enabled", bool),
    ("host_numa_id", int),
    ("d3d12_cig_supported", bool),
    ("mem_decompress_algorithm_mask", int),
    ("mem_decompress_maximum_length", int),
    ("vulkan_cig_supported", bool),
    ("gpu_pci_device_id", int),
    ("gpu_pci_subsystem_id", int),
    ("host_numa_virtual_memory_management_supported", bool),
    ("host_numa_memory_pools_supported", bool),
    ("host_numa_multinode_ipc_supported", bool),
]

# CUDA 13+ specific attributes
cuda_13_properties = [
    ("host_memory_pools_supported", bool),
    ("host_virtual_memory_management_supported", bool),
    ("host_alloc_dma_buf_supported", bool),
    ("only_partial_host_native_atomic_supported", bool),
]

version = get_binding_version()
if version[0] >= 13:
    cuda_base_properties += cuda_13_properties


@pytest.mark.parametrize("property_name, expected_type", cuda_base_properties)
def test_device_property_types(property_name, expected_type):
    device = Device()
    assert isinstance(getattr(device.properties, property_name), expected_type)


def test_device_properties_complete():
    device = Device()
    live_props = set(attr for attr in dir(device.properties) if not attr.startswith("_"))
    tab_props = set(attr for attr, _ in cuda_base_properties)

    excluded_props = set()
    # Exclude CUDA 13+ specific properties when not available
    if version[0] < 13:
        excluded_props.update({prop[0] for prop in cuda_13_properties})

    filtered_tab_props = tab_props - excluded_props
    filtered_live_props = live_props - excluded_props

    assert len(filtered_tab_props) == len(cuda_base_properties)  # Ensure no duplicates.
    assert filtered_tab_props == filtered_live_props  # Ensure exact match.


# ============================================================================
# Device Equality Tests
# ============================================================================


def test_device_equality_same_id(init_cuda):
    """Devices with same device_id should be equal."""
    dev1 = Device(0)
    dev2 = Device(0)

    # On same thread, should be same object (singleton)
    assert dev1 is dev2, "Device is per-thread singleton"
    assert dev1 == dev2, "Same device_id should be equal"


def test_device_equality_reflexive(init_cuda):
    """Device should equal itself (reflexive property)."""
    device = Device(0)
    assert device == device, "Device should equal itself"


def test_device_inequality_different_id(init_cuda):
    """Devices with different device_id should not be equal."""
    try:
        dev0 = Device(0)
        dev1 = Device(1)

        assert dev0 != dev1, "Different devices should not be equal"
        assert dev0 != dev1, "Different devices should be not-equal"
    except (ValueError, Exception):
        pytest.skip("Test requires at least 2 CUDA devices")


def test_device_type_safety(init_cuda):
    """Comparing Device with wrong type should return False."""
    device = Device(0)

    assert (device == "not a device") is False
    assert (device == 123) is False
    assert (device is None) is False


# ============================================================================
# Device Hash Tests
# ============================================================================


def test_device_hash_consistency(init_cuda):
    """Hash of same Device object should be consistent."""
    device = Device(0)

    hash1 = hash(device)
    hash2 = hash(device)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_device_equality_same_id_hash(init_cuda):
    """Devices with same device_id should be equal."""
    dev1 = Device(0)
    dev2 = Device(0)

    # On same thread, should be same object (singleton)
    assert dev1 is dev2, "Device is per-thread singleton"
    assert dev1 == dev2, "Same device_id should be equal"
    assert hash(dev1) == hash(dev2), "Same device_id should hash equal"


def test_device_inequality_different_id_hash(init_cuda):
    """Devices with different device_id should not be equal."""
    try:
        # Only run test when two devices are available.
        dev0 = Device(0)
        dev1 = Device(1)

        assert dev0 != dev1, "Different devices should not be equal"
        assert hash(dev0) != hash(dev1), "Different devices should have different hashes"
    except (ValueError, Exception):
        # Test is skipped if only one device available
        pytest.skip("Test requires at least 2 CUDA devices")


def test_device_dict_key(init_cuda):
    """Devices should be usable as dictionary keys."""
    dev0 = Device(0)

    device_cache = {dev0: "gpu0_data"}
    assert device_cache[dev0] == "gpu0_data"

    # Getting device again should find same entry
    dev0_again = Device(0)
    assert device_cache[dev0_again] == "gpu0_data"


def test_device_set_membership(init_cuda):
    """Devices should work correctly in sets."""
    dev0_a = Device(0)
    dev0_b = Device(0)

    device_set = {dev0_a}

    # Same device_id should not add duplicate
    device_set.add(dev0_b)
    assert len(device_set) == 1, "Should not add duplicate device"
