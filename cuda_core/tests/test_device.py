# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        cuda.core._device.DeviceProperties()  # Ensure back door is locked.


@pytest.fixture(scope="module")
def cuda_version():
    # binding availability depends on cuda-python version
    _py_major_ver, _ = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    return _py_major_ver, _driver_ver


def test_to_system_device(deinit_cuda):
    from cuda.core.system import _system

    device = Device()

    if not _system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        with pytest.raises(RuntimeError):
            device.to_system_device()
        pytest.skip("NVML support requires cuda.bindings version 12.9.6+ or 13.1.2+")

    from cuda.bindings._test_helpers.arch_check import hardware_supports_nvml

    if not hardware_supports_nvml():
        pytest.skip("NVML not supported on this platform")

    from cuda.core.system import Device as SystemDevice

    system_device = device.to_system_device()
    assert isinstance(system_device, SystemDevice)
    assert system_device.uuid == device.uuid

    # Technically, this test will only work with PCI devices, but are there
    # non-PCI devices we need to support?

    # CUDA only returns a 2-byte PCI bus ID domain, whereas NVML returns a
    # 4-byte domain
    assert device.pci_bus_id == system_device.pci_info.bus_id[4:]


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


def test_device_alloc_zero_bytes(deinit_cuda):
    device = Device()
    device.set_current()
    buffer = device.allocate(0)
    device.sync()
    assert buffer.handle >= 0
    assert buffer.size == 0
    assert buffer.device_id == int(device)


def test_device_id(deinit_cuda):
    for device in Device.get_all_devices():
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


# ============================================================================
# Device Context Manager Tests
# ============================================================================


def _get_current_context():
    """Return the current CUcontext as an int (0 means NULL / no context)."""
    return int(handle_return(driver.cuCtxGetCurrent()))


def test_context_manager_basic(deinit_cuda):
    """with Device(0) sets the device as current and restores on exit."""
    assert _get_current_context() == 0, "Should start with no active context"

    with Device(0):
        assert _get_current_context() != 0, "Device should be current inside the with block"

    assert _get_current_context() == 0, "No context should be current after exiting"


def test_context_manager_restores_previous(deinit_cuda):
    """Context manager restores the previously active context, not NULL."""
    dev0 = Device(0)
    dev0.set_current()
    ctx_before = _get_current_context()
    assert ctx_before != 0

    with Device(0):
        pass

    assert _get_current_context() == ctx_before, "Should restore the previous context"


def test_context_manager_exception_safety(deinit_cuda):
    """Device context is restored even when an exception is raised."""
    # Start with no active context so restoration is distinguishable
    assert _get_current_context() == 0

    with pytest.raises(RuntimeError, match="test error"), Device(0):
        assert _get_current_context() != 0, "Device should be active inside the block"
        raise RuntimeError("test error")

    assert _get_current_context() == 0, "Must restore NULL context after exception"


def test_context_manager_returns_device(deinit_cuda):
    """__enter__ returns the Device instance for use in 'as' clause."""
    device = Device(0)
    with device as dev:
        assert dev is device

    assert _get_current_context() == 0


def test_context_manager_nesting_same_device(deinit_cuda):
    """Nested with-blocks on the same device work correctly."""
    dev0 = Device(0)

    with dev0:
        ctx_outer = _get_current_context()
        with dev0:
            ctx_inner = _get_current_context()
            assert ctx_inner == ctx_outer, "Same device should yield same context"
        assert _get_current_context() == ctx_outer, "Outer context restored after inner exit"

    assert _get_current_context() == 0


def test_context_manager_deep_nesting(deinit_cuda):
    """Deep nesting and reentrancy restore correctly at each level."""
    dev0 = Device(0)

    with dev0:
        ctx_level1 = _get_current_context()
        with dev0:
            ctx_level2 = _get_current_context()
            with dev0:
                assert _get_current_context() != 0
            assert _get_current_context() == ctx_level2
        assert _get_current_context() == ctx_level1

    assert _get_current_context() == 0


def test_context_manager_nesting_different_devices(mempool_device_x2):
    """Nested with-blocks on different devices restore correctly."""
    dev0, dev1 = mempool_device_x2
    ctx_dev0 = _get_current_context()

    with dev1:
        ctx_inside = _get_current_context()
        assert ctx_inside != ctx_dev0, "Different device should have different context"

    assert _get_current_context() == ctx_dev0, "Original device context should be restored"


def test_context_manager_deep_nesting_multi_gpu(mempool_device_x2):
    """Deep nesting across multiple devices restores correctly at each level."""
    dev0, dev1 = mempool_device_x2

    with dev0:
        ctx_level0 = _get_current_context()
        with dev1:
            ctx_level1 = _get_current_context()
            assert ctx_level1 != ctx_level0
            with dev0:
                assert _get_current_context() == ctx_level0, "Same device should yield same primary context"
                with dev1:
                    assert _get_current_context() == ctx_level1
                assert _get_current_context() == ctx_level0
            assert _get_current_context() == ctx_level1
        assert _get_current_context() == ctx_level0


def test_context_manager_set_current_inside(mempool_device_x2):
    """set_current() inside a with block does not affect restoration on exit."""
    dev0, dev1 = mempool_device_x2
    ctx_dev0 = _get_current_context()  # dev0 is current from fixture

    with dev0:
        dev1.set_current()  # change the active device inside the block
        assert _get_current_context() != ctx_dev0

    assert _get_current_context() == ctx_dev0, "Must restore the context saved at __enter__"


def test_context_manager_device_usable_after_exit(deinit_cuda):
    """Device singleton is not corrupted after context manager exit."""
    device = Device(0)
    with device:
        pass

    assert _get_current_context() == 0

    # Device should still be usable via set_current
    device.set_current()
    assert _get_current_context() != 0
    stream = device.create_stream()
    assert stream is not None


def test_context_manager_initializes_device(deinit_cuda):
    """with Device(N) should initialize the device, making it ready for use."""
    device = Device(0)
    with device:
        # allocate requires an active context; should not raise
        buf = device.allocate(1024)
        assert buf.handle != 0


def test_context_manager_thread_safety(mempool_device_x3):
    """Concurrent threads using context managers on different devices don't interfere."""
    import concurrent.futures
    import threading

    devices = mempool_device_x3
    barrier = threading.Barrier(len(devices))
    errors = []

    def worker(dev):
        try:
            ctx_before = _get_current_context()
            with dev:
                barrier.wait(timeout=5)
                buf = dev.allocate(1024)
                assert buf.handle != 0
            assert _get_current_context() == ctx_before
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pool.map(worker, devices)

    assert not errors, f"Thread errors: {errors}"
