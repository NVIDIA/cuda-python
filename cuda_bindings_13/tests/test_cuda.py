# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import contextlib
import ctypes
import os
import shutil
import tempfile

import numpy as np
import pytest
from cuda_python_test_helpers.memory import (
    POOL_SIZE,
    supports_managed_memory,
    supports_memory_pools,
    xfail_graph_mempool_oom,
)

from cuda.bindings._internal.utils import FunctionNotFoundError
from cuda.bindings._v2 import driver
from cuda_python_test_helpers import driver_version_less_than

# NULL dependency array: the node has no dependencies.
NO_DEPS = 0


def supportsCudaAPI(name):
    return name in dir(driver)


def callableBinary(name):
    return shutil.which(name) is not None


def node_array(nodes):
    # Build a C array of graph node handles; the caller passes
    # ctypes.addressof(array) and must keep the array alive during the call.
    return (ctypes.c_void_p * len(nodes))(*nodes)


@pytest.mark.skipif(True, reason="Always skip!")
def test_always_skip():
    pass


def test_cuda_memcpy():
    # Get device

    # Allocate dev memory
    size = int(1024 * np.uint8().itemsize)
    dptr = driver.mem_alloc_v2(size)

    # Set h1 and h2 memory to be different
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert np.array_equal(h1, h2) is False

    # h1 to D
    driver.memcpy_htod_v2(dptr, h1, size)

    # D to h2
    driver.memcpy_dtoh_v2(h2, dptr, size)

    # Validate h1 == h2
    assert np.array_equal(h1, h2)

    # Host arguments accept any contiguous buffer: ctypes scalars and arrays, not just NumPy.
    src = (ctypes.c_int * 4)(1, 2, 3, 4)
    driver.memcpy_htod_v2(dptr, src, ctypes.sizeof(src))
    dst = (ctypes.c_int * 4)()
    driver.memcpy_dtoh_v2(dst, dptr, ctypes.sizeof(dst))
    assert list(dst) == [1, 2, 3, 4]

    scalar = ctypes.c_int(42)
    driver.memcpy_htod_v2(dptr, scalar, ctypes.sizeof(scalar))
    out = ctypes.c_int()
    driver.memcpy_dtoh_v2(out, dptr, ctypes.sizeof(out))
    assert out.value == 42

    matrix = np.arange(4, dtype=np.int32).reshape(2, 2)
    driver.memcpy_htod_v2(dptr, matrix, matrix.nbytes)
    back = np.zeros((2, 2), dtype=np.int32)
    driver.memcpy_dtoh_v2(back, dptr, back.nbytes)
    assert np.array_equal(matrix, back)

    # Cleanup
    driver.mem_free_v2(dptr)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.skipif(driver_version_less_than(13041), reason="cuGraphAddNode_v3 was added in CUDA 13.4.1")
def test_graph_add_node_v3_returns_the_node():
    graph = driver.graph_create(0)
    params = driver.GraphNodeParams()
    params.type = driver.GraphNodeType.EMPTY
    node = driver.graph_add_node_v3(graph, NO_DEPS, None, 0, params)
    assert isinstance(node, int) and node != 0
    driver.graph_destroy(graph)


def test_cuda_array():
    # No context created
    desc = driver.ArrayDescriptor_v2()
    with pytest.raises((driver.InvalidContextError, driver.InvalidValueError)):
        driver.array_create_v2(desc)

    # Desciption not filled
    with pytest.raises(driver.InvalidValueError):
        driver.array_create_v2(desc)

    # Pass
    desc.format = driver.ArrayFormat.SIGNED_INT8
    desc.num_channels = 1
    desc.width = 1
    arr = driver.array_create_v2(desc)

    driver.array_destroy(arr)


def test_cuda_uuid_list_access(device):
    uuid = driver.device_get_uuid_v2(device)
    assert isinstance(uuid.bytes, bytes)
    assert len(uuid.bytes) == 16

    # The UUID is 16 raw bytes, not text: values that are not valid UTF-8 round-trip.
    raw = bytes(range(0x80, 0x90))
    copy = driver.Uuid()
    copy.bytes = raw
    assert copy.bytes == raw
    with pytest.raises(ValueError):
        copy.bytes = b"too short"

    jit_option = driver.JitOption
    options = {
        jit_option.INFO_LOG_BUFFER: 1,
        jit_option.INFO_LOG_BUFFER_SIZE_BYTES: 2,
        jit_option.ERROR_LOG_BUFFER: 3,
        jit_option.ERROR_LOG_BUFFER_SIZE_BYTES: 4,
        jit_option.LOG_VERBOSE: 5,
    }


_JIT_PTX = b""".version 7.5
.target sm_75
.address_size 64
.visible .entry k()
{
    ret;
}
\0"""


def _load_ptx_with_log_options(ptx):
    # Returns (module-or-exception, info_log, error_log, error_log_size) for a load using the JIT log options.
    option_keys = [
        driver.JitOption.INFO_LOG_BUFFER,
        driver.JitOption.INFO_LOG_BUFFER_SIZE_BYTES,
        driver.JitOption.ERROR_LOG_BUFFER,
        driver.JitOption.ERROR_LOG_BUFFER_SIZE_BYTES,
        driver.JitOption.LOG_VERBOSE,
    ]
    keys = (ctypes.c_int * len(option_keys))(*[int(k) for k in option_keys])
    info_log = ctypes.create_string_buffer(1024)
    error_log = ctypes.create_string_buffer(1024)
    values = (ctypes.c_void_p * len(option_keys))(
        ctypes.addressof(info_log), len(info_log), ctypes.addressof(error_log), len(error_log), 1
    )
    try:
        result = driver.module_load_data_ex(ptx, len(option_keys), ctypes.addressof(keys), ctypes.addressof(values))
    except driver.DriverError as e:
        result = e
    return result, info_log.value, error_log.value, values[3]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_cuda_cuModuleLoadDataEx():
    mod, _, error_log, _ = _load_ptx_with_log_options(_JIT_PTX)
    assert not isinstance(mod, Exception), mod
    try:
        assert error_log == b""
        assert driver.module_get_function(mod, b"k")
    finally:
        driver.module_unload(mod)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_cuda_cuModuleLoadDataEx_error_log():
    # The JIT writes its diagnostics into the caller's buffer, and updates the size option value to the
    # number of bytes written.
    bad_ptx = b".version 7.5\n.target sm_75\nbogus\n\0"
    result, _, error_log, error_log_size = _load_ptx_with_log_options(bad_ptx)
    assert isinstance(result, driver.InvalidPtxError)
    assert b"syntax error" in error_log
    assert 0 < error_log_size <= 1024


def test_cuda_repr():
    # The v2 structs do not have a custom __repr__; check the default-initialized values instead.
    actual = driver.ExternalSemaphoreSignalParams_v1()
    assert isinstance(actual, driver.ExternalSemaphoreSignalParams_v1)
    assert actual.flags_ == 0
    assert actual.params.fence.value == 0
    assert actual.params.nv_sci_sync.fence == 0
    assert actual.params.keyed_mutex.key == 0

    kernel_params = driver.KernelNodeParams_v2()
    assert kernel_params.func == 0
    assert kernel_params.grid_dim_x == 0
    assert kernel_params.grid_dim_y == 0
    assert kernel_params.grid_dim_z == 0
    assert kernel_params.block_dim_x == 0
    assert kernel_params.block_dim_y == 0
    assert kernel_params.block_dim_z == 0
    assert kernel_params.shared_mem_bytes == 0
    assert kernel_params.kernel_params == 0
    assert kernel_params.extra == 0


def test_cuda_struct_list_of_enums():
    desc = driver.TextureDesc_v1()
    desc.address_mode = [
        driver.AddressMode.WRAP,
        driver.AddressMode.CLAMP,
        driver.AddressMode.MIRROR,
    ]
    assert list(desc.address_mode) == [
        driver.AddressMode.WRAP,
        driver.AddressMode.CLAMP,
        driver.AddressMode.MIRROR,
    ]

    # # Too many args
    # desc.address_mode = [driver.AddressMode.WRAP,
    #                      driver.AddressMode.CLAMP,
    #                      driver.AddressMode.MIRROR,
    #                      driver.AddressMode.BORDER]

    # # Too little args
    # desc.address_mode = [driver.AddressMode.WRAP,
    #                      driver.AddressMode.CLAMP]


def test_cuda_CUstreamBatchMemOpParams():
    params = driver.StreamBatchMemOpParams_v1()
    params.operation = driver.StreamBatchMemOpType.WAIT_VALUE_32
    params.wait_value.operation = driver.StreamBatchMemOpType.WAIT_VALUE_32
    params.write_value.operation = driver.StreamBatchMemOpType.WAIT_VALUE_32
    params.flush_remote_writes.operation = driver.StreamBatchMemOpType.WAIT_VALUE_32
    params.wait_value.value64 = 666
    assert np.asarray(params.wait_value.value64).item() == 666

    # A struct-valued member of an array-backed type is a wrapper object (its scalars are plain ints),
    # never a raw NumPy record whose scalars come back as 1-element arrays.
    batch = driver.StreamBatchMemOpParams_v1()
    assert isinstance(batch.wait_value, driver.StreamMemOpWaitValueParams)
    assert isinstance(batch.wait_value.flags_, int)
    batch.wait_value.value = 7
    assert batch.wait_value.value == 7
    assert batch[0].wait_value.value == 7

    access = driver.MemAccessDesc_v1()
    assert isinstance(access.location, driver.MemLocation_v1)
    assert isinstance(access.location.id, int)

    # Elements of a larger array behave the same way.
    assert isinstance(driver.StreamBatchMemOpParams_v1(3)[2].wait_value.flags_, int)
    assert isinstance(driver.ArrayMapInfo_v1(2)[1].resource.array, int)


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supports_memory_pools(), reason="When new attributes were introduced"
)
def test_cuda_memPool_attr():
    poolProps = driver.MemPoolProps_v1()
    poolProps.alloc_type = driver.MemAllocationType.PINNED
    poolProps.location.id = 0
    poolProps.location.type = driver.MemLocationType.DEVICE
    poolProps.max_size = POOL_SIZE

    attr_list = [None] * 8
    pool = driver.mem_pool_create(poolProps)

    def get_attr(attr):
        return int(driver.mem_pool_get_attribute(pool, attr))

    for idx, attr in enumerate(
        [
            driver.MemPoolAttribute.REUSE_FOLLOW_EVENT_DEPENDENCIES,
            driver.MemPoolAttribute.REUSE_ALLOW_OPPORTUNISTIC,
            driver.MemPoolAttribute.REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            driver.MemPoolAttribute.RELEASE_THRESHOLD,
            driver.MemPoolAttribute.RESERVED_MEM_CURRENT,
            driver.MemPoolAttribute.RESERVED_MEM_HIGH,
            driver.MemPoolAttribute.USED_MEM_CURRENT,
            driver.MemPoolAttribute.USED_MEM_HIGH,
        ]
    ):
        attr_list[idx] = get_attr(attr)

    for idxA, attr in enumerate(
        [
            driver.MemPoolAttribute.REUSE_FOLLOW_EVENT_DEPENDENCIES,
            driver.MemPoolAttribute.REUSE_ALLOW_OPPORTUNISTIC,
            driver.MemPoolAttribute.REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ]
    ):
        value = ctypes.c_int(0)
        driver.mem_pool_set_attribute(pool, attr, ctypes.addressof(value))
    for idx, attr in enumerate([driver.MemPoolAttribute.RELEASE_THRESHOLD]):
        value = ctypes.c_uint64(9)
        driver.mem_pool_set_attribute(pool, attr, ctypes.addressof(value))

    for idx, attr in enumerate(
        [
            driver.MemPoolAttribute.REUSE_FOLLOW_EVENT_DEPENDENCIES,
            driver.MemPoolAttribute.REUSE_ALLOW_OPPORTUNISTIC,
            driver.MemPoolAttribute.REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            driver.MemPoolAttribute.RELEASE_THRESHOLD,
        ]
    ):
        attr_list[idx] = get_attr(attr)
    assert attr_list[0] == 0
    assert attr_list[1] == 0
    assert attr_list[2] == 0
    assert int(attr_list[3]) == 9

    driver.mem_pool_destroy(pool)


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supports_managed_memory(), reason="When new attributes were introduced"
)
def test_cuda_pointer_attr():
    ptr = driver.mem_alloc_managed(0x1000, driver.MemAttachFlags.GLOBAL.value)

    # Individual version
    attr_type_list = [
        driver.PointerAttribute.CONTEXT,
        driver.PointerAttribute.MEMORY_TYPE,
        driver.PointerAttribute.DEVICE_POINTER,
        driver.PointerAttribute.HOST_POINTER,
        # driver.PointerAttribute.P2P_TOKENS, # TODO: Can I somehow test this?
        driver.PointerAttribute.SYNC_MEMOPS,
        driver.PointerAttribute.BUFFER_ID,
        driver.PointerAttribute.IS_MANAGED,
        driver.PointerAttribute.DEVICE_ORDINAL,
        driver.PointerAttribute.IS_LEGACY_CUDA_IPC_CAPABLE,
        driver.PointerAttribute.RANGE_START_ADDR,
        driver.PointerAttribute.RANGE_SIZE,
        driver.PointerAttribute.MAPPED,
        driver.PointerAttribute.ALLOWED_HANDLE_TYPES,
        driver.PointerAttribute.IS_GPU_DIRECT_RDMA_CAPABLE,
        driver.PointerAttribute.ACCESS_FLAGS,
        driver.PointerAttribute.MEMPOOL_HANDLE,
    ]
    attr_value_list = [driver.pointer_get_attribute(attr, ptr) for attr in attr_type_list]

    # List version
    attr_value_list_v2 = driver.pointer_get_attributes(attr_type_list, ptr)
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert attr1 == attr2

    # Test setting values
    for val in (True, False):
        value = ctypes.c_int(val)
        driver.pointer_set_attribute(ctypes.addressof(value), driver.PointerAttribute.SYNC_MEMOPS, ptr)
        assert bool(driver.pointer_get_attribute(driver.PointerAttribute.SYNC_MEMOPS, ptr)) == val

    driver.mem_free_v2(ptr)


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supports_managed_memory(), reason="When new attributes were introduced"
)
def test_pointer_get_attributes_device_ordinal():
    (value,) = driver.pointer_get_attributes([driver.PointerAttribute.DEVICE_ORDINAL], 0)

    # Drivers use either negative sentinel for a null pointer. Both ensure the
    # signed device ordinal is returned rather than an unsigned value.
    assert value in (driver.SpecialDevice.CPU, driver.SpecialDevice.INVALID)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.skipif(
    driver_version_less_than(11030) or not supports_managed_memory(), reason="When new attributes were introduced"
)
def test_pointer_allowed_handle_types_preserves_64_bits():
    ptr = driver.mem_alloc_managed(0x1000, driver.MemAttachFlags.GLOBAL.value)
    try:
        assert driver.pointer_get_attribute(driver.PointerAttribute.ALLOWED_HANDLE_TYPES, ptr) == 0
    finally:
        driver.mem_free_v2(ptr)


@pytest.mark.skipif(not supports_managed_memory(), reason="When new attributes were introduced")
def test_cuda_mem_range_attr(device):
    size = 0x1000
    location_device = driver.MemLocation_v1()
    location_device.type = driver.MemLocationType.DEVICE
    location_device.id = int(device)
    location_cpu = driver.MemLocation_v1()
    location_cpu.type = driver.MemLocationType.HOST
    location_cpu.id = int(driver.SpecialDevice.CPU)

    ptr = driver.mem_alloc_managed(size, driver.MemAttachFlags.GLOBAL.value)
    driver.mem_advise_v2(ptr, size, driver.MemAdvise.SET_READ_MOSTLY, location_device)
    driver.mem_advise_v2(ptr, size, driver.MemAdvise.SET_PREFERRED_LOCATION, location_cpu)
    driver.mem_advise_v2(ptr, size, driver.MemAdvise.SET_ACCESSED_BY, location_cpu)
    concurrentSupported = driver.device_get_attribute(driver.DeviceAttribute.CONCURRENT_MANAGED_ACCESS, device)
    if concurrentSupported:
        driver.mem_advise_v2(ptr, size, driver.MemAdvise.SET_ACCESSED_BY, location_device)
        expected_values_list = ([1, -1, [0, -1, -2], -2],)
    else:
        expected_values_list = ([1, -1, [-1, -2, -2], -2], [0, -2, [-2, -2, -2], -2])

    # Individual version
    attr_type_list = [
        driver.MemRangeAttribute.READ_MOSTLY,
        driver.MemRangeAttribute.PREFERRED_LOCATION,
        driver.MemRangeAttribute.ACCESSED_BY,
        driver.MemRangeAttribute.LAST_PREFETCH_LOCATION,
    ]
    attr_type_size_list = [4, 4, 12, 4]
    attr_value_list = [None] * len(attr_type_list)
    for idx in range(len(attr_type_list)):
        attr_value_list[idx] = driver.mem_range_get_attribute(attr_type_size_list[idx], attr_type_list[idx], ptr, size)

    matched = False
    for expected_values in expected_values_list:
        if expected_values == attr_value_list:
            matched = True
            break
    if not matched:
        raise RuntimeError(f"attr_value_list {attr_value_list} did not match any {expected_values_list}")

    # List version
    attr_value_list_v2 = driver.mem_range_get_attributes(attr_type_size_list, attr_type_list, ptr, size)
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert attr1 == attr2

    driver.mem_free_v2(ptr)


@pytest.mark.skipif(
    driver_version_less_than(11040) or not supports_memory_pools(), reason="Mempool for graphs not supported"
)
@pytest.mark.thread_unsafe(reason="used high memory can be higher if threaded.")
def test_cuda_graphMem_attr(device):
    stream = driver.stream_create(0)

    graph = driver.graph_create(0)

    allocSize = 1

    params = driver.MemAllocNodeParams_v1()
    params.pool_props.location.type = driver.MemLocationType.DEVICE
    params.pool_props.location.id = device
    params.pool_props.alloc_type = driver.MemAllocationType.PINNED
    params.bytesize = allocSize

    try:
        allocNode = driver.graph_add_mem_alloc_node(graph, NO_DEPS, 0, params)
    except driver.OutOfMemoryError:
        driver.graph_destroy(graph)
        driver.stream_destroy_v2(stream)
        xfail_graph_mempool_oom()
    deps = node_array([allocNode])
    freeNode = driver.graph_add_mem_free_node(graph, ctypes.addressof(deps), 1, params.dptr)

    graphExec = driver.graph_instantiate(graph, 0)

    driver.graph_launch(graphExec, stream)

    def get_graph_mem_attr(attr):
        return int(driver.device_get_graph_mem_attribute(device, attr))

    used = get_graph_mem_attr(driver.GraphMemAttribute.USED_MEM_CURRENT)
    usedHigh = get_graph_mem_attr(driver.GraphMemAttribute.USED_MEM_HIGH)
    reserved = get_graph_mem_attr(driver.GraphMemAttribute.RESERVED_MEM_CURRENT)
    reservedHigh = get_graph_mem_attr(driver.GraphMemAttribute.RESERVED_MEM_HIGH)

    assert int(used) >= allocSize
    assert int(usedHigh) == int(used)
    assert int(reserved) == int(usedHigh)
    assert int(reservedHigh) == int(reserved)

    driver.graph_destroy(graph)
    driver.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(12010)
    or not supportsCudaAPI("coredump_set_attribute_global")
    or not supportsCudaAPI("coredump_get_attribute_global"),
    reason="Coredump API not present",
)
@pytest.mark.thread_unsafe(reason="coredump settings are process-global")
def test_cuda_coredump_attr():
    attr_list = [None] * 6

    driver.coredump_set_attribute_global(driver.CoredumpSettings.TRIGGER_HOST, False)
    driver.coredump_set_attribute_global(driver.CoredumpSettings.FILE, b"corefile")
    driver.coredump_set_attribute_global(driver.CoredumpSettings.PIPE, b"corepipe")
    driver.coredump_set_attribute_global(driver.CoredumpSettings.LIGHTWEIGHT, True)

    for idx, attr in enumerate(
        [
            driver.CoredumpSettings.TRIGGER_HOST,
            driver.CoredumpSettings.FILE,
            driver.CoredumpSettings.PIPE,
            driver.CoredumpSettings.LIGHTWEIGHT,
        ]
    ):
        attr_tmp = driver.coredump_get_attribute_global(attr)
        attr_list[idx] = attr_tmp

    assert attr_list[0] is False
    assert attr_list[1] == b"corefile"
    assert attr_list[2] == b"corepipe"
    assert attr_list[3] is True


@pytest.mark.thread_unsafe(reason="coredump settings are process-global")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_coredump_bool_uses_single_byte_storage():
    # Round-trip a bool-valued setting in both directions; a getter that read
    # more than one byte of the storage would be affected by stale high bytes.
    # LIGHTWEIGHT is used because the Windows driver does not round-trip TRIGGER_HOST.
    driver.coredump_set_attribute_global(driver.CoredumpSettings.LIGHTWEIGHT, True)
    assert driver.coredump_get_attribute_global(driver.CoredumpSettings.LIGHTWEIGHT) is True
    driver.coredump_set_attribute_global(driver.CoredumpSettings.LIGHTWEIGHT, False)
    assert driver.coredump_get_attribute_global(driver.CoredumpSettings.LIGHTWEIGHT) is False


@pytest.mark.agent_authored(model="claude-sonnet-4-6")
def test_ffi_coverage_new_pointer_attributes():
    """Smoke-test that every newly added PointerAttribute can be queried
    without raising TypeError (driver errors for unsupported attributes are tolerated)."""
    new_attrs = [
        driver.PointerAttribute.MAPPING_SIZE,
        driver.PointerAttribute.MEMORY_BLOCK_ID,
        driver.PointerAttribute.IS_HW_DECOMPRESS_CAPABLE,
        driver.PointerAttribute.LOCALITY_DOMAIN_ORDINAL,
    ]
    ptr = driver.mem_alloc_v2(0x1000)
    try:
        for attr in new_attrs:
            # Not supported by this driver version: tolerate driver errors
            with contextlib.suppress(driver.DriverError):
                driver.pointer_get_attribute(attr, ptr)
    finally:
        driver.mem_free_v2(ptr)


@pytest.mark.agent_authored(model="claude-sonnet-4-6")
def test_ffi_coverage_mapping_base_addr():
    """PointerAttribute.MAPPING_BASE_ADDR uses the CUdeviceptr getter path."""
    ptr = driver.mem_alloc_v2(0x1000)
    try:
        assert driver.pointer_get_attribute(driver.PointerAttribute.MAPPING_BASE_ADDR, ptr) != 0
    finally:
        driver.mem_free_v2(ptr)


@pytest.mark.agent_authored(model="claude-sonnet-4-6")
@pytest.mark.skipif(
    driver_version_less_than(11030) or not supports_memory_pools(), reason="When new attributes were introduced"
)
def test_ffi_coverage_new_mempool_attributes():
    """Smoke-test all newly added MemPoolAttribute queries (driver errors for
    attributes unsupported by this driver are tolerated)."""
    pool_props = driver.MemPoolProps_v1()
    pool_props.alloc_type = driver.MemAllocationType.PINNED
    pool_props.location.id = 0
    pool_props.location.type = driver.MemLocationType.DEVICE
    pool_props.max_size = POOL_SIZE
    pool = driver.mem_pool_create(pool_props)
    try:
        for attr in [
            driver.MemPoolAttribute.ALLOCATION_TYPE,
            driver.MemPoolAttribute.LOCATION_ID,
            driver.MemPoolAttribute.LOCATION_TYPE,
            driver.MemPoolAttribute.HW_DECOMPRESS_ENABLED,
            driver.MemPoolAttribute.LOCALITY_DOMAIN_ID,
            driver.MemPoolAttribute.EXPORT_HANDLE_TYPES,
        ]:
            # Not supported by this driver version: tolerate driver errors
            with contextlib.suppress(driver.DriverError):
                driver.mem_pool_get_attribute(pool, attr)
    finally:
        driver.mem_pool_destroy(pool)


@pytest.mark.agent_authored(model="claude-sonnet-4-6")
@pytest.mark.skipif(not supports_managed_memory(), reason="When new attributes were introduced")
def test_ffi_coverage_new_mem_range_attributes():
    """Smoke-test all newly added MemRangeAttribute queries (scalar int)."""
    size = 0x1000
    ptr = driver.mem_alloc_managed(size, driver.MemAttachFlags.GLOBAL.value)
    try:
        new_attrs = [
            driver.MemRangeAttribute.PREFERRED_LOCATION_TYPE,
            driver.MemRangeAttribute.PREFERRED_LOCATION_ID,
            driver.MemRangeAttribute.LAST_PREFETCH_LOCATION_TYPE,
            driver.MemRangeAttribute.LAST_PREFETCH_LOCATION_ID,
        ]
        for attr in new_attrs:
            # Not supported by this driver version: tolerate driver errors
            with contextlib.suppress(driver.DriverError):
                driver.mem_range_get_attribute(4, attr, ptr, size)
    finally:
        driver.mem_free_v2(ptr)


@pytest.mark.thread_unsafe(reason="coredump settings are process-global")
@pytest.mark.agent_authored(model="claude-sonnet-4-6")
def test_ffi_coverage_coredump_generation_flags():
    """CoredumpSettings.GENERATION_FLAGS is an unsigned int bitmask."""
    settings = driver.CoredumpSettings.GENERATION_FLAGS
    flags_enum = driver.CoredumpGenerationFlags

    original = driver.coredump_get_attribute_global(settings)
    assert isinstance(original, int)
    try:
        wanted = int(flags_enum.SKIP_GLOBAL_MEMORY) | int(flags_enum.SKIP_SHARED_MEMORY)
        driver.coredump_set_attribute_global(settings, wanted)
        assert driver.coredump_get_attribute_global(settings) == wanted
    finally:
        driver.coredump_set_attribute_global(settings, original)


def test_get_error_name_and_string():
    device = driver.device_get(0)
    s = driver.get_error_string(driver.Result.SUCCESS)
    assert s == "no error"
    s = driver.get_error_name(driver.Result.SUCCESS)
    assert s == "CUDA_SUCCESS"

    with pytest.raises(driver.InvalidDeviceError) as excinfo:
        driver.device_get(-1)
    err = excinfo.value.status
    s = driver.get_error_string(err)
    assert s == "invalid device ordinal"
    s = driver.get_error_name(err)
    assert s == "CUDA_ERROR_INVALID_DEVICE"

    with pytest.raises(driver.InvalidValueError):
        driver.get_error_string(123456)
    with pytest.raises(driver.InvalidValueError):
        driver.get_error_name(123456)


# TODO: cuStreamGetCaptureInfo_v2
@pytest.mark.skipif(driver_version_less_than(11030), reason="Driver too old for cuStreamGetCaptureInfo_v2")
def test_stream_capture():
    pass


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cuda_launch_host_func_graph_replay():
    # A host function captured into a graph runs once per graph launch. The
    # bindings must hand the function pointer and the user data to CUDA
    # unchanged so that every launch reaches the same function (issue #790).
    counter = ctypes.c_int(0)

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def host_fn(userData):
        ctypes.c_int.from_address(userData).value += 1

    callback = ctypes.cast(host_fn, ctypes.c_void_p).value
    stream = driver.stream_create(0)
    try:
        # Thread-local mode: pytest-run-parallel runs this test in several threads
        # at once, and in global mode a call such as graph_instantiate in one
        # thread fails while another thread is capturing.
        driver.stream_begin_capture_v2(stream, driver.StreamCaptureMode.THREAD_LOCAL)
        driver.launch_host_func(stream, callback, ctypes.addressof(counter))
        graph = driver.stream_end_capture(stream)
        try:
            graph_exec = driver.graph_instantiate(graph, 0)
            try:
                for launch in range(1, 6):
                    driver.graph_launch(graph_exec, stream)
                    driver.stream_synchronize(stream)
                    assert counter.value == launch
            finally:
                driver.graph_exec_destroy(graph_exec)
        finally:
            driver.graph_destroy(graph)
    finally:
        driver.stream_destroy_v2(stream)


def test_profiler():
    driver.profiler_start()
    driver.profiler_stop()


def test_anon_assign():
    val1 = driver.ExecAffinityParam_v1()
    val2 = driver.ExecAffinityParam_v1()

    assert val1.param.sm_count.val == 0
    val1.param.sm_count.val = 5
    assert val1.param.sm_count.val == 5
    val2.param.sm_count.val = 11
    assert val2.param.sm_count.val == 11

    val1.param = val2.param
    assert val1.param.sm_count.val == 11


def test_union_assign():
    val = driver.LaunchAttributeValue()
    val.cluster_dim.x, val.cluster_dim.y, val.cluster_dim.z = 9, 9, 9
    attr = driver.LaunchAttribute()
    attr.value = val

    assert val.cluster_dim.x == 9
    assert val.cluster_dim.y == 9
    assert val.cluster_dim.z == 9
    assert not hasattr(val, "pad")

    # A union blob is raw bytes, not text: values that are not valid UTF-8 round-trip.
    params = driver.GraphNodeParams()
    assert isinstance(params.as_bytes, bytes)
    raw = bytes((0x80 + i) % 256 for i in range(len(params.as_bytes)))
    params.as_bytes = raw
    assert params.as_bytes == raw
    with pytest.raises(ValueError):
        params.as_bytes = b"short"


def test_invalid_repr_attribute():
    val = driver.LaunchAttributeValue()
    string = str(val)


@pytest.mark.skipif(
    driver_version_less_than(12020)
    or not supportsCudaAPI("graph_add_node_v2")
    or not supportsCudaAPI("graph_node_set_params")
    or not supportsCudaAPI("graph_exec_node_set_params"),
    reason="Polymorphic graph APIs required",
)
def test_graph_poly():
    stream = driver.stream_create(0)

    # cuGraphAddNode

    # Create 2 buffers
    size = int(1024 * np.uint8().itemsize)
    buffers = []
    for _ in range(2):
        dptr = driver.mem_alloc_v2(size)
        buffers += [(np.full(size, 2).astype(np.uint8), dptr)]

    # Update dev buffers
    for host, device in buffers:
        driver.memcpy_htod_v2(device, host, size)

    # Create graph
    nodes = []
    graph = driver.graph_create(0)

    # Memset
    host, device = buffers[0]
    memsetParams = driver.GraphNodeParams()
    memsetParams.type = driver.GraphNodeType.MEMSET
    memsetParams.memset.element_size = np.uint8().itemsize
    memsetParams.memset.width = size
    memsetParams.memset.height = 1
    memsetParams.memset.dst = device
    memsetParams.memset.value = 1
    node = driver.graph_add_node_v2(graph, NO_DEPS, None, 0, memsetParams)
    nodes += [node]

    # Memcpy
    host, device = buffers[1]
    memcpyParams = driver.GraphNodeParams()
    memcpyParams.type = driver.GraphNodeType.MEMCPY
    memcpyParams.memcpy.copy_params.src_memory_type = driver.MemoryType.DEVICE
    memcpyParams.memcpy.copy_params.src_device = device
    memcpyParams.memcpy.copy_params.dst_memory_type = driver.MemoryType.HOST
    memcpyParams.memcpy.copy_params.dst_host = host.ctypes.data
    memcpyParams.memcpy.copy_params.width_in_bytes = size
    memcpyParams.memcpy.copy_params.height = 1
    memcpyParams.memcpy.copy_params.depth = 1
    node = driver.graph_add_node_v2(graph, NO_DEPS, None, 0, memcpyParams)
    nodes += [node]

    # Instantiate, execute, validate
    graphExec = driver.graph_instantiate(graph, 0)
    driver.graph_launch(graphExec, stream)
    driver.stream_synchronize(stream)

    # Validate
    for host, device in buffers:
        driver.memcpy_dtoh_v2(host, device, size)
    assert np.array_equal(buffers[0][0], np.full(size, 1).astype(np.uint8))
    assert np.array_equal(buffers[1][0], np.full(size, 2).astype(np.uint8))

    # cuGraphNodeSetParams
    host, device = buffers[1]
    memcpyParamsCopy = driver.graph_memcpy_node_get_params(nodes[1])
    assert int(memcpyParamsCopy.src_device) == int(device)
    host, device = buffers[0]
    memcpyParams.memcpy.copy_params.src_device = device
    driver.graph_node_set_params(nodes[1], memcpyParams)
    memcpyParamsCopy = driver.graph_memcpy_node_get_params(nodes[1])
    assert int(memcpyParamsCopy.src_device) == int(device)

    # cuGraphExecNodeSetParams
    memsetParams.memset.value = 11
    driver.graph_exec_node_set_params(graphExec, nodes[0], memsetParams)
    driver.graph_launch(graphExec, stream)
    driver.stream_synchronize(stream)
    driver.memcpy_dtoh_v2(buffers[0][0], buffers[0][1], size)
    assert np.array_equal(buffers[0][0], np.full(size, 11).astype(np.uint8))

    # Cleanup
    driver.mem_free_v2(buffers[0][1])
    driver.mem_free_v2(buffers[1][1])
    driver.graph_exec_destroy(graphExec)
    driver.graph_destroy(graph)
    driver.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(12040) or not supportsCudaAPI("device_get_dev_resource"),
    reason="Polymorphic graph APIs required",
)
def test_cuDeviceGetDevResource(device, ctx):
    resource_in = driver.device_get_dev_resource(device, driver.DevResourceType.SM)
    resource_from_ctx = driver.ctx_get_dev_resource(ctx, driver.DevResourceType.SM)
    for resource in (resource_in, resource_from_ctx):
        assert isinstance(resource, driver.DevResource_v1)
        assert resource.type == driver.DevResourceType.SM
        assert resource.sm.sm_count > 0

    # The wrapper queries the number of groups itself and returns all of them, plus the remainder
    res, remainder = driver.dev_sm_resource_split_by_count(resource_in, 0, 2)
    assert len(res) != 0
    assert isinstance(remainder, driver.DevResource_v1)
    assert len(res[:3]) == min(3, len(res))


@pytest.mark.skipif(
    driver_version_less_than(12030) or not supportsCudaAPI("graph_conditional_handle_create"),
    reason="Conditional graph APIs required",
)
def test_conditional(ctx):
    graph = driver.graph_create(0)
    handle = driver.graph_conditional_handle_create(graph, ctx, 0, 0)

    params = driver.GraphNodeParams()
    params.type = driver.GraphNodeType.CONDITIONAL
    params.conditional.handle = handle
    params.conditional.type = driver.GraphConditionalNodeType.IF_
    params.conditional.size_ = 1
    params.conditional.ctx = ctx

    # ph_graph_out is a pointer to an array of `size` graphs filled in by the driver
    assert params.conditional.ph_graph_out == 0
    node = driver.graph_add_node_v2(graph, NO_DEPS, None, 0, params)

    assert params.conditional.ph_graph_out != 0
    assert ctypes.c_void_p.from_address(params.conditional.ph_graph_out).value != 0


def test_CUmemDecompressParams_st():
    desc = driver.MemDecompressParams()
    assert int(desc.dst_act_bytes) == 0


def test_all_CUresult_codes():
    max_code = int(max(driver.Result))
    # Smoke test. CUDA_ERROR_UNKNOWN = 999, but intentionally using literal value.
    assert max_code >= 999
    num_good = 0
    for code in range(max_code + 2):  # One past max_code
        try:
            error = driver.Result(code)
        except ValueError:
            pass  # enum does not exist for this code
        else:
            try:
                name = driver.get_error_name(error)
            except driver.InvalidValueError:
                # enum exists but is not known to an older driver
                # (example: cuda-bindings built with CTK 12.8, driver from CTK 12.0)
                with pytest.raises(driver.InvalidValueError):
                    driver.get_error_string(error)
            else:
                assert name
                desc = driver.get_error_string(error)
                assert desc
                num_good += 1
    # Smoke test: Do we have at least some "good" codes?
    # The number will increase over time as new enums are added and support for
    # old CTKs is dropped, but it is not critical that this number is updated.
    assert num_good >= 76  # CTK 11.0.3_450.51.06


@pytest.mark.skipif(driver_version_less_than(12030), reason="Driver too old for cuKernelGetName")
def test_cuKernelGetName_failure():
    with pytest.raises(driver.InvalidValueError):
        driver.kernel_get_name(0)


@pytest.mark.skipif(driver_version_less_than(12030), reason="Driver too old for cuFuncGetName")
def test_cuFuncGetName_failure():
    with pytest.raises(driver.InvalidValueError):
        driver.func_get_name(0)


@pytest.mark.skipif(
    driver_version_less_than(12080) or not supportsCudaAPI("checkpoint_process_get_state"),
    reason="When API was introduced",
)
def test_cuCheckpointProcessGetState_failure():
    with pytest.raises(driver.DriverError):
        driver.checkpoint_process_get_state(123434)


@pytest.mark.skipif(
    driver_version_less_than(12090),
    reason="cuLogs* functions were introduced in CUDA 12.9",
)
def test_culogs_functions():
    # The driver rejects a NULL callback, so register a real (no-op) one.
    log_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)

    def log_callback(data, level, message, length):
        pass

    log_callback_c = log_callback_type(log_callback)
    try:
        callback_handle = driver.logs_register_callback(ctypes.cast(log_callback_c, ctypes.c_void_p).value, 0)

        driver.logs_current(0)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_path = f.name
        try:
            driver.logs_dump_to_file(0, log_path.encode(), 0)
        finally:
            os.remove(log_path)

        # cuLogsDumpToMemory's own docstring states the driver's internal log
        # buffer is capped at 25600 bytes, so a buffer of that size is always
        # large enough to hold the dump without risking a buffer overrun.
        buf = ctypes.create_string_buffer(25600)
        size = ctypes.c_size_t(len(buf))
        driver.logs_dump_to_memory(0, ctypes.addressof(buf), ctypes.addressof(size), 0)

        driver.logs_unregister_callback(callback_handle)
    except FunctionNotFoundError as e:
        pytest.fail(f"cuLogs* function unexpectedly not found on the driver: {e}")


def test_private_function_pointer_inspector():
    from cuda.bindings._internal.driver import _inspect_function_pointer

    assert _inspect_function_pointer("__cuGetErrorString") != 0


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_get_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphGetId(device, ctx):
    """Test cuGraphGetId - get graph ID."""
    graph = driver.graph_create(0)

    graph_id = driver.graph_get_id(graph)
    assert isinstance(graph_id, int)
    assert graph_id > 0

    # Create another graph and verify it has a different ID
    graph2 = driver.graph_create(0)
    graph_id2 = driver.graph_get_id(graph2)
    assert graph_id2 != graph_id

    driver.graph_destroy(graph)
    driver.graph_destroy(graph2)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_exec_get_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphExecGetId(device, ctx):
    """Test cuGraphExecGetId - get graph exec ID."""
    stream = driver.stream_create(0)

    graph = driver.graph_create(0)

    # Add an empty node to make the graph valid
    node = driver.graph_add_empty_node(graph, NO_DEPS, 0)

    graphExec = driver.graph_instantiate(graph, 0)

    graph_exec_id = driver.graph_exec_get_id(graphExec)
    assert isinstance(graph_exec_id, int)
    assert graph_exec_id > 0

    # Create another graph exec and verify it has a different ID
    graph2 = driver.graph_create(0)
    node2 = driver.graph_add_empty_node(graph2, NO_DEPS, 0)
    graphExec2 = driver.graph_instantiate(graph2, 0)
    graph_exec_id2 = driver.graph_exec_get_id(graphExec2)
    assert graph_exec_id2 != graph_exec_id

    driver.graph_exec_destroy(graphExec)
    driver.graph_exec_destroy(graphExec2)
    driver.graph_destroy(graph)
    driver.graph_destroy(graph2)
    driver.stream_destroy_v2(stream)


def test_cuGraphGetEdges_edgeData_outlives_call(device, ctx):
    # Regression test for https://github.com/NVIDIA/cuda-python/issues/1804
    # cuGraphGetEdges previously returned CUgraphEdgeData wrappers backed by
    # a scratch buffer that was freed before the call returned, leaving the
    # wrappers pointing at freed memory. Ensure the returned objects remain
    # readable after the call and after subsequent allocations.
    graph = driver.graph_create(0)
    try:
        n0 = driver.graph_add_empty_node(graph, NO_DEPS, 0)
        # A raw address of an array of handles is accepted ...
        deps = node_array([n0])
        n1 = driver.graph_add_empty_node(graph, ctypes.addressof(deps), 1)
        # ... as is a sequence of handles.
        n2 = driver.graph_add_empty_node(graph, [n0, n1], 2)

        from_nodes, to_nodes, edge_data = driver.graph_get_edges_v2(graph)
        num_edges = len(edge_data)
        assert num_edges == 3
        assert len(from_nodes) == len(to_nodes) == 3

        # Stir the heap to make a use-after-free more likely to surface.
        for _ in range(64):
            driver.graph_get_edges_v2(graph)
            driver.graph_node_get_dependencies_v2(n1)

        # Each wrapper must still own its data.
        for ed in edge_data:
            assert ed.from_port == 0
            assert ed.to_port == 0
            assert int(ed.type) == 0
            # CUgraphEdgeData_st layout: from_port(1), to_port(1), type(1), reserved[5]
            raw = (ctypes.c_uint8 * 8).from_address(ed.ptr)
            assert bytes(raw[3:]) == b"\x00" * 5
    finally:
        driver.graph_destroy(graph)


def test_cuGraphNodeGetDependencies_edgeData_outlives_call(device, ctx):
    # Companion regression test for #1804 covering the dependency-query path.
    graph = driver.graph_create(0)
    try:
        n0 = driver.graph_add_empty_node(graph, [], 0)
        n1 = driver.graph_add_empty_node(graph, [n0], 1)

        dependencies, edge_data = driver.graph_node_get_dependencies_v2(n1)
        num_deps = len(edge_data)
        assert num_deps == 1
        assert len(dependencies) == 1

        dependents, dep_edge_data = driver.graph_node_get_dependent_nodes_v2(n0)
        num_dependents = len(dep_edge_data)
        assert num_dependents == 1
        assert len(dependents) == 1

        for _ in range(64):
            driver.graph_node_get_dependencies_v2(n1)
            driver.graph_node_get_dependent_nodes_v2(n0)

        for ed in list(edge_data) + list(dep_edge_data):
            assert ed.from_port == 0
            assert ed.to_port == 0
            assert int(ed.type) == 0
            # CUgraphEdgeData_st layout: from_port(1), to_port(1), type(1), reserved[5]
            raw = (ctypes.c_uint8 * 8).from_address(ed.ptr)
            assert bytes(raw[3:]) == b"\x00" * 5
    finally:
        driver.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_local_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetLocalId(device, ctx):
    """Test cuGraphNodeGetLocalId - get node local ID."""
    graph = driver.graph_create(0)

    # Add multiple nodes
    node1 = driver.graph_add_empty_node(graph, NO_DEPS, 0)

    deps = node_array([node1])
    node2 = driver.graph_add_empty_node(graph, ctypes.addressof(deps), 1)

    deps = node_array([node1, node2])
    node3 = driver.graph_add_empty_node(graph, ctypes.addressof(deps), 2)

    # Get local IDs for each node
    node_id1 = driver.graph_node_get_local_id(node1)
    assert isinstance(node_id1, int)
    assert node_id1 >= 0

    node_id2 = driver.graph_node_get_local_id(node2)
    assert isinstance(node_id2, int)
    assert node_id2 >= 0
    assert node_id2 != node_id1

    node_id3 = driver.graph_node_get_local_id(node3)
    assert isinstance(node_id3, int)
    assert node_id3 >= 0
    assert node_id3 != node_id1
    assert node_id3 != node_id2

    driver.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_tools_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetToolsId(device, ctx):
    """Test cuGraphNodeGetToolsId - get node tools ID."""
    graph = driver.graph_create(0)

    node = driver.graph_add_empty_node(graph, NO_DEPS, 0)

    tools_node_id = driver.graph_node_get_tools_id(node)
    assert isinstance(tools_node_id, int)
    # toolsNodeId is unsigned long long, so it can be any non-negative value
    assert tools_node_id >= 0

    # Add another node and verify it has a different tools ID
    deps = node_array([node])
    node2 = driver.graph_add_empty_node(graph, ctypes.addressof(deps), 1)
    tools_node_id2 = driver.graph_node_get_tools_id(node2)
    assert tools_node_id2 != tools_node_id

    driver.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_containing_graph"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetContainingGraph(device, ctx):
    """Test cuGraphNodeGetContainingGraph - get graph containing a node."""
    graph = driver.graph_create(0)

    node = driver.graph_add_empty_node(graph, NO_DEPS, 0)

    # Get the containing graph
    containing_graph = driver.graph_node_get_containing_graph(node)
    # Verify it's the same graph
    assert int(containing_graph) == int(graph)

    # Test with a child graph node
    # Create a child graph node
    child_graph = driver.graph_create(0)
    child_node = driver.graph_add_empty_node(child_graph, NO_DEPS, 0)

    # Add child graph node to parent graph
    childGraphNodeParams = driver.GraphNodeParams()
    childGraphNodeParams.type = driver.GraphNodeType.GRAPH
    childGraphNodeParams.graph.graph = child_graph
    child_graph_node = driver.graph_add_node_v2(graph, NO_DEPS, None, 0, childGraphNodeParams)

    # Get containing graph for the child graph node
    containing_graph_for_child = driver.graph_node_get_containing_graph(child_graph_node)
    assert int(containing_graph_for_child) == int(graph)

    # Get containing graph for node inside child graph
    containing_graph_for_nested = driver.graph_node_get_containing_graph(child_node)
    assert int(containing_graph_for_nested) == int(child_graph)

    driver.graph_destroy(graph)
    driver.graph_destroy(child_graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("stream_get_dev_resource"),
    reason="Requires CUDA 13.1+",
)
def test_cuStreamGetDevResource(device, ctx):
    """Test cuStreamGetDevResource - get device resource from stream."""
    stream = driver.stream_create(0)

    # Get SM resource from stream
    resource = driver.stream_get_dev_resource(stream, driver.DevResourceType.SM)
    # Verify the resource was filled in
    assert resource.type == driver.DevResourceType.SM

    driver.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("dev_sm_resource_split"),
    reason="Requires CUDA 13.1+",
)
def test_cuDevSmResourceSplit(device, ctx):
    """Test cuDevSmResourceSplit - split SM resource into structured groups."""
    resource_in = driver.device_get_dev_resource(device, driver.DevResourceType.SM)

    # Test case 1: Split into 1 group
    nb_groups = 1
    group_params = driver.DevSmResourceGroupParams(nb_groups)
    # Set up group: request 4 SMs with coscheduled count of 2
    group_params[0].sm_count = 4
    group_params[0].coscheduled_sm_count = 2

    res, rem = driver.dev_sm_resource_split(nb_groups, resource_in, 0, group_params)
    assert len(res) == nb_groups
    assert isinstance(rem, driver.DevResource_v1)

    # Test case 2: Split into 2 groups (if device has enough SMs)
    # First, get the device resource again for a fresh split
    resource_in = driver.device_get_dev_resource(device, driver.DevResourceType.SM)

    nb_groups = 2
    group_params = driver.DevSmResourceGroupParams(nb_groups)
    # First group: request 4 SMs with coscheduled count of 2
    group_params[0].sm_count = 4
    group_params[0].coscheduled_sm_count = 2
    # Second group: request 4 SMs with coscheduled count of 2
    group_params[1].sm_count = 4
    group_params[1].coscheduled_sm_count = 2

    # This may succeed or fail depending on device SM count, but should handle gracefully
    try:
        res, rem = driver.dev_sm_resource_split(nb_groups, resource_in, 0, group_params)
    except (driver.InvalidResourceConfigurationError, driver.InvalidValueError):
        # If it fails, it should be due to insufficient resources, not a binding error
        pass
    else:
        assert len(res) == nb_groups

    # Test case 3: Empty list (0 groups) - should handle gracefully
    # Note: According to CUDA docs, nbGroups specifies number of groups, so 0 might not be valid
    # But we test that the binding accepts a NULL group list without crashing
    nb_groups = 0

    # If it fails, it should be a valid CUDA error, not a Python binding error
    with contextlib.suppress(driver.InvalidValueError, driver.InvalidResourceConfigurationError):
        driver.dev_sm_resource_split(nb_groups, resource_in, 0, None)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_memcpy_batch_async(device, ctx):
    stream = driver.stream_create(0)
    src = driver.mem_alloc_v2(16)
    dst = driver.mem_alloc_v2(16)
    try:
        driver.memcpy_htod_v2(src, np.arange(16, dtype=np.uint8), 16)
        attrs = driver.MemCpyAttributes_v1()
        attrs.src_access_order = driver.MemCpySrcAccessOrder.STREAM

        driver.memcpy_batch_async(
            np.array([dst], dtype=np.uint64),
            np.array([src], dtype=np.uint64),
            np.array([16], dtype=np.uint64),
            attrs,
            np.array([0], dtype=np.uint64),
            stream,
        )
        driver.stream_synchronize(stream)

        result = np.zeros(16, dtype=np.uint8)
        driver.memcpy_dtoh_v2(result, dst, 16)
        assert np.array_equal(result, np.arange(16, dtype=np.uint8))

        # Inconsistent buffers are rejected before reaching the C call.
        u64 = lambda *values: np.array(values, dtype=np.uint64)  # noqa: E731
        with pytest.raises(ValueError, match="same number of items"):
            driver.memcpy_batch_async(u64(1, 2), u64(1), u64(1), attrs, u64(0), stream)
        # Items of the wrong size would be reinterpreted by the C call.
        with pytest.raises(ValueError, match="8 bytes"):
            driver.memcpy_batch_async(np.array([1], dtype=np.uint32), u64(1), u64(1), attrs, u64(0), stream)
        with pytest.raises(ValueError, match="both be None or both be provided"):
            driver.memcpy_batch_async(u64(1), u64(1), u64(1), attrs, None, stream)
        with pytest.raises(ValueError, match="same number of items"):
            driver.memcpy_batch_async(u64(1), u64(1), u64(1), attrs, u64(0, 0), stream)
    finally:
        driver.mem_free_v2(src)
        driver.mem_free_v2(dst)
        driver.stream_destroy_v2(stream)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.skipif(not supportsCudaAPI("device_get_host_atomic_capabilities"), reason="Requires a newer CUDA")
def test_device_get_host_atomic_capabilities_item_size(device, ctx):
    operations = np.array([driver.AtomicOperation.INTEGER_ADD], dtype=np.int32)
    assert len(driver.device_get_host_atomic_capabilities(operations, device)) == 1

    # int64 items would be read as pairs of 4-byte operations.
    with pytest.raises(ValueError, match="4 bytes"):
        driver.device_get_host_atomic_capabilities(operations.astype(np.int64), device)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_get_error_string_and_name_reject_unknown_error_code():
    with pytest.raises(driver.InvalidValueError):
        driver.get_error_string(123456)
    with pytest.raises(driver.InvalidValueError):
        driver.get_error_name(123456)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_thread_exchange_stream_capture_mode(ctx):
    # The new mode is taken as an argument and the previous one is returned.
    previous = driver.thread_exchange_stream_capture_mode(driver.StreamCaptureMode.RELAXED)
    try:
        assert previous == driver.StreamCaptureMode.GLOBAL
        assert driver.thread_exchange_stream_capture_mode(driver.StreamCaptureMode.THREAD_LOCAL) == (
            driver.StreamCaptureMode.RELAXED
        )
        assert driver.thread_exchange_stream_capture_mode(previous) == driver.StreamCaptureMode.THREAD_LOCAL
    finally:
        driver.thread_exchange_stream_capture_mode(previous)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_driver_namespace_naming():
    # Enum members are named without the C prefix; the legacy full C names are not kept as aliases.
    assert driver.GraphConditionalNodeType.IF_ == 0
    assert not hasattr(driver.GraphConditionalNodeType, "CU_GRAPH_COND_TYPE_IF")
    assert not hasattr(driver.Result, "CUDA_SUCCESS")

    classes = {name: getattr(driver, name) for name in dir(driver) if isinstance(getattr(driver, name), type)}

    # Struct class names drop the C tag suffix.
    assert not [name for name in classes if name.endswith(("_st", "St"))]
    assert driver.StreamMemOpAtomicReductionParams is not None

    # Helper types that are used publicly are not underscore-prefixed.
    params = driver.DevSmResourceGroupParams(2)
    assert len(params) == 2
    assert not hasattr(driver, "_DevSmResourceGroupParams")

    # `cl_event_flags` (the NVCL scheduling enum) and `CUevent_flags` both want the name `EventFlags`.
    assert driver.EventFlags is not driver.NvclEventFlags
    assert driver.EventFlags.DEFAULT == 0
    assert driver.EventFlags.BLOCKING_SYNC == 1
    assert set(driver.NvclEventFlags.__members__) != set(driver.EventFlags.__members__)

    # Exceptions do not shadow Python builtins ...
    exceptions = [cls for cls in classes.values() if issubclass(cls, driver.DriverError)]
    assert exceptions
    assert [cls.__name__ for cls in exceptions if hasattr(builtins, cls.__name__)] == []
    # ... but the status codes of the builtin-named errors are still mapped, to the prefixed classes.
    assert issubclass(driver.DriverTimeoutError, driver.DriverError)
    assert issubclass(driver.DriverFileNotFoundError, driver.DriverError)
    assert not hasattr(driver, "TimeoutError")


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_pointer_member_is_raw_address():
    # The v2 structs never copy or take a reference to the memory a pointer member refers to: the member holds
    # exactly the address it is given, and the caller keeps the buffer alive.
    host = np.full(1024, 2, dtype=np.uint8)
    params = driver.GraphNodeParams()
    params.memcpy.copy_params.dst_host = host.ctypes.data
    assert params.memcpy.copy_params.dst_host == host.ctypes.data

    # Passing the buffer object itself is rejected rather than silently storing a dangling pointer.
    with pytest.raises(TypeError):
        params.memcpy.copy_params.dst_host = host

    ptr = ctypes.cast(params.memcpy.copy_params.dst_host, ctypes.POINTER(ctypes.c_uint8))
    assert np.all(np.ctypeslib.as_array(ptr, shape=(host.size,)) == 2)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_array_member_assignment_and_aliased_dealloc():
    # Counterpart of the legacy double-free / dangling-pointer regressions for list-valued struct members.
    # In the v2 structs the array member (`LaunchConfig.attrs`) is a raw address plus a separate count, so no
    # buffer is allocated or freed by the setter, and repeated assignment must not corrupt anything.
    attrs = ctypes.create_string_buffer(8 * ctypes.sizeof(ctypes.c_uint64) * 8)
    outer = driver.LaunchConfig()
    outer.attrs = ctypes.addressof(attrs)
    outer.num_attrs = 4
    outer.attrs = 0
    outer.num_attrs = 0
    outer.attrs = ctypes.addressof(attrs)
    outer.num_attrs = 8
    assert outer.attrs == ctypes.addressof(attrs)
    assert outer.num_attrs == 8

    # A wrapper aliasing the same struct must leave the member untouched when it is destroyed.
    inner = driver.LaunchConfig.from_ptr(outer.ptr)
    assert inner.attrs == ctypes.addressof(attrs)
    del inner
    assert outer.attrs == ctypes.addressof(attrs)
    assert outer.num_attrs == 8
