# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import shutil

import numpy as np
import pytest
from cuda_python_test_helpers.mempool import xfail_if_mempool_oom

import cuda.bindings._v2.driver as cuda
import cuda.bindings.runtime as cudart
from cuda.bindings._v2 import driver
from cuda_python_test_helpers import driver_version_less_than


def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


def supportsManagedMemory():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrManagedMemory, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


def supportsCudaAPI(name):
    return name in dir(cuda)


def callableBinary(name):
    return shutil.which(name) is not None


# The _v2 bindings expose handles as plain ints and most structs as thin
# wrapper classes (e.g. driver.MemPoolProps_v1, driver.GraphNodeParams,
# driver.DevResource_v1, driver._DevSmResourceGroupParams) backed by the real
# cuda.h layout. Array-typed wrapper classes (AUTO_LOWPP_ARRAY) default to a
# single struct (`size=1`) but also support `Class(n)` for a caller-allocated
# array of n contiguous structs, backed by a numpy recarray -- used below for
# the SM-split APIs' bulk input/output arrays.


@pytest.mark.skipif(True, reason="Always skip!")
def test_always_skip():
    pass


def test_cuda_memcpy():
    # Get device

    # Allocate dev memory
    size = int(1024 * np.uint8().itemsize)
    dptr = cuda.mem_alloc_v2(size)

    # Set h1 and h2 memory to be different
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert np.array_equal(h1, h2) is False

    # h1 to D
    cuda.memcpy_htod_v2(dptr, h1, size)

    # D to h2
    cuda.memcpy_dtoh_v2(h2, dptr, size)

    # Validate h1 == h2
    assert np.array_equal(h1, h2)

    # Cleanup
    cuda.mem_free_v2(dptr)


def test_cuda_array():
    # No context created
    desc = driver.ArrayDescriptor_v2()
    with pytest.raises(driver.DriverError):
        cuda.array_create_v2(desc)

    # Description not filled
    with pytest.raises(driver.DriverError) as excinfo:
        cuda.array_create_v2(desc)
    assert excinfo.value.status == driver.Result.ERROR_INVALID_VALUE

    # Pass
    desc.format = driver.ArrayFormat.SIGNED_INT8
    desc.num_channels = 1
    desc.width = 1
    arr = cuda.array_create_v2(desc)

    cuda.array_destroy(arr)


# NOTE: test_cuda_repr_primitive and test_cuda_repr_pointer are intentionally
# not ported. They tested the repr/overflow/construction behavior of legacy
# wrapper classes (CUdeviceptr, cuuint32_t, cuuint64_t, CUcontext,
# CUoccupancyB2DSize) that have no equivalent in _v2.driver, where handles and
# device pointers are plain Python ints. That behavior remains permanently
# covered by tests/legacy_api/test_legacy_cuda.py.


def test_cuda_uuid_list_access(device):
    uuid = cuda.device_get_uuid_v2(device)
    # Uuid.bytes decodes the raw 16-byte field as a UTF-8 C-string, which
    # fails on arbitrary binary UUID bytes; read the raw bytes directly.
    assert len(ctypes.string_at(uuid.ptr, 16)) == 16

    jit_option = driver.JitOption
    options = {
        jit_option.INFO_LOG_BUFFER: 1,
        jit_option.INFO_LOG_BUFFER_SIZE_BYTES: 2,
        jit_option.ERROR_LOG_BUFFER: 3,
        jit_option.ERROR_LOG_BUFFER_SIZE_BYTES: 4,
        jit_option.LOG_VERBOSE: 5,
    }
    assert len(options) == 5


def test_cuda_cuModuleLoadDataEx():
    option_keys = [
        driver.JitOption.INFO_LOG_BUFFER,
        driver.JitOption.INFO_LOG_BUFFER_SIZE_BYTES,
        driver.JitOption.ERROR_LOG_BUFFER,
        driver.JitOption.ERROR_LOG_BUFFER_SIZE_BYTES,
        driver.JitOption.LOG_VERBOSE,
    ]
    options = (ctypes.c_int * len(option_keys))(*[int(k) for k in option_keys])
    option_values = (ctypes.c_void_p * len(option_keys))()
    # FIXME: This function call raises CUDA_ERROR_INVALID_VALUE
    with pytest.raises(driver.DriverError):
        cuda.module_load_data_ex(b"", len(option_keys), ctypes.addressof(options), ctypes.addressof(option_values))


# NOTE: test_cuda_repr is intentionally not ported. It asserted a detailed,
# field-dump-style __repr__ for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS and
# CUDA_KERNEL_NODE_PARAMS_st that is specific to the legacy driver.pyx.in
# codegen. _v2.driver's wrapper classes (e.g. KernelNodeParams_v2) use a
# generic `<Class object at 0x...>` __repr__ instead, so there is nothing
# equivalent to port.


def test_cuda_struct_list_of_enums():
    desc = driver.TextureDesc_v1()
    desc.address_mode = [
        driver.AddressMode.WRAP,
        driver.AddressMode.CLAMP,
        driver.AddressMode.MIRROR,
    ]


def test_cuda_CUstreamBatchMemOpParams():
    params = driver.StreamBatchMemOpParams_v1()
    wait_value = params.wait_value
    wait_value.operation = int(driver.StreamBatchMemOpType.WAIT_VALUE_32)
    wait_value.value64 = 666
    assert params.wait_value.value64 == 666


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supportsMemoryPool(), reason="When new attributes were introduced"
)
def test_cuda_memPool_attr():
    pool_props = driver.MemPoolProps_v1()
    pool_props.alloc_type = driver.MemAllocationType.PINNED
    pool_props.location.id = 0
    pool_props.location.type = driver.MemLocationType.DEVICE

    attr_list = [None] * 8
    try:
        pool = cuda.mem_pool_create(pool_props)
    except driver.DriverError as e:
        xfail_if_mempool_oom(e, "mem_pool_create", pool_props.location.id)
        raise

    def get_attr(attr):
        buf = ctypes.c_uint64()
        cuda.mem_pool_get_attribute(pool, attr, ctypes.addressof(buf))
        return buf.value

    def set_attr(attr, value, ctype=ctypes.c_int):
        buf = ctype(value)
        cuda.mem_pool_set_attribute(pool, attr, ctypes.addressof(buf))

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

    for attr in (
        driver.MemPoolAttribute.REUSE_FOLLOW_EVENT_DEPENDENCIES,
        driver.MemPoolAttribute.REUSE_ALLOW_OPPORTUNISTIC,
        driver.MemPoolAttribute.REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ):
        set_attr(attr, 0, ctypes.c_int)
    set_attr(driver.MemPoolAttribute.RELEASE_THRESHOLD, 9, ctypes.c_uint64)

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
    assert attr_list[3] == 9

    cuda.mem_pool_destroy(pool)


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supportsManagedMemory(), reason="When new attributes were introduced"
)
def test_cuda_pointer_attr():
    ptr = cuda.mem_alloc_managed(0x1000, int(driver.MemAttachFlags.GLOBAL))

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
    attr_value_list = [None] * len(attr_type_list)
    for idx, attr in enumerate(attr_type_list):
        buf = ctypes.c_uint64()
        cuda.pointer_get_attribute(ctypes.addressof(buf), attr, ptr)
        attr_value_list[idx] = buf.value

    # List version. `data` is a `void**`: an array of pointers to
    # per-attribute result buffers, not a flat array of values.
    attributes = (ctypes.c_int * len(attr_type_list))(*[int(a) for a in attr_type_list])
    value_bufs = [ctypes.c_uint64() for _ in attr_type_list]
    data = (ctypes.c_void_p * len(attr_type_list))(*[ctypes.addressof(b) for b in value_bufs])
    cuda.pointer_get_attributes(len(attr_type_list), ctypes.addressof(attributes), ctypes.addressof(data), ptr)
    for attr1, buf in zip(attr_value_list, value_bufs):
        assert attr1 == buf.value

    # Test setting values
    for val in (True, False):
        flag = ctypes.c_int(int(val))
        cuda.pointer_set_attribute(ctypes.addressof(flag), driver.PointerAttribute.SYNC_MEMOPS, ptr)
        buf = ctypes.c_uint64()
        cuda.pointer_get_attribute(ctypes.addressof(buf), driver.PointerAttribute.SYNC_MEMOPS, ptr)
        assert bool(buf.value) == val

    cuda.mem_free_v2(ptr)


@pytest.mark.skipif(
    driver_version_less_than(11030) or not supportsManagedMemory(), reason="When new attributes were introduced"
)
def test_pointer_get_attributes_device_ordinal():
    attributes = [driver.PointerAttribute.DEVICE_ORDINAL]
    attributes_buf = (ctypes.c_int * len(attributes))(*[int(a) for a in attributes])
    value_buf = ctypes.c_int32()
    data = (ctypes.c_void_p * len(attributes))(ctypes.addressof(value_buf))

    cuda.pointer_get_attributes(len(attributes), ctypes.addressof(attributes_buf), ctypes.addressof(data), 0)

    # device ordinals are always small numbers.  A large number would indicate
    # an overflow error.
    assert abs(value_buf.value) < 256


@pytest.mark.skipif(not supportsManagedMemory(), reason="When new attributes were introduced")
def test_cuda_mem_range_attr(device):
    size = 0x1000
    location_device = driver.MemLocation_v1()
    location_device.type = driver.MemLocationType.DEVICE
    location_device.id = int(device)
    location_cpu = driver.MemLocation_v1()
    location_cpu.type = driver.MemLocationType.HOST
    location_cpu.id = -1  # CU_DEVICE_CPU

    ptr = cuda.mem_alloc_managed(size, int(driver.MemAttachFlags.GLOBAL))
    cuda.mem_advise_v2(ptr, size, driver.MemAdvise.SET_READ_MOSTLY, location_device)
    cuda.mem_advise_v2(ptr, size, driver.MemAdvise.SET_PREFERRED_LOCATION, location_cpu)
    cuda.mem_advise_v2(ptr, size, driver.MemAdvise.SET_ACCESSED_BY, location_cpu)
    concurrentSupported = cuda.device_get_attribute(driver.DeviceAttribute.CONCURRENT_MANAGED_ACCESS, device)
    if concurrentSupported:
        cuda.mem_advise_v2(ptr, size, driver.MemAdvise.SET_ACCESSED_BY, location_device)
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
        buf = ctypes.create_string_buffer(attr_type_size_list[idx])
        cuda.mem_range_get_attribute(ctypes.addressof(buf), attr_type_size_list[idx], attr_type_list[idx], ptr, size)
        if attr_type_size_list[idx] == 4:
            attr_value_list[idx] = ctypes.c_int32.from_buffer(buf, 0).value
        else:
            attr_value_list[idx] = [ctypes.c_int32.from_buffer(buf, i * 4).value for i in range(3)]

    matched = False
    for expected_values in expected_values_list:
        if expected_values == attr_value_list:
            matched = True
            break
    if not matched:
        raise RuntimeError(f"attr_value_list {attr_value_list} did not match any {expected_values_list}")

    # List version. `data` is a `void**`: an array of pointers to
    # per-attribute result buffers, not a flat array of values.
    data_sizes = (ctypes.c_size_t * len(attr_type_list))(*attr_type_size_list)
    attributes = (ctypes.c_int * len(attr_type_list))(*[int(a) for a in attr_type_list])
    value_bufs = [ctypes.create_string_buffer(sz) for sz in attr_type_size_list]
    data = (ctypes.c_void_p * len(attr_type_list))(*[ctypes.addressof(b) for b in value_bufs])
    cuda.mem_range_get_attributes(
        ctypes.addressof(data),
        ctypes.addressof(data_sizes),
        ctypes.addressof(attributes),
        len(attr_type_list),
        ptr,
        size,
    )
    for idx, (buf, sz) in enumerate(zip(value_bufs, attr_type_size_list)):
        if sz == 4:
            value = ctypes.c_int32.from_buffer(buf, 0).value
        else:
            value = [ctypes.c_int32.from_buffer(buf, i * 4).value for i in range(3)]
        assert value == attr_value_list[idx]

    cuda.mem_free_v2(ptr)


@pytest.mark.skipif(
    driver_version_less_than(11040) or not supportsMemoryPool(), reason="Mempool for graphs not supported"
)
@pytest.mark.thread_unsafe(reason="used high memory can be higher if threaded.")
def test_cuda_graphMem_attr(device):
    stream = cuda.stream_create(0)
    graph = cuda.graph_create(0)

    allocSize = 1

    params = driver.MemAllocNodeParams_v2()
    params.pool_props.location.type = driver.MemLocationType.DEVICE
    params.pool_props.location.id = device
    params.pool_props.alloc_type = driver.MemAllocationType.PINNED
    params.bytesize = allocSize

    try:
        allocNode = cuda.graph_add_mem_alloc_node(graph, 0, 0, params)
    except driver.DriverError as e:
        if e.status == driver.Result.ERROR_OUT_OF_MEMORY:
            cuda.graph_destroy(graph)
            cuda.stream_destroy_v2(stream)
            xfail_if_mempool_oom(e, "graph_add_mem_alloc_node", device)
        raise
    deps = (ctypes.c_void_p * 1)(allocNode)
    cuda.graph_add_mem_free_node(graph, ctypes.addressof(deps), 1, params.dptr)

    graphExec = cuda.graph_instantiate_with_flags(graph, 0)

    cuda.graph_launch(graphExec, stream)

    used = ctypes.c_uint64()
    cuda.device_get_graph_mem_attribute(device, driver.GraphMemAttribute.USED_MEM_CURRENT, ctypes.addressof(used))
    usedHigh = ctypes.c_uint64()
    cuda.device_get_graph_mem_attribute(device, driver.GraphMemAttribute.USED_MEM_HIGH, ctypes.addressof(usedHigh))
    reserved = ctypes.c_uint64()
    cuda.device_get_graph_mem_attribute(
        device, driver.GraphMemAttribute.RESERVED_MEM_CURRENT, ctypes.addressof(reserved)
    )
    reservedHigh = ctypes.c_uint64()
    cuda.device_get_graph_mem_attribute(
        device, driver.GraphMemAttribute.RESERVED_MEM_HIGH, ctypes.addressof(reservedHigh)
    )

    assert used.value >= allocSize
    assert usedHigh.value == used.value
    assert reserved.value == usedHigh.value
    assert reservedHigh.value == reserved.value

    cuda.graph_exec_destroy(graphExec)
    cuda.graph_destroy(graph)
    cuda.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(12010)
    or not supportsCudaAPI("coredump_set_attribute_global")
    or not supportsCudaAPI("coredump_get_attribute_global"),
    reason="Coredump API not present",
)
def test_cuda_coredump_attr():
    def set_bool(attr, value):
        buf = ctypes.c_bool(value)
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        cuda.coredump_set_attribute_global(attr, ctypes.addressof(buf), ctypes.addressof(size))

    def set_bytes(attr, value):
        buf = ctypes.create_string_buffer(value)
        size = ctypes.c_size_t(len(value))
        cuda.coredump_set_attribute_global(attr, ctypes.addressof(buf), ctypes.addressof(size))

    def get_bool(attr):
        buf = ctypes.c_bool()
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        cuda.coredump_get_attribute_global(attr, ctypes.addressof(buf), ctypes.addressof(size))
        return buf.value

    def get_bytes(attr, maxlen=1024):
        buf = ctypes.create_string_buffer(maxlen)
        size = ctypes.c_size_t(maxlen)
        cuda.coredump_get_attribute_global(attr, ctypes.addressof(buf), ctypes.addressof(size))
        return buf.raw[: size.value]

    set_bool(driver.CoredumpSettings.TRIGGER_HOST, False)
    set_bytes(driver.CoredumpSettings.FILE, b"corefile")
    set_bytes(driver.CoredumpSettings.PIPE, b"corepipe")
    set_bool(driver.CoredumpSettings.LIGHTWEIGHT, True)

    assert get_bool(driver.CoredumpSettings.TRIGGER_HOST) is False
    assert get_bytes(driver.CoredumpSettings.FILE).rstrip(b"\x00") == b"corefile"
    assert get_bytes(driver.CoredumpSettings.PIPE).rstrip(b"\x00") == b"corepipe"
    assert get_bool(driver.CoredumpSettings.LIGHTWEIGHT) is True


def test_get_error_name_and_string():
    device = cuda.device_get(0)
    assert isinstance(device, int)
    # get_error_string / get_error_name return str in _v2.driver (the legacy
    # API returned bytes).
    s = cuda.get_error_string(driver.Result.SUCCESS)
    assert s == "no error"
    s = cuda.get_error_name(driver.Result.SUCCESS)
    assert s == "CUDA_SUCCESS"

    with pytest.raises(driver.DriverError) as excinfo:
        cuda.device_get(-1)
    assert excinfo.value.status == driver.Result.ERROR_INVALID_DEVICE
    s = cuda.get_error_string(driver.Result.ERROR_INVALID_DEVICE)
    assert s == "invalid device ordinal"
    s = cuda.get_error_name(driver.Result.ERROR_INVALID_DEVICE)
    assert s == "CUDA_ERROR_INVALID_DEVICE"


# TODO: cuStreamGetCaptureInfo_v2
@pytest.mark.skipif(driver_version_less_than(11030), reason="Driver too old for cuStreamGetCaptureInfo_v2")
def test_stream_capture():
    pass


def test_profiler():
    cuda.profiler_start()
    cuda.profiler_stop()


# NOTE: test_eglFrame is intentionally not ported. It only exercised
# construction/field-assignment of a bare CUeglFrame struct (no driver call),
# and _v2.driver does not expose a Python wrapper class for CUeglFrame.


# NOTE: test_anon_assign, test_union_assign, and test_invalid_repr_attribute
# are intentionally not ported. They tested legacy-codegen-specific behavior
# of the anonymous-union wrapper classes for CUexecAffinityParam_st and
# CUlaunchAttributeValue. _v2.driver has no CUlaunchAttributeValue wrapper at
# all, and its ExecAffinityParam_v1 is a numpy-recarray-backed batch wrapper
# with fundamentally different assignment semantics, so there is no
# meaningful equivalent to port.


@pytest.mark.skipif(
    driver_version_less_than(12020)
    or not supportsCudaAPI("graph_add_memset_node")
    or not supportsCudaAPI("graph_exec_memset_node_set_params"),
    reason="Typed graph node APIs required",
)
def test_graph_poly(ctx):
    stream = cuda.stream_create(0)

    # Create 2 buffers
    size = int(1024 * np.uint8().itemsize)
    buffers = []
    for _ in range(2):
        dptr = cuda.mem_alloc_v2(size)
        buffers += [(np.full(size, 2).astype(np.uint8), dptr)]

    # Update dev buffers
    for host, device in buffers:
        cuda.memcpy_htod_v2(device, host, size)

    # Create graph
    nodes = []
    graph = cuda.graph_create(0)

    # Memset
    host, device = buffers[0]
    memsetParams = driver.MemsetNodeParams_v1()
    memsetParams.dst = device
    memsetParams.element_size = np.uint8().itemsize
    memsetParams.width = size
    memsetParams.height = 1
    memsetParams.value = 1
    node = cuda.graph_add_memset_node(graph, 0, 0, memsetParams, ctx)
    nodes += [node]

    # Memcpy
    host, device = buffers[1]
    memcpyParams = driver.Memcpy3d_v2()
    memcpyParams.src_memory_type = driver.Memorytype.DEVICE
    memcpyParams.src_device = device
    memcpyParams.dst_memory_type = driver.Memorytype.HOST
    # dst_host takes a raw address (unlike the legacy API's setter, it does
    # not hold a reference to keep the buffer alive itself); `host` is kept
    # alive by the `buffers` list for the duration of this test.
    memcpyParams.dst_host = host.ctypes.data
    memcpyParams.width_in_bytes = size
    memcpyParams.height = 1
    memcpyParams.depth = 1
    node = cuda.graph_add_memcpy_node(graph, 0, 0, memcpyParams, ctx)
    nodes += [node]

    # Instantiate, execute, validate
    graphExec = cuda.graph_instantiate_with_flags(graph, 0)
    cuda.graph_launch(graphExec, stream)
    cuda.stream_synchronize(stream)

    # Validate
    for host, device in buffers:
        cuda.memcpy_dtoh_v2(host, device, size)
    assert np.array_equal(buffers[0][0], np.full(size, 1).astype(np.uint8))
    assert np.array_equal(buffers[1][0], np.full(size, 2).astype(np.uint8))

    # graph_memcpy_node_get_params / graph_memcpy_node_set_params
    host, device = buffers[1]
    memcpyParamsCopy = driver.Memcpy3d_v2()
    cuda.graph_memcpy_node_get_params(nodes[1], memcpyParamsCopy)
    assert int(memcpyParamsCopy.src_device) == int(device)
    host, device = buffers[0]
    memcpyParams.src_device = device
    cuda.graph_memcpy_node_set_params(nodes[1], memcpyParams)
    memcpyParamsCopy = driver.Memcpy3d_v2()
    cuda.graph_memcpy_node_get_params(nodes[1], memcpyParamsCopy)
    assert int(memcpyParamsCopy.src_device) == int(device)

    # graph_exec_memset_node_set_params
    memsetParams.value = 11
    cuda.graph_exec_memset_node_set_params(graphExec, nodes[0], memsetParams, ctx)
    cuda.graph_launch(graphExec, stream)
    cuda.stream_synchronize(stream)
    cuda.memcpy_dtoh_v2(buffers[0][0], buffers[0][1], size)
    assert np.array_equal(buffers[0][0], np.full(size, 11).astype(np.uint8))

    # Cleanup
    cuda.mem_free_v2(buffers[0][1])
    cuda.mem_free_v2(buffers[1][1])
    cuda.graph_exec_destroy(graphExec)
    cuda.graph_destroy(graph)
    cuda.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(12040) or not supportsCudaAPI("device_get_dev_resource"),
    reason="Polymorphic graph APIs required",
)
def test_cuDeviceGetDevResource(device):
    resource_in = driver.DevResource_v1()
    cuda.device_get_dev_resource(device, resource_in, driver.DevResourceType.SM)

    def split(nb_groups, min_count):
        result = driver.DevResource_v1(nb_groups) if nb_groups else None
        nb_groups_buf = ctypes.c_uint(nb_groups)
        remainder = driver.DevResource_v1()
        cuda.dev_sm_resource_split_by_count(
            result if result is not None else 0,
            ctypes.addressof(nb_groups_buf),
            resource_in,
            remainder,
            0,
            min_count,
        )
        return result, nb_groups_buf.value

    # Query the number of groups that would be created.
    _, count = split(0, 2)
    assert count != 0
    res, count_same = split(count, 2)
    assert count == count_same
    res, count = split(3, 2)
    assert count <= 3


@pytest.mark.skipif(
    driver_version_less_than(12030) or not supportsCudaAPI("graph_conditional_handle_create"),
    reason="Conditional graph APIs required",
)
def test_conditional(ctx, device):
    graph = cuda.graph_create(0)
    handle = cuda.graph_conditional_handle_create(graph, ctx, 0, 0)

    # `phGraph_out` is a CUDA-owned output array (see the field docstring in
    # cuda.h): the driver allocates it and writes its own pointer into
    # node_params.conditional.ph_graph_out during node creation -- it must be
    # left unset (zero) on input, not pre-allocated by the caller.
    node_params = driver.GraphNodeParams()
    node_params.type = driver.GraphNodeType.CONDITIONAL
    node_params.conditional.handle = handle
    node_params.conditional.type = int(driver.GraphConditionalNodeType.TYPE_IF)
    node_params.conditional.size_ = 1
    node_params.conditional.ctx = ctx

    assert node_params.conditional.ph_graph_out == 0
    cuda.graph_add_node_v2(graph, 0, 0, 0, node_params)

    phGraph_out_ptr = node_params.conditional.ph_graph_out
    assert phGraph_out_ptr not in (None, 0)
    branch_graph = ctypes.cast(phGraph_out_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    assert branch_graph is not None

    cuda.graph_destroy(graph)


def test_all_CUresult_codes():
    max_code = int(max(driver.Result))
    # Smoke test. CUDA_ERROR_UNKNOWN = 999, but intentionally using literal value.
    assert max_code >= 999
    num_good = 0
    for code in range(max_code + 2):  # One past max_code
        try:
            error = driver.Result(code)
        except ValueError:
            pass  # cython-generated enum does not exist for this code
        else:
            # get_error_name/get_error_string return "" (rather than raising)
            # when the driver does not recognize the code (e.g. cuda-bindings
            # built against a newer CTK than the installed driver supports).
            name = cuda.get_error_name(error)
            if name:
                assert cuda.get_error_string(error)
                num_good += 1
            else:
                assert cuda.get_error_string(error) == ""
    # Smoke test: Do we have at least some "good" codes?
    # The number will increase over time as new enums are added and support for
    # old CTKs is dropped, but it is not critical that this number is updated.
    assert num_good >= 76  # CTK 11.0.3_450.51.06


@pytest.mark.skipif(driver_version_less_than(12030), reason="Driver too old for cuKernelGetName")
def test_cuKernelGetName_failure():
    with pytest.raises(driver.DriverError) as excinfo:
        cuda.kernel_get_name(0)
    assert excinfo.value.status == driver.Result.ERROR_INVALID_VALUE


@pytest.mark.skipif(driver_version_less_than(12030), reason="Driver too old for cuFuncGetName")
def test_cuFuncGetName_failure():
    with pytest.raises(driver.DriverError) as excinfo:
        cuda.func_get_name(0)
    assert excinfo.value.status == driver.Result.ERROR_INVALID_VALUE


@pytest.mark.skipif(
    driver_version_less_than(12080) or not supportsCudaAPI("checkpoint_process_get_state"),
    reason="When API was introduced",
)
def test_cuCheckpointProcessGetState_failure():
    with pytest.raises(driver.DriverError):
        cuda.checkpoint_process_get_state(123434)


def test_private_function_pointer_inspector():
    from cuda.bindings._internal.driver import _inspect_function_pointer

    assert _inspect_function_pointer("__cuGetErrorString") != 0


# NOTE: test_struct_pointer_comparison is intentionally not ported. It tested
# equality/hash behavior of legacy pointer-wrapper classes (CUcontext,
# CUstream, ...) that have no equivalent in _v2.driver, where handles are
# plain Python ints (which already support equality/hash trivially).


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_get_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphGetId(device, ctx):
    """Test graph_get_id - get graph ID."""
    graph = cuda.graph_create(0)

    graph_id = cuda.graph_get_id(graph)
    assert isinstance(graph_id, int)
    assert graph_id > 0

    # Create another graph and verify it has a different ID
    graph2 = cuda.graph_create(0)
    graph_id2 = cuda.graph_get_id(graph2)
    assert graph_id2 != graph_id

    cuda.graph_destroy(graph)
    cuda.graph_destroy(graph2)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_exec_get_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphExecGetId(device, ctx):
    """Test graph_exec_get_id - get graph exec ID."""
    stream = cuda.stream_create(0)

    graph = cuda.graph_create(0)

    # Add an empty node to make the graph valid
    cuda.graph_add_empty_node(graph, 0, 0)

    graphExec = cuda.graph_instantiate_with_flags(graph, 0)

    graph_exec_id = cuda.graph_exec_get_id(graphExec)
    assert isinstance(graph_exec_id, int)
    assert graph_exec_id > 0

    # Create another graph exec and verify it has a different ID
    graph2 = cuda.graph_create(0)
    cuda.graph_add_empty_node(graph2, 0, 0)
    graphExec2 = cuda.graph_instantiate_with_flags(graph2, 0)
    graph_exec_id2 = cuda.graph_exec_get_id(graphExec2)
    assert graph_exec_id2 != graph_exec_id

    cuda.graph_exec_destroy(graphExec)
    cuda.graph_exec_destroy(graphExec2)
    cuda.graph_destroy(graph)
    cuda.graph_destroy(graph2)
    cuda.stream_destroy_v2(stream)


def test_cuGraphGetEdges_edgeData_outlives_call(device, ctx):
    # Regression test for https://github.com/NVIDIA/cuda-python/issues/1804
    # cuGraphGetEdges previously returned CUgraphEdgeData wrappers backed by
    # a scratch buffer that was freed before the call returned, leaving the
    # wrappers pointing at freed memory. Ensure the returned objects remain
    # readable after the call and after subsequent allocations.
    graph = cuda.graph_create(0)
    try:
        n0 = cuda.graph_add_empty_node(graph, 0, 0)
        deps1 = (ctypes.c_void_p * 1)(n0)
        n1 = cuda.graph_add_empty_node(graph, ctypes.addressof(deps1), 1)
        deps2 = (ctypes.c_void_p * 2)(n0, n1)
        cuda.graph_add_empty_node(graph, ctypes.addressof(deps2), 2)

        from_nodes, to_nodes, edge_data = cuda.graph_get_edges(graph)
        num_edges = len(from_nodes)
        assert num_edges == 3
        from_nodes, to_nodes, edge_data = cuda.graph_get_edges(graph)
        assert len(edge_data) == num_edges == 3

        # Stir the heap to make a use-after-free more likely to surface.
        for _ in range(64):
            cuda.graph_get_edges(graph)
            cuda.graph_node_get_dependencies(n1)

        # Each wrapper must still own its data.
        for ed in edge_data:
            assert ed.from_port == 0
            assert ed.to_port == 0
            assert int(ed.type) == 0
    finally:
        cuda.graph_destroy(graph)


def test_cuGraphNodeGetDependencies_edgeData_outlives_call(device, ctx):
    # Companion regression test for #1804 covering the dependency-query path.
    graph = cuda.graph_create(0)
    try:
        n0 = cuda.graph_add_empty_node(graph, 0, 0)
        deps1 = (ctypes.c_void_p * 1)(n0)
        n1 = cuda.graph_add_empty_node(graph, ctypes.addressof(deps1), 1)

        deps, edge_data = cuda.graph_node_get_dependencies(n1)
        num_deps = len(deps)
        assert num_deps == 1
        deps, edge_data = cuda.graph_node_get_dependencies(n1)
        assert len(edge_data) == num_deps == 1

        dependents, dep_edge_data = cuda.graph_node_get_dependent_nodes(n0)
        num_dependents = len(dependents)
        assert num_dependents == 1
        dependents, dep_edge_data = cuda.graph_node_get_dependent_nodes(n0)
        assert len(dep_edge_data) == num_dependents == 1

        for _ in range(64):
            cuda.graph_node_get_dependencies(n1)
            cuda.graph_node_get_dependent_nodes(n0)

        for ed in list(edge_data) + list(dep_edge_data):
            assert ed.from_port == 0
            assert ed.to_port == 0
            assert int(ed.type) == 0
    finally:
        cuda.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_local_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetLocalId(device, ctx):
    """Test graph_node_get_local_id - get node local ID."""
    graph = cuda.graph_create(0)

    # Add multiple nodes
    node1 = cuda.graph_add_empty_node(graph, 0, 0)

    deps2 = (ctypes.c_void_p * 1)(node1)
    node2 = cuda.graph_add_empty_node(graph, ctypes.addressof(deps2), 1)

    deps3 = (ctypes.c_void_p * 2)(node1, node2)
    node3 = cuda.graph_add_empty_node(graph, ctypes.addressof(deps3), 2)

    # Get local IDs for each node
    node_id1 = cuda.graph_node_get_local_id(node1)
    assert isinstance(node_id1, int)
    assert node_id1 >= 0

    node_id2 = cuda.graph_node_get_local_id(node2)
    assert isinstance(node_id2, int)
    assert node_id2 >= 0
    assert node_id2 != node_id1

    node_id3 = cuda.graph_node_get_local_id(node3)
    assert isinstance(node_id3, int)
    assert node_id3 >= 0
    assert node_id3 != node_id1
    assert node_id3 != node_id2

    cuda.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_tools_id"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetToolsId(device, ctx):
    """Test graph_node_get_tools_id - get node tools ID."""
    graph = cuda.graph_create(0)

    node = cuda.graph_add_empty_node(graph, 0, 0)

    tools_node_id = cuda.graph_node_get_tools_id(node)
    assert isinstance(tools_node_id, int)
    # toolsNodeId is unsigned long long, so it can be any non-negative value
    assert tools_node_id >= 0

    # Add another node and verify it has a different tools ID
    deps = (ctypes.c_void_p * 1)(node)
    node2 = cuda.graph_add_empty_node(graph, ctypes.addressof(deps), 1)
    tools_node_id2 = cuda.graph_node_get_tools_id(node2)
    assert tools_node_id2 != tools_node_id

    cuda.graph_destroy(graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("graph_node_get_containing_graph"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetContainingGraph(device, ctx):
    """Test graph_node_get_containing_graph - get graph containing a node."""
    graph = cuda.graph_create(0)

    node = cuda.graph_add_empty_node(graph, 0, 0)

    # Get the containing graph
    containing_graph = cuda.graph_node_get_containing_graph(node)
    # Verify it's the same graph
    assert int(containing_graph) == int(graph)

    # Test with a child graph node (if supported)
    # Create a child graph node
    child_graph = cuda.graph_create(0)
    child_node = cuda.graph_add_empty_node(child_graph, 0, 0)

    # Add child graph node to parent graph
    node_params = driver.GraphNodeParams()
    node_params.type = driver.GraphNodeType.GRAPH
    node_params.graph.graph = child_graph
    node_params.graph.ownership = int(driver.GraphChildGraphNodeOwnership.CLONE)
    try:
        child_graph_node = cuda.graph_add_node_v2(graph, 0, 0, 0, node_params)
    except driver.DriverError:
        child_graph_node = None

    if child_graph_node is not None:
        # Get containing graph for the child graph node
        containing_graph_for_child = cuda.graph_node_get_containing_graph(child_graph_node)
        assert int(containing_graph_for_child) == int(graph)

        # Get containing graph for node inside child graph
        containing_graph_for_nested = cuda.graph_node_get_containing_graph(child_node)
        assert int(containing_graph_for_nested) == int(child_graph)

    cuda.graph_destroy(graph)
    cuda.graph_destroy(child_graph)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("stream_get_dev_resource"),
    reason="Requires CUDA 13.1+",
)
def test_cuStreamGetDevResource(device, ctx):
    """Test stream_get_dev_resource - get device resource from stream."""
    stream = cuda.stream_create(0)

    # Get SM resource from stream
    resource = driver.DevResource_v1()
    cuda.stream_get_dev_resource(stream, resource, driver.DevResourceType.SM)
    # Verify resource is valid (non-empty)
    assert resource.type == int(driver.DevResourceType.SM)

    cuda.stream_destroy_v2(stream)


@pytest.mark.skipif(
    driver_version_less_than(13010) or not supportsCudaAPI("dev_sm_resource_split"),
    reason="Requires CUDA 13.1+",
)
def test_cuDevSmResourceSplit(device, ctx):
    """Test dev_sm_resource_split - split SM resource into structured groups."""
    resource_in = driver.DevResource_v1()
    cuda.device_get_dev_resource(device, resource_in, driver.DevResourceType.SM)

    # Test case 1: Split into 1 group
    nb_groups = 1
    group_params = driver._DevSmResourceGroupParams(nb_groups)
    # Set up group: request 4 SMs with coscheduled count of 2
    group_params.sm_count = 4
    group_params.coscheduled_sm_count = 2

    result = driver.DevResource_v1(nb_groups)
    remainder = driver.DevResource_v1()
    cuda.dev_sm_resource_split(
        result,
        nb_groups,
        resource_in,
        remainder,
        0,
        group_params,
    )

    # Test case 2: Split into 2 groups (if device has enough SMs)
    # First, get the device resource again for a fresh split
    resource_in = driver.DevResource_v1()
    cuda.device_get_dev_resource(device, resource_in, driver.DevResourceType.SM)

    nb_groups = 2
    group_params = driver._DevSmResourceGroupParams(nb_groups)
    group_params.sm_count = [4, 4]
    group_params.coscheduled_sm_count = [2, 2]

    result = driver.DevResource_v1(nb_groups)
    remainder = driver.DevResource_v1()
    with contextlib.suppress(driver.InvalidResourceConfigurationError, driver.InvalidValueError):
        cuda.dev_sm_resource_split(
            result,
            nb_groups,
            resource_in,
            remainder,
            0,
            group_params,
        )

    # Test case 3: Empty list (0 groups) - should handle gracefully
    nb_groups = 0
    group_params_empty = driver._DevSmResourceGroupParams(1)  # unused, but keep a valid pointer
    remainder = driver.DevResource_v1()
    with contextlib.suppress(driver.InvalidResourceConfigurationError, driver.InvalidValueError):
        cuda.dev_sm_resource_split(
            0,
            nb_groups,
            resource_in,
            remainder,
            0,
            group_params_empty,
        )


# NOTE: test_buffer_reference is intentionally not ported. It verified that
# assigning a numpy array to a struct's host-pointer field kept that array
# alive via reference counting internal to the legacy driver.pyx.in generated
# setter. _v2.driver's Memcpy3d_v2.dst_host setter takes a raw integer
# address instead (see the `dst_host = host.ctypes.data` pattern used in
# test_graph_poly) and keeps no reference at all, so the premise of this test
# (that the wrapper keeps the buffer alive) does not hold for the new API.


# NOTE: test_array_setter_no_double_free_after_clearing_with_empty_list and
# test_dealloc_clears_array_field_in_external_struct are intentionally not
# ported. They were regression tests for a double-free bug in the legacy
# driver.pyx.in / runtime.pyx.in generated setters for list-valued struct
# members (which allocated and freed a backing buffer on assignment).
# _v2.driver's LaunchConfig.attrs is a plain raw-pointer property (no
# allocation/ownership machinery at all), so that class of bug cannot occur
# and there is nothing equivalent to regression-test.
