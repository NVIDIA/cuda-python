# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import platform
import shutil
import textwrap

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import numpy as np
import pytest
from cuda.bindings import driver
from cuda_python_test_helpers.managed_memory import managed_memory_skip_reason


def driverVersionLessThan(target):
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, version = cuda.cuDriverGetVersion()
    assert err == cuda.CUresult.CUDA_SUCCESS
    return version < target


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


def skip_if_concurrent_managed_access_disabled():
    reason = managed_memory_skip_reason()
    if reason:
        pytest.skip(reason)


def test_cuda_memcpy():
    # Get device

    # Allocate dev memory
    size = int(1024 * np.uint8().itemsize)
    err, dptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Set h1 and h2 memory to be different
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert np.array_equal(h1, h2) is False

    # h1 to D
    (err,) = cuda.cuMemcpyHtoD(dptr, h1, size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # D to h2
    (err,) = cuda.cuMemcpyDtoH(h2, dptr, size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Validate h1 == h2
    assert np.array_equal(h1, h2)

    # Cleanup
    (err,) = cuda.cuMemFree(dptr)
    assert err == cuda.CUresult.CUDA_SUCCESS


def test_cuda_array():
    # No context created
    desc = cuda.CUDA_ARRAY_DESCRIPTOR()
    err, arr = cuda.cuArrayCreate(desc)
    assert err == cuda.CUresult.CUDA_ERROR_INVALID_CONTEXT or err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE

    # Desciption not filled
    err, arr = cuda.cuArrayCreate(desc)
    assert err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE

    # Pass
    desc.Format = cuda.CUarray_format.CU_AD_FORMAT_SIGNED_INT8
    desc.NumChannels = 1
    desc.Width = 1
    err, arr = cuda.cuArrayCreate(desc)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuArrayDestroy(arr)
    assert err == cuda.CUresult.CUDA_SUCCESS


def test_cuda_repr_primitive(device, ctx):
    assert str(device) == "<CUdevice 0>"
    assert int(device) == 0

    assert str(ctx).startswith("<CUcontext 0x")
    assert int(ctx) > 0
    assert hex(ctx) == hex(int(ctx))

    # CUdeviceptr
    err, dptr = cuda.cuMemAlloc(1024 * np.uint8().itemsize)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert str(dptr).startswith("<CUdeviceptr ")
    assert int(dptr) > 0
    (err,) = cuda.cuMemFree(dptr)
    size = 7
    dptr = cuda.CUdeviceptr(size)
    assert str(dptr) == f"<CUdeviceptr {size}>"
    assert int(dptr) == size
    size = 4294967295
    dptr = cuda.CUdeviceptr(size)
    assert str(dptr) == f"<CUdeviceptr {size}>"
    assert int(dptr) == size
    size = 18446744073709551615
    dptr = cuda.CUdeviceptr(size)
    assert str(dptr) == f"<CUdeviceptr {size}>"
    assert int(dptr) == size

    # cuuint32_t
    size = 7
    int32 = cuda.cuuint32_t(size)
    assert str(int32) == f"<cuuint32_t {size}>"
    assert int(int32) == size
    size = 4294967295
    int32 = cuda.cuuint32_t(size)
    assert str(int32) == f"<cuuint32_t {size}>"
    assert int(int32) == size
    size = 18446744073709551615
    try:
        int32 = cuda.cuuint32_t(size)
        raise RuntimeError("int32 = cuda.cuuint32_t(18446744073709551615) did not fail")
    except OverflowError as err:
        pass

    # cuuint64_t
    size = 7
    int64 = cuda.cuuint64_t(size)
    assert str(int64) == f"<cuuint64_t {size}>"
    assert int(int64) == size
    size = 4294967295
    int64 = cuda.cuuint64_t(size)
    assert str(int64) == f"<cuuint64_t {size}>"
    assert int(int64) == size
    size = 18446744073709551615
    int64 = cuda.cuuint64_t(size)
    assert str(int64) == f"<cuuint64_t {size}>"
    assert int(int64) == size


def test_cuda_repr_pointer(ctx):
    # Test 1: Classes representing pointers
    assert str(ctx).startswith("<CUcontext 0x")
    assert int(ctx) > 0
    assert hex(ctx) == hex(int(ctx))
    randomCtxPointer = 12345
    randomCtx = cuda.CUcontext(randomCtxPointer)
    assert str(randomCtx) == f"<CUcontext {hex(randomCtxPointer)}>"
    assert int(randomCtx) == randomCtxPointer
    assert hex(randomCtx) == hex(randomCtxPointer)

    # Test 2: Function pointers
    func = 12345
    b2d_cb = cuda.CUoccupancyB2DSize(func)
    assert str(b2d_cb) == f"<CUoccupancyB2DSize {hex(func)}>"
    assert int(b2d_cb) == func
    assert hex(b2d_cb) == hex(func)


def test_cuda_uuid_list_access(device):
    err, uuid = cuda.cuDeviceGetUuid(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert len(uuid.bytes) <= 16

    jit_option = cuda.CUjit_option
    options = {
        jit_option.CU_JIT_INFO_LOG_BUFFER: 1,
        jit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: 2,
        jit_option.CU_JIT_ERROR_LOG_BUFFER: 3,
        jit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: 4,
        jit_option.CU_JIT_LOG_VERBOSE: 5,
    }


def test_cuda_cuModuleLoadDataEx():
    option_keys = [
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_LOG_VERBOSE,
    ]
    # FIXME: This function call raises CUDA_ERROR_INVALID_VALUE
    err, mod = cuda.cuModuleLoadDataEx(0, 0, option_keys, [])


def test_cuda_repr():
    actual = cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
    assert isinstance(actual, cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS)

    actual_repr = actual.__repr__()
    expected_repr = textwrap.dedent("""
    params :
    fence :
        value : 0
    nvSciSync :
        fence : 0x0
        reserved : 0
    keyedMutex :
        key : 0
    reserved : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
flags : 0
reserved : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
""")
    assert actual_repr.split() == expected_repr.split()

    actual_repr = cuda.CUDA_KERNEL_NODE_PARAMS_st().__repr__()
    expected_repr = textwrap.dedent("""
    func : <CUfunction 0x0>
gridDimX : 0
gridDimY : 0
gridDimZ : 0
blockDimX : 0
blockDimY : 0
blockDimZ : 0
sharedMemBytes : 0
kernelParams : 0
extra : 0
""")
    assert actual_repr.split() == expected_repr.split()


def test_cuda_struct_list_of_enums():
    desc = cuda.CUDA_TEXTURE_DESC_st()
    desc.addressMode = [
        cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
        cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
        cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR,
    ]

    # # Too many args
    # desc.addressMode = [cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
    #                     cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
    #                     cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR,
    #                     cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_BORDER]

    # # Too little args
    # desc.addressMode = [cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
    #                     cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP]


def test_cuda_CUstreamBatchMemOpParams():
    params = cuda.CUstreamBatchMemOpParams()
    params.operation = cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32
    params.waitValue.operation = cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32
    params.writeValue.operation = cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32
    params.flushRemoteWrites.operation = cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32
    params.waitValue.value64 = 666
    assert int(params.waitValue.value64) == 666


@pytest.mark.skipif(
    driverVersionLessThan(11030) or not supportsMemoryPool(), reason="When new attributes were introduced"
)
def test_cuda_memPool_attr():
    poolProps = cuda.CUmemPoolProps()
    poolProps.allocType = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    poolProps.location.id = 0
    poolProps.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

    attr_list = [None] * 8
    err, pool = cuda.cuMemPoolCreate(poolProps)
    assert err == cuda.CUresult.CUDA_SUCCESS

    for idx, attr in enumerate(
        [
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        ]
    ):
        err, attr_tmp = cuda.cuMemPoolGetAttribute(pool, attr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        attr_list[idx] = attr_tmp

    for idxA, attr in enumerate(
        [
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ]
    ):
        (err,) = cuda.cuMemPoolSetAttribute(pool, attr, 0)
        assert err == cuda.CUresult.CUDA_SUCCESS
    for idx, attr in enumerate([cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD]):
        (err,) = cuda.cuMemPoolSetAttribute(pool, attr, cuda.cuuint64_t(9))
        assert err == cuda.CUresult.CUDA_SUCCESS

    for idx, attr in enumerate(
        [
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        ]
    ):
        err, attr_tmp = cuda.cuMemPoolGetAttribute(pool, attr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        attr_list[idx] = attr_tmp
    assert attr_list[0] == 0
    assert attr_list[1] == 0
    assert attr_list[2] == 0
    assert int(attr_list[3]) == 9

    (err,) = cuda.cuMemPoolDestroy(pool)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(11030) or not supportsManagedMemory(), reason="When new attributes were introduced"
)
def test_cuda_pointer_attr():
    skip_if_concurrent_managed_access_disabled()
    err, ptr = cuda.cuMemAllocManaged(0x1000, cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Individual version
    attr_type_list = [
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_HOST_POINTER,
        # cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_P2P_TOKENS, # TODO: Can I somehow test this?
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_BUFFER_ID,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPED,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
    ]
    attr_value_list = [None] * len(attr_type_list)
    for idx, attr in enumerate(attr_type_list):
        err, attr_tmp = cuda.cuPointerGetAttribute(attr, ptr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        attr_value_list[idx] = attr_tmp

    # List version
    err, attr_value_list_v2 = cuda.cuPointerGetAttributes(len(attr_type_list), attr_type_list, ptr)
    assert err == cuda.CUresult.CUDA_SUCCESS
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert str(attr1) == str(attr2)

    # Test setting values
    for val in (True, False):
        (err,) = cuda.cuPointerSetAttribute(val, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, attr_tmp = cuda.cuPointerGetAttribute(cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        assert attr_tmp == val

    (err,) = cuda.cuMemFree(ptr)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(11030) or not supportsManagedMemory(), reason="When new attributes were introduced"
)
def test_pointer_get_attributes_device_ordinal():
    attributes = [
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    ]

    attrs = cuda.cuPointerGetAttributes(len(attributes), attributes, 0)

    # device ordinals are always small numbers.  A large number would indicate
    # an overflow error.

    assert abs(attrs[1][0]) < 256


@pytest.mark.skipif(not supportsManagedMemory(), reason="When new attributes were introduced")
def test_cuda_mem_range_attr(device):
    skip_if_concurrent_managed_access_disabled()
    size = 0x1000
    location_device = cuda.CUmemLocation()
    location_device.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location_device.id = int(device)
    location_cpu = cuda.CUmemLocation()
    location_cpu.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
    location_cpu.id = int(cuda.CU_DEVICE_CPU)

    err, ptr = cuda.cuMemAllocManaged(size, cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY, location_device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_PREFERRED_LOCATION, location_cpu)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY, location_cpu)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, concurrentSupported = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device
    )
    assert err == cuda.CUresult.CUDA_SUCCESS
    if concurrentSupported:
        (err,) = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY, location_device)
        assert err == cuda.CUresult.CUDA_SUCCESS
        expected_values_list = ([1, -1, [0, -1, -2], -2],)
    else:
        expected_values_list = ([1, -1, [-1, -2, -2], -2], [0, -2, [-2, -2, -2], -2])

    # Individual version
    attr_type_list = [
        cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
        cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
        cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    ]
    attr_type_size_list = [4, 4, 12, 4]
    attr_value_list = [None] * len(attr_type_list)
    for idx in range(len(attr_type_list)):
        err, attr_tmp = cuda.cuMemRangeGetAttribute(attr_type_size_list[idx], attr_type_list[idx], ptr, size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        attr_value_list[idx] = attr_tmp

    matched = False
    for expected_values in expected_values_list:
        if expected_values == attr_value_list:
            matched = True
            break
    if not matched:
        raise RuntimeError(f"attr_value_list {attr_value_list} did not match any {expected_values_list}")

    # List version
    err, attr_value_list_v2 = cuda.cuMemRangeGetAttributes(
        attr_type_size_list, attr_type_list, len(attr_type_list), ptr, size
    )
    assert err == cuda.CUresult.CUDA_SUCCESS
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert str(attr1) == str(attr2)

    (err,) = cuda.cuMemFree(ptr)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(driverVersionLessThan(11040) or not supportsMemoryPool(), reason="Mempool for graphs not supported")
def test_cuda_graphMem_attr(device):
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    allocSize = 1

    params = cuda.CUDA_MEM_ALLOC_NODE_PARAMS()
    params.poolProps.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    params.poolProps.location.id = device
    params.poolProps.allocType = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    params.bytesize = allocSize

    err, allocNode = cuda.cuGraphAddMemAllocNode(graph, None, 0, params)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, freeNode = cuda.cuGraphAddMemFreeNode(graph, [allocNode], 1, params.dptr)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuGraphLaunch(graphExec, stream)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, used = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, usedHigh = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, reserved = cuda.cuDeviceGetGraphMemAttribute(
        device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT
    )
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, reservedHigh = cuda.cuDeviceGetGraphMemAttribute(
        device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH
    )
    assert err == cuda.CUresult.CUDA_SUCCESS

    assert int(used) >= allocSize
    assert int(usedHigh) == int(used)
    assert int(reserved) == int(usedHigh)
    assert int(reservedHigh) == int(reserved)

    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuStreamDestroy(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(12010)
    or not supportsCudaAPI("cuCoredumpSetAttributeGlobal")
    or not supportsCudaAPI("cuCoredumpGetAttributeGlobal"),
    reason="Coredump API not present",
)
def test_cuda_coredump_attr():
    attr_list = [None] * 6

    (err,) = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_TRIGGER_HOST, False)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_FILE, b"corefile")
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_PIPE, b"corepipe")
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_LIGHTWEIGHT, True)
    assert err == cuda.CUresult.CUDA_SUCCESS

    for idx, attr in enumerate(
        [
            cuda.CUcoredumpSettings.CU_COREDUMP_TRIGGER_HOST,
            cuda.CUcoredumpSettings.CU_COREDUMP_FILE,
            cuda.CUcoredumpSettings.CU_COREDUMP_PIPE,
            cuda.CUcoredumpSettings.CU_COREDUMP_LIGHTWEIGHT,
        ]
    ):
        err, attr_tmp = cuda.cuCoredumpGetAttributeGlobal(attr)
        assert err == cuda.CUresult.CUDA_SUCCESS
        attr_list[idx] = attr_tmp

    assert attr_list[0] is False
    assert attr_list[1] == b"corefile"
    assert attr_list[2] == b"corepipe"
    assert attr_list[3] is True


def test_get_error_name_and_string():
    err, device = cuda.cuDeviceGet(0)
    _, s = cuda.cuGetErrorString(err)
    assert s == b"no error"
    _, s = cuda.cuGetErrorName(err)
    assert s == b"CUDA_SUCCESS"

    err, device = cuda.cuDeviceGet(-1)
    _, s = cuda.cuGetErrorString(err)
    assert s == b"invalid device ordinal"
    _, s = cuda.cuGetErrorName(err)
    assert s == b"CUDA_ERROR_INVALID_DEVICE"


@pytest.mark.skipif(not callableBinary("nvidia-smi"), reason="Binary existence needed")
def test_device_get_name(device):
    # TODO: Refactor this test once we have nvml bindings to avoid the use of subprocess
    import subprocess

    p = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
        shell=False,
        stderr=subprocess.PIPE,
    )

    delimiter = b"\r\n" if platform.system() == "Windows" else b"\n"
    expect = p.split(delimiter)
    size = 64
    _, got = cuda.cuDeviceGetName(size, device)
    got = got.split(b"\x00")[0]
    if any(b"Unable to determine the device handle for" in result for result in expect):
        # Undeterministic devices get waived
        pass
    else:
        assert any(got in result for result in expect)


# TODO: cuStreamGetCaptureInfo_v2
@pytest.mark.skipif(driverVersionLessThan(11030), reason="Driver too old for cuStreamGetCaptureInfo_v2")
def test_stream_capture():
    pass


def test_profiler():
    (err,) = cuda.cuProfilerStart()
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuProfilerStop()
    assert err == cuda.CUresult.CUDA_SUCCESS


def test_eglFrame():
    val = cuda.CUeglFrame()
    # [<CUarray 0x0>, <CUarray 0x0>, <CUarray 0x0>]
    assert int(val.frame.pArray[0]) == 0
    assert int(val.frame.pArray[1]) == 0
    assert int(val.frame.pArray[2]) == 0
    val.frame.pArray = [1, 2, 3]
    # [<CUarray 0x1>, <CUarray 0x2>, <CUarray 0x3>]
    assert int(val.frame.pArray[0]) == 1
    assert int(val.frame.pArray[1]) == 2
    assert int(val.frame.pArray[2]) == 3
    val.frame.pArray = [cuda.CUarray(4), 2, 3]
    # [<CUarray 0x4>, <CUarray 0x2>, <CUarray 0x3>]
    assert int(val.frame.pArray[0]) == 4
    assert int(val.frame.pArray[1]) == 2
    assert int(val.frame.pArray[2]) == 3
    val.frame.pPitch = [4, 2, 3]
    # [4, 2, 3]
    assert int(val.frame.pPitch[0]) == 4
    assert int(val.frame.pPitch[1]) == 2
    assert int(val.frame.pPitch[2]) == 3
    val.frame.pPitch = [1, 2, 3]
    assert int(val.frame.pPitch[0]) == 1
    assert int(val.frame.pPitch[1]) == 2
    assert int(val.frame.pPitch[2]) == 3


def test_char_range():
    val = cuda.CUipcMemHandle_st()
    for x in range(-128, 0):
        val.reserved = [x] * 64
        assert val.reserved[0] == 256 + x
    for x in range(0, 256):
        val.reserved = [x] * 64
        assert val.reserved[0] == x


def test_anon_assign():
    val1 = cuda.CUexecAffinityParam_st()
    val2 = cuda.CUexecAffinityParam_st()

    assert val1.param.smCount.val == 0
    val1.param.smCount.val = 5
    assert val1.param.smCount.val == 5
    val2.param.smCount.val = 11
    assert val2.param.smCount.val == 11

    val1.param = val2.param
    assert val1.param.smCount.val == 11


def test_union_assign():
    val = cuda.CUlaunchAttributeValue()
    val.clusterDim.x, val.clusterDim.y, val.clusterDim.z = 9, 9, 9
    attr = cuda.CUlaunchAttribute()
    attr.value = val

    assert val.clusterDim.x == 9
    assert val.clusterDim.y == 9
    assert val.clusterDim.z == 9


def test_invalid_repr_attribute():
    val = cuda.CUlaunchAttributeValue()
    string = str(val)


@pytest.mark.skipif(
    driverVersionLessThan(12020)
    or not supportsCudaAPI("cuGraphAddNode")
    or not supportsCudaAPI("cuGraphNodeSetParams")
    or not supportsCudaAPI("cuGraphExecNodeSetParams"),
    reason="Polymorphic graph APIs required",
)
def test_graph_poly():
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # cuGraphAddNode

    # Create 2 buffers
    size = int(1024 * np.uint8().itemsize)
    buffers = []
    for _ in range(2):
        err, dptr = cuda.cuMemAlloc(size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buffers += [(np.full(size, 2).astype(np.uint8), dptr)]

    # Update dev buffers
    for host, device in buffers:
        (err,) = cuda.cuMemcpyHtoD(device, host, size)
        assert err == cuda.CUresult.CUDA_SUCCESS

    # Create graph
    nodes = []
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Memset
    host, device = buffers[0]
    memsetParams = cuda.CUgraphNodeParams()
    memsetParams.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET
    memsetParams.memset.elementSize = np.uint8().itemsize
    memsetParams.memset.width = size
    memsetParams.memset.height = 1
    memsetParams.memset.dst = device
    memsetParams.memset.value = 1
    err, node = cuda.cuGraphAddNode(graph, None, None, 0, memsetParams)
    assert err == cuda.CUresult.CUDA_SUCCESS
    nodes += [node]

    # Memcpy
    host, device = buffers[1]
    memcpyParams = cuda.CUgraphNodeParams()
    memcpyParams.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY
    memcpyParams.memcpy.copyParams.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    memcpyParams.memcpy.copyParams.srcDevice = device
    memcpyParams.memcpy.copyParams.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_HOST
    memcpyParams.memcpy.copyParams.dstHost = host
    memcpyParams.memcpy.copyParams.WidthInBytes = size
    memcpyParams.memcpy.copyParams.Height = 1
    memcpyParams.memcpy.copyParams.Depth = 1
    err, node = cuda.cuGraphAddNode(graph, None, None, 0, memcpyParams)
    assert err == cuda.CUresult.CUDA_SUCCESS
    nodes += [node]

    # Instantiate, execute, validate
    err, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphLaunch(graphExec, stream)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuStreamSynchronize(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Validate
    for host, device in buffers:
        (err,) = cuda.cuMemcpyDtoH(host, device, size)
        assert err == cuda.CUresult.CUDA_SUCCESS
    assert np.array_equal(buffers[0][0], np.full(size, 1).astype(np.uint8))
    assert np.array_equal(buffers[1][0], np.full(size, 2).astype(np.uint8))

    # cuGraphNodeSetParams
    host, device = buffers[1]
    err, memcpyParamsCopy = cuda.cuGraphMemcpyNodeGetParams(nodes[1])
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert int(memcpyParamsCopy.srcDevice) == int(device)
    host, device = buffers[0]
    memcpyParams.memcpy.copyParams.srcDevice = device
    (err,) = cuda.cuGraphNodeSetParams(nodes[1], memcpyParams)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, memcpyParamsCopy = cuda.cuGraphMemcpyNodeGetParams(nodes[1])
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert int(memcpyParamsCopy.srcDevice) == int(device)

    # cuGraphExecNodeSetParams
    memsetParams.memset.value = 11
    (err,) = cuda.cuGraphExecNodeSetParams(graphExec, nodes[0], memsetParams)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphLaunch(graphExec, stream)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuStreamSynchronize(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemcpyDtoH(buffers[0][0], buffers[0][1], size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert np.array_equal(buffers[0][0], np.full(size, 11).astype(np.uint8))

    # Cleanup
    (err,) = cuda.cuMemFree(buffers[0][1])
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemFree(buffers[1][1])
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphExecDestroy(graphExec)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuStreamDestroy(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(12040) or not supportsCudaAPI("cuDeviceGetDevResource"),
    reason="Polymorphic graph APIs required",
)
def test_cuDeviceGetDevResource(device):
    err, resource_in = cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)

    err, res, count, rem = cuda.cuDevSmResourceSplitByCount(0, resource_in, 0, 2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert count != 0
    assert len(res) == 0
    err, res, count_same, rem = cuda.cuDevSmResourceSplitByCount(count, resource_in, 0, 2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert count == count_same
    assert len(res) == count
    err, res, count, rem = cuda.cuDevSmResourceSplitByCount(3, resource_in, 0, 2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert len(res) == 3


@pytest.mark.skipif(
    driverVersionLessThan(12030) or not supportsCudaAPI("cuGraphConditionalHandleCreate"),
    reason="Conditional graph APIs required",
)
def test_conditional(ctx):
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, handle = cuda.cuGraphConditionalHandleCreate(graph, ctx, 0, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    params = cuda.CUgraphNodeParams()
    params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    params.conditional.handle = handle
    params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
    params.conditional.size = 1
    params.conditional.ctx = ctx

    assert len(params.conditional.phGraph_out) == 1
    assert int(params.conditional.phGraph_out[0]) == 0
    err, node = cuda.cuGraphAddNode(graph, None, None, 0, params)
    assert err == cuda.CUresult.CUDA_SUCCESS

    assert len(params.conditional.phGraph_out) == 1
    assert int(params.conditional.phGraph_out[0]) != 0


def test_CUmemDecompressParams_st():
    desc = cuda.CUmemDecompressParams_st()
    assert int(desc.dstActBytes) == 0


def test_all_CUresult_codes():
    max_code = int(max(cuda.CUresult))
    # Smoke test. CUDA_ERROR_UNKNOWN = 999, but intentionally using literal value.
    assert max_code >= 999
    num_good = 0
    for code in range(max_code + 2):  # One past max_code
        try:
            error = cuda.CUresult(code)
        except ValueError:
            pass  # cython-generated enum does not exist for this code
        else:
            err_name, name = cuda.cuGetErrorName(error)
            if err_name == cuda.CUresult.CUDA_SUCCESS:
                assert name
                err_desc, desc = cuda.cuGetErrorString(error)
                assert err_desc == cuda.CUresult.CUDA_SUCCESS
                assert desc
                num_good += 1
            else:
                # cython-generated enum exists but is not known to an older driver
                # (example: cuda-bindings built with CTK 12.8, driver from CTK 12.0)
                assert name is None
                assert err_name == cuda.CUresult.CUDA_ERROR_INVALID_VALUE
                err_desc, desc = cuda.cuGetErrorString(error)
                assert err_desc == cuda.CUresult.CUDA_ERROR_INVALID_VALUE
                assert desc is None
    # Smoke test: Do we have at least some "good" codes?
    # The number will increase over time as new enums are added and support for
    # old CTKs is dropped, but it is not critical that this number is updated.
    assert num_good >= 76  # CTK 11.0.3_450.51.06


@pytest.mark.skipif(driverVersionLessThan(12030), reason="Driver too old for cuKernelGetName")
def test_cuKernelGetName_failure():
    err, name = cuda.cuKernelGetName(0)
    assert err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE
    assert name is None


@pytest.mark.skipif(driverVersionLessThan(12030), reason="Driver too old for cuFuncGetName")
def test_cuFuncGetName_failure():
    err, name = cuda.cuFuncGetName(0)
    assert err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE
    assert name is None


@pytest.mark.skipif(
    driverVersionLessThan(12080) or not supportsCudaAPI("cuCheckpointProcessGetState"),
    reason="When API was introduced",
)
def test_cuCheckpointProcessGetState_failure():
    err, state = cuda.cuCheckpointProcessGetState(123434)
    assert err != cuda.CUresult.CUDA_SUCCESS
    assert state is None


def test_private_function_pointer_inspector():
    from cuda.bindings._bindings.cydriver import _inspect_function_pointer

    assert _inspect_function_pointer("__cuGetErrorString") != 0


@pytest.mark.parametrize(
    "target",
    (
        driver.CUcontext,
        driver.CUstream,
        driver.CUevent,
        driver.CUmodule,
        driver.CUlibrary,
        driver.CUfunction,
        driver.CUkernel,
        driver.CUgraph,
        driver.CUgraphNode,
        driver.CUgraphExec,
        driver.CUmemoryPool,
    ),
)
def test_struct_pointer_comparison(target):
    a = target(123)
    b = target(123)
    assert a == b
    assert hash(a) == hash(b)
    c = target(456)
    assert a != c
    assert hash(a) != hash(c)


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuGraphGetId"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphGetId(device, ctx):
    """Test cuGraphGetId - get graph ID."""
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graph_id = cuda.cuGraphGetId(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(graph_id, int)
    assert graph_id > 0

    # Create another graph and verify it has a different ID
    err, graph2 = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, graph_id2 = cuda.cuGraphGetId(graph2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert graph_id2 != graph_id

    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphDestroy(graph2)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuGraphExecGetId"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphExecGetId(device, ctx):
    """Test cuGraphExecGetId - get graph exec ID."""
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Add an empty node to make the graph valid
    err, node = cuda.cuGraphAddEmptyNode(graph, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, graph_exec_id = cuda.cuGraphExecGetId(graphExec)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(graph_exec_id, int)
    assert graph_exec_id > 0

    # Create another graph exec and verify it has a different ID
    err, graph2 = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, node2 = cuda.cuGraphAddEmptyNode(graph2, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, graphExec2 = cuda.cuGraphInstantiate(graph2, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, graph_exec_id2 = cuda.cuGraphExecGetId(graphExec2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert graph_exec_id2 != graph_exec_id

    (err,) = cuda.cuGraphExecDestroy(graphExec)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphExecDestroy(graphExec2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphDestroy(graph2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuStreamDestroy(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuGraphNodeGetLocalId"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetLocalId(device, ctx):
    """Test cuGraphNodeGetLocalId - get node local ID."""
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Add multiple nodes
    err, node1 = cuda.cuGraphAddEmptyNode(graph, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, node2 = cuda.cuGraphAddEmptyNode(graph, [node1], 1)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, node3 = cuda.cuGraphAddEmptyNode(graph, [node1, node2], 2)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Get local IDs for each node
    err, node_id1 = cuda.cuGraphNodeGetLocalId(node1)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(node_id1, int)
    assert node_id1 >= 0

    err, node_id2 = cuda.cuGraphNodeGetLocalId(node2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(node_id2, int)
    assert node_id2 >= 0
    assert node_id2 != node_id1

    err, node_id3 = cuda.cuGraphNodeGetLocalId(node3)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(node_id3, int)
    assert node_id3 >= 0
    assert node_id3 != node_id1
    assert node_id3 != node_id2

    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuGraphNodeGetToolsId"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetToolsId(device, ctx):
    """Test cuGraphNodeGetToolsId - get node tools ID."""
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, node = cuda.cuGraphAddEmptyNode(graph, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, tools_node_id = cuda.cuGraphNodeGetToolsId(node)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert isinstance(tools_node_id, int)
    # toolsNodeId is unsigned long long, so it can be any non-negative value
    assert tools_node_id >= 0

    # Add another node and verify it has a different tools ID
    err, node2 = cuda.cuGraphAddEmptyNode(graph, [node], 1)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, tools_node_id2 = cuda.cuGraphNodeGetToolsId(node2)
    assert err == cuda.CUresult.CUDA_SUCCESS
    assert tools_node_id2 != tools_node_id

    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuGraphNodeGetContainingGraph"),
    reason="Requires CUDA 13.1+",
)
def test_cuGraphNodeGetContainingGraph(device, ctx):
    """Test cuGraphNodeGetContainingGraph - get graph containing a node."""
    err, graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, node = cuda.cuGraphAddEmptyNode(graph, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Get the containing graph
    err, containing_graph = cuda.cuGraphNodeGetContainingGraph(node)
    assert err == cuda.CUresult.CUDA_SUCCESS
    # Verify it's the same graph
    assert int(containing_graph) == int(graph)

    # Test with a child graph node (if supported)
    # Create a child graph node
    err, child_graph = cuda.cuGraphCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, child_node = cuda.cuGraphAddEmptyNode(child_graph, None, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Add child graph node to parent graph
    childGraphNodeParams = cuda.CUgraphNodeParams()
    childGraphNodeParams.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH
    childGraphNodeParams.graph.graph = child_graph
    err, child_graph_node = cuda.cuGraphAddNode(graph, None, None, 0, childGraphNodeParams)
    if err == cuda.CUresult.CUDA_SUCCESS:
        # Get containing graph for the child graph node
        err, containing_graph_for_child = cuda.cuGraphNodeGetContainingGraph(child_graph_node)
        assert err == cuda.CUresult.CUDA_SUCCESS
        assert int(containing_graph_for_child) == int(graph)

        # Get containing graph for node inside child graph
        err, containing_graph_for_nested = cuda.cuGraphNodeGetContainingGraph(child_node)
        assert err == cuda.CUresult.CUDA_SUCCESS
        assert int(containing_graph_for_nested) == int(child_graph)

    (err,) = cuda.cuGraphDestroy(graph)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuGraphDestroy(child_graph)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuStreamGetDevResource"),
    reason="Requires CUDA 13.1+",
)
def test_cuStreamGetDevResource(device, ctx):
    """Test cuStreamGetDevResource - get device resource from stream."""
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Get SM resource from stream
    err, resource = cuda.cuStreamGetDevResource(stream, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    assert err == cuda.CUresult.CUDA_SUCCESS
    # Verify resource is valid (non-None)
    assert resource is not None

    (err,) = cuda.cuStreamDestroy(stream)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(
    driverVersionLessThan(13010) or not supportsCudaAPI("cuDevSmResourceSplit"),
    reason="Requires CUDA 13.1+",
)
def test_cuDevSmResourceSplit(device, ctx):
    """Test cuDevSmResourceSplit - split SM resource into structured groups."""
    err, resource_in = cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Create group params for splitting into 1 group (simpler test)
    nb_groups = 1
    group_params = cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
    # Set up group: request 4 SMs with coscheduled count of 2
    group_params.smCount = 4
    group_params.coscheduledSmCount = 2

    # Split the resource
    err, res, rem = cuda.cuDevSmResourceSplit(nb_groups, resource_in, 0, group_params)
    assert err == cuda.CUresult.CUDA_SUCCESS
    # Verify we got results
    assert len(res) == nb_groups
    # Verify remainder is valid (may be None if no remainder)
    assert rem is not None or len(res) > 0


def test_buffer_reference():
    # Create a host buffer
    size = int(1024 * np.uint8().itemsize)
    host = np.full(size, 2).astype(np.uint8)

    # Set the buffer to a struct member
    memcpyParams = cuda.CUgraphNodeParams()
    memcpyParams.memcpy.copyParams.dstHost = host

    # Delete the local reference to the host buffer.  The reference in the
    # struct should keep it alive.
    del host

    # Create a new numpy array from the pointer and make sure the memory is
    # intact and hasn't been freed.  If the reference counting in
    # copyParams.dstHost is incorrect, we will either see over-written memory or
    # a segmentation fault here.
    ptr = ctypes.cast(memcpyParams.memcpy.copyParams.dstHost, ctypes.POINTER(ctypes.c_uint8))
    x = np.ctypeslib.as_array(ptr, shape=(size,))
    assert np.all(x == 2)
