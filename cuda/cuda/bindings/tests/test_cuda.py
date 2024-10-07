# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import platform
import pytest
import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np
import textwrap
import shutil
from sysconfig import get_paths

def driverVersionLessThan(target):
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, version = cuda.cuDriverGetVersion()
    assert(err == cuda.CUresult.CUDA_SUCCESS)
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
    return shutil.which(name) != None

def test_cuda_memcpy():
    # Init CUDA
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Get device
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Construct context
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Allocate dev memory
    size = int(1024 * np.uint8().itemsize)
    err, dptr = cuda.cuMemAlloc(size)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Set h1 and h2 memory to be different
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # h1 to D
    err, = cuda.cuMemcpyHtoD(dptr, h1, size)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # D to h2
    err, = cuda.cuMemcpyDtoH(h2, dptr, size)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Validate h1 == h2
    assert(np.array_equal(h1, h2))

    # Cleanup
    err, = cuda.cuMemFree(dptr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_cuda_array():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # No context created
    desc = cuda.CUDA_ARRAY_DESCRIPTOR()
    err, arr = cuda.cuArrayCreate(desc)
    assert(err == cuda.CUresult.CUDA_ERROR_INVALID_CONTEXT or err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE)

    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Desciption not filled
    err, arr = cuda.cuArrayCreate(desc)
    assert(err == cuda.CUresult.CUDA_ERROR_INVALID_VALUE)

    # Pass
    desc.Format = cuda.CUarray_format.CU_AD_FORMAT_SIGNED_INT8
    desc.NumChannels = 1
    desc.Width = 1
    err, arr = cuda.cuArrayCreate(desc)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, = cuda.cuArrayDestroy(arr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_cuda_repr_primitive():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(str(device) == '<CUdevice 0>')
    assert(int(device) == 0)

    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(str(ctx).startswith('<CUcontext 0x'))
    assert(int(ctx) > 0)
    assert(hex(ctx) == hex(int(ctx)))

    # CUdeviceptr
    err, dptr = cuda.cuMemAlloc(1024 * np.uint8().itemsize)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(str(dptr).startswith('<CUdeviceptr '))
    assert(int(dptr) > 0)
    err, = cuda.cuMemFree(dptr)
    size = 7
    dptr = cuda.CUdeviceptr(size)
    assert(str(dptr) == '<CUdeviceptr {}>'.format(size))
    assert(int(dptr) == size)
    size = 4294967295
    dptr = cuda.CUdeviceptr(size)
    assert(str(dptr) == '<CUdeviceptr {}>'.format(size))
    assert(int(dptr) == size)
    size = 18446744073709551615
    dptr = cuda.CUdeviceptr(size)
    assert(str(dptr) == '<CUdeviceptr {}>'.format(size))
    assert(int(dptr) == size)

    # cuuint32_t
    size = 7
    int32 = cuda.cuuint32_t(size)
    assert(str(int32) == '<cuuint32_t {}>'.format(size))
    assert(int(int32) == size)
    size = 4294967295
    int32 = cuda.cuuint32_t(size)
    assert(str(int32) == '<cuuint32_t {}>'.format(size))
    assert(int(int32) == size)
    size = 18446744073709551615
    try:
        int32 = cuda.cuuint32_t(size)
        raise RuntimeError('int32 = cuda.cuuint32_t(18446744073709551615) did not fail')
    except OverflowError as err:
        pass

    # cuuint64_t
    size = 7
    int64 = cuda.cuuint64_t(size)
    assert(str(int64) == '<cuuint64_t {}>'.format(size))
    assert(int(int64) == size)
    size = 4294967295
    int64 = cuda.cuuint64_t(size)
    assert(str(int64) == '<cuuint64_t {}>'.format(size))
    assert(int(int64) == size)
    size = 18446744073709551615
    int64 = cuda.cuuint64_t(size)
    assert(str(int64) == '<cuuint64_t {}>'.format(size))
    assert(int(int64) == size)

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_cuda_repr_pointer():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Test 1: Classes representing pointers
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(str(ctx).startswith('<CUcontext 0x'))
    assert(int(ctx) > 0)
    assert(hex(ctx) == hex(int(ctx)))
    randomCtxPointer = 12345
    randomCtx = cuda.CUcontext(randomCtxPointer)
    assert(str(randomCtx) == '<CUcontext {}>'.format(hex(randomCtxPointer)))
    assert(int(randomCtx) == randomCtxPointer)
    assert(hex(randomCtx) == hex(randomCtxPointer))

    # Test 2: Function pointers
    func = 12345
    b2d_cb = cuda.CUoccupancyB2DSize(func)
    assert(str(b2d_cb) == '<CUoccupancyB2DSize {}>'.format(hex(func)))
    assert(int(b2d_cb) == func)
    assert(hex(b2d_cb) == hex(func))

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_cuda_uuid_list_access():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, uuid = cuda.cuDeviceGetUuid(device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(len(uuid.bytes) <= 16)

    jit_option = cuda.CUjit_option
    options = {
        jit_option.CU_JIT_INFO_LOG_BUFFER: 1,
        jit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: 2,
        jit_option.CU_JIT_ERROR_LOG_BUFFER: 3,
        jit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: 4,
        jit_option.CU_JIT_LOG_VERBOSE: 5,
    }

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_cuda_cuModuleLoadDataEx():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, dev = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, dev)
    assert(err == cuda.CUresult.CUDA_SUCCESS)


    option_keys = [
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_LOG_VERBOSE
    ]
    err, mod = cuda.cuModuleLoadDataEx(0, 0, option_keys, [])

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)


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
    desc.addressMode = [cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
                        cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
                        cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR]

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
    assert(int(params.waitValue.value64) == 666)

@pytest.mark.skipif(driverVersionLessThan(11030) or not supportsMemoryPool(), reason='When new attributes were introduced')
def test_cuda_memPool_attr():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    poolProps = cuda.CUmemPoolProps()
    poolProps.allocType = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    poolProps.location.id = 0
    poolProps.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

    attr_list = [None] * 8
    err, pool = cuda.cuMemPoolCreate(poolProps)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    for idx, attr in enumerate([cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH]):
        err, attr_tmp = cuda.cuMemPoolGetAttribute(pool, attr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        attr_list[idx] = attr_tmp

    for idxA, attr in enumerate([cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                                 cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                                 cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES]):
        err, = cuda.cuMemPoolSetAttribute(pool, attr, 0)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
    for idx, attr in enumerate([cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD]):
        err, = cuda.cuMemPoolSetAttribute(pool, attr, cuda.cuuint64_t(9))
        assert(err == cuda.CUresult.CUDA_SUCCESS)

    for idx, attr in enumerate([cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD]):
        err, attr_tmp = cuda.cuMemPoolGetAttribute(pool, attr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        attr_list[idx] = attr_tmp
    assert(attr_list[0] == 0)
    assert(attr_list[1] == 0)
    assert(attr_list[2] == 0)
    assert(int(attr_list[3]) == 9)

    err, = cuda.cuMemPoolDestroy(pool)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(driverVersionLessThan(11030) or not supportsManagedMemory(), reason='When new attributes were introduced')
def test_cuda_pointer_attr():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ptr = cuda.cuMemAllocManaged(0x1000, cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Individual version
    attr_type_list = [cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT,
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
                      cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE]
    attr_value_list = [None] * len(attr_type_list)
    for idx, attr in enumerate(attr_type_list):
        err, attr_tmp = cuda.cuPointerGetAttribute(attr, ptr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        attr_value_list[idx] = attr_tmp

    # List version
    err, attr_value_list_v2 = cuda.cuPointerGetAttributes(len(attr_type_list), attr_type_list, ptr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert(str(attr1) == str(attr2))

    # Test setting values
    for val in (True, False):
        err, = cuda.cuPointerSetAttribute(val, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        err, attr_tmp = cuda.cuPointerGetAttribute(cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        assert(attr_tmp == val)

    err, = cuda.cuMemFree(ptr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(not supportsManagedMemory(), reason='When new attributes were introduced')
def test_cuda_mem_range_attr():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    size = int(0x1000)
    err, ptr = cuda.cuMemAllocManaged(size, cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_PREFERRED_LOCATION, cuda.CU_DEVICE_CPU)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY, cuda.CU_DEVICE_CPU)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, concurrentSupported = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    if concurrentSupported:
        err, = cuda.cuMemAdvise(ptr, size, cuda.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY, device)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        expected_values_list = ([1, -1, [0, -1, -2], -2],)
    else:
        expected_values_list = ([1, -1, [-1, -2, -2], -2], [0, -2, [-2, -2, -2], -2])

    # Individual version
    attr_type_list = [cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                      cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
                      cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION]
    attr_type_size_list = [4, 4, 12, 4]
    attr_value_list = [None] * len(attr_type_list)
    for idx in range(len(attr_type_list)):
        err, attr_tmp = cuda.cuMemRangeGetAttribute(attr_type_size_list[idx], attr_type_list[idx], ptr, size)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        attr_value_list[idx] = attr_tmp

    matched = False
    for expected_values in expected_values_list:
        if expected_values == attr_value_list:
            matched = True
            break
    if not matched:
        raise RuntimeError(f'attr_value_list {attr_value_list} did not match any {expected_values_list}')

    # List version
    err, attr_value_list_v2 = cuda.cuMemRangeGetAttributes(attr_type_size_list, attr_type_list, len(attr_type_list), ptr, size)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    for attr1, attr2 in zip(attr_value_list, attr_value_list_v2):
        assert(str(attr1) == str(attr2))

    err, = cuda.cuMemFree(ptr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(driverVersionLessThan(11040) or not supportsMemoryPool(), reason='Mempool for graphs not supported')
def test_cuda_graphMem_attr():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, stream = cuda.cuStreamCreate(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, graph = cuda.cuGraphCreate(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    allocSize = 1

    params = cuda.CUDA_MEM_ALLOC_NODE_PARAMS()
    params.poolProps.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    params.poolProps.location.id = device
    params.poolProps.allocType = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    params.bytesize = allocSize

    err, allocNode = cuda.cuGraphAddMemAllocNode(graph, None, 0, params)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, freeNode = cuda.cuGraphAddMemFreeNode(graph, [allocNode], 1, params.dptr)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, = cuda.cuGraphLaunch(graphExec, stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, used = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, usedHigh = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, reserved = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, reservedHigh = cuda.cuDeviceGetGraphMemAttribute(device, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    assert int(used) >= allocSize
    assert int(usedHigh) == int(used)
    assert int(reserved) == int(usedHigh)
    assert int(reservedHigh) == int(reserved)

    err, = cuda.cuGraphDestroy(graph)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuStreamDestroy(stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(driverVersionLessThan(12010)
                    or not supportsCudaAPI('cuCoredumpSetAttributeGlobal')
                    or not supportsCudaAPI('cuCoredumpGetAttributeGlobal'), reason='Coredump API not present')
def test_cuda_coredump_attr():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    attr_list = [None] * 6

    err, = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_TRIGGER_HOST, False)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_FILE, b'corefile')
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_PIPE, b'corepipe')
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCoredumpSetAttributeGlobal(cuda.CUcoredumpSettings.CU_COREDUMP_LIGHTWEIGHT, True)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    for idx, attr in enumerate([cuda.CUcoredumpSettings.CU_COREDUMP_TRIGGER_HOST,
                                cuda.CUcoredumpSettings.CU_COREDUMP_FILE,
                                cuda.CUcoredumpSettings.CU_COREDUMP_PIPE,
                                cuda.CUcoredumpSettings.CU_COREDUMP_LIGHTWEIGHT,
                                ]):
        err, attr_tmp = cuda.cuCoredumpGetAttributeGlobal(attr)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        attr_list[idx] = attr_tmp

    assert(attr_list[0] == False)
    assert(attr_list[1] == b'corefile')
    assert(attr_list[2] == b'corepipe')
    assert(attr_list[3] == True)

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_get_error_name_and_string():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

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
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(not callableBinary('nvidia-smi'), reason='Binary existance needed')
def test_device_get_name():
    import subprocess

    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    delimiter = b'\r\n' if platform.system() == "Windows" else b'\n'
    expect = p.stdout.split(delimiter)
    size = 64
    _, got = cuda.cuDeviceGetName(size, device)
    got = got.split(b'\x00')[0]
    if any(b'Unable to determine the device handle for' in result for result in expect):
        # Undeterministic devices get waived
        pass
    else:
        assert any(got in result for result in expect)


    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

# TODO: cuStreamGetCaptureInfo_v2
@pytest.mark.skipif(driverVersionLessThan(11030), reason='Driver too old for cuStreamGetCaptureInfo_v2')
def test_stream_capture():
    pass

def test_profiler():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuProfilerStart()
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuProfilerStop()
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

def test_eglFrame():
    val = cuda.CUeglFrame()
    # [<CUarray 0x0>, <CUarray 0x0>, <CUarray 0x0>]
    assert(int(val.frame.pArray[0]) == 0)
    assert(int(val.frame.pArray[1]) == 0)
    assert(int(val.frame.pArray[2]) == 0)
    val.frame.pArray = [1,2,3]
    # [<CUarray 0x1>, <CUarray 0x2>, <CUarray 0x3>]
    assert(int(val.frame.pArray[0]) == 1)
    assert(int(val.frame.pArray[1]) == 2)
    assert(int(val.frame.pArray[2]) == 3)
    val.frame.pArray = [cuda.CUarray(4),2,3]
    # [<CUarray 0x4>, <CUarray 0x2>, <CUarray 0x3>]
    assert(int(val.frame.pArray[0]) == 4)
    assert(int(val.frame.pArray[1]) == 2)
    assert(int(val.frame.pArray[2]) == 3)
    val.frame.pPitch = [4, 2, 3]
    # [4, 2, 3]
    assert(int(val.frame.pPitch[0]) == 4)
    assert(int(val.frame.pPitch[1]) == 2)
    assert(int(val.frame.pPitch[2]) == 3)
    val.frame.pPitch = [1,2,3]
    assert(int(val.frame.pPitch[0]) == 1)
    assert(int(val.frame.pPitch[1]) == 2)
    assert(int(val.frame.pPitch[2]) == 3)

def test_char_range():
    val = cuda.CUipcMemHandle_st()
    for x in range(-128, 0):
        val.reserved = [x] * 64
        assert(val.reserved[0] == 256 + x)
    for x in range(0, 256):
        val.reserved = [x] * 64
        assert(val.reserved[0] == x)

def test_anon_assign():
    val1 = cuda.CUexecAffinityParam_st()
    val2 = cuda.CUexecAffinityParam_st()

    assert(val1.param.smCount.val == 0)
    val1.param.smCount.val = 5
    assert(val1.param.smCount.val == 5)
    val2.param.smCount.val = 11
    assert(val2.param.smCount.val == 11)

    val1.param = val2.param
    assert(val1.param.smCount.val == 11)

def test_union_assign():
    val = cuda.CUlaunchAttributeValue()
    val.clusterDim.x, val.clusterDim.y, val.clusterDim.z = 9,9,9
    attr = cuda.CUlaunchAttribute()
    attr.value = val

    assert(val.clusterDim.x == 9)
    assert(val.clusterDim.y == 9)
    assert(val.clusterDim.z == 9)

def test_invalid_repr_attribute():
    val = cuda.CUlaunchAttributeValue()
    string = str(val)

@pytest.mark.skipif(driverVersionLessThan(12020)
                    or not supportsCudaAPI('cuGraphAddNode')
                    or not supportsCudaAPI('cuGraphNodeSetParams')
                    or not supportsCudaAPI('cuGraphExecNodeSetParams'), reason='Polymorphic graph APIs required')
def test_graph_poly():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, stream = cuda.cuStreamCreate(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # cuGraphAddNode

    # Create 2 buffers
    size = int(1024 * np.uint8().itemsize)
    buffers = []
    for _ in range(2):
        err, dptr = cuda.cuMemAlloc(size)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        buffers += [(np.full(size, 2).astype(np.uint8), dptr)]

    # Update dev buffers
    for host, device in buffers:
        err, = cuda.cuMemcpyHtoD(device, host, size)
        assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Create graph
    nodes = []
    err, graph = cuda.cuGraphCreate(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Memset
    host, device = buffers[0]
    memsetParams = cuda.CUgraphNodeParams()
    memsetParams.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET
    memsetParams.memset.elementSize = np.uint8().itemsize
    memsetParams.memset.width = size
    memsetParams.memset.height = 1
    memsetParams.memset.dst = device
    memsetParams.memset.value = 1
    err, node = cuda.cuGraphAddNode(graph, None, 0, memsetParams)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
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
    err, node = cuda.cuGraphAddNode(graph, None, 0, memcpyParams)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    nodes += [node]

    # Instantiate, execute, validate
    err, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuGraphLaunch(graphExec, stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuStreamSynchronize(stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Validate
    for host, device in buffers:
        err, = cuda.cuMemcpyDtoH(host, device, size)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(np.array_equal(buffers[0][0], np.full(size, 1).astype(np.uint8)))
    assert(np.array_equal(buffers[1][0], np.full(size, 2).astype(np.uint8)))

    # cuGraphNodeSetParams
    host, device = buffers[1]
    err, memcpyParamsCopy = cuda.cuGraphMemcpyNodeGetParams(nodes[1])
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(int(memcpyParamsCopy.srcDevice) == int(device))
    host, device = buffers[0]
    memcpyParams.memcpy.copyParams.srcDevice = device
    err, = cuda.cuGraphNodeSetParams(nodes[1], memcpyParams)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, memcpyParamsCopy = cuda.cuGraphMemcpyNodeGetParams(nodes[1])
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(int(memcpyParamsCopy.srcDevice) == int(device))

    # cuGraphExecNodeSetParams
    memsetParams.memset.value = 11
    err, = cuda.cuGraphExecNodeSetParams(graphExec, nodes[0], memsetParams)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuGraphLaunch(graphExec, stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuStreamSynchronize(stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemcpyDtoH(buffers[0][0], buffers[0][1], size)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(np.array_equal(buffers[0][0], np.full(size, 11).astype(np.uint8)))

    # Cleanup
    err, = cuda.cuMemFree(buffers[0][1])
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuMemFree(buffers[1][1])
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuGraphExecDestroy(graphExec)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuGraphDestroy(graph)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuStreamDestroy(stream)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(driverVersionLessThan(12040)
                    or not supportsCudaAPI('cuDeviceGetDevResource'), reason='Polymorphic graph APIs required')
def test_cuDeviceGetDevResource():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, resource_in = cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, res, count, rem = cuda.cuDevSmResourceSplitByCount(0, resource_in, 0, 2)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(count != 0)
    assert(len(res) == 0)
    err, res, count_same, rem = cuda.cuDevSmResourceSplitByCount(count, resource_in, 0, 2)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(count == count_same)
    assert(len(res) == count)
    err, res, count, rem = cuda.cuDevSmResourceSplitByCount(3, resource_in, 0, 2)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    assert(len(res) == 3)

    err, = cuda.cuCtxDestroy(ctx)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(driverVersionLessThan(12030)
                    or not supportsCudaAPI('cuGraphConditionalHandleCreate'), reason='Conditional graph APIs required')
def test_conditional():
    err, = cuda.cuInit(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, device = cuda.cuDeviceGet(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, ctx = cuda.cuCtxCreate(0, device)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    err, graph = cuda.cuGraphCreate(0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)
    err, handle = cuda.cuGraphConditionalHandleCreate(graph, ctx, 0, 0)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    params = cuda.CUgraphNodeParams()
    params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    params.conditional.handle = handle
    params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
    params.conditional.size = 1
    params.conditional.ctx = ctx

    assert(len(params.conditional.phGraph_out) == 1)
    assert(int(params.conditional.phGraph_out[0]) == 0)
    err, node = cuda.cuGraphAddNode(graph, None, 0, params)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    assert(len(params.conditional.phGraph_out) == 1)
    assert(int(params.conditional.phGraph_out[0]) != 0)
