# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from _pytest.mark.structures import store_mark
import ctypes
import cuda.cuda as cuda
import cuda.cudart as cudart
import math
import numpy as np
import pytest

def isSuccess(err):
    return err == cudart.cudaError_t.cudaSuccess

def assertSuccess(err):
    assert(isSuccess(err))

def driverVersionLessThan(target):
    err, version = cudart.cudaDriverGetVersion()
    assertSuccess(err)
    return version < target

def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return isSuccess(err) and isSupported

def supportsSparseTexturesDeviceFilter():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrSparseCudaArraySupported, 0)
    return isSuccess(err) and isSupported

def supportsCudaAPI(name):
    return name in dir(cuda) or dir(cudart)

def test_cudart_memcpy():
    # Allocate dev memory
    size = 1024 * np.uint8().itemsize
    err, dptr = cudart.cudaMalloc(size)
    assertSuccess(err)

    # Set h1 and h2 memory to be different
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # h1 to D
    err, = cudart.cudaMemcpy(dptr, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assertSuccess(err)

    # D to h2
    err, = cudart.cudaMemcpy(h2, dptr, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assertSuccess(err)

    # Validate h1 == h2
    assert(np.array_equal(h1, h2))

    # Cleanup
    err, = cudart.cudaFree(dptr)
    assertSuccess(err)

def test_cudart_hostRegister():
    # Use hostRegister API to check for correct enum return values
    page_size = 80
    addr_host = np.full(page_size * 3, 1).astype(np.uint8)
    addr = addr_host.ctypes.data

    size_0 = ((16 * page_size) / 8)
    addr_0 = addr + int(((0 * page_size) / 8))
    size_1 = ((16 * page_size) / 8)
    addr_1 = addr + int(((8 * page_size) / 8))

    err, = cudart.cudaHostRegister(addr_0, size_0, 3)
    assertSuccess(err)
    err, = cudart.cudaHostRegister(addr_1, size_1, 3)
    assert(err == cudart.cudaError_t.cudaErrorHostMemoryAlreadyRegistered)

    err, = cudart.cudaHostUnregister(addr_1)
    assert(err == cudart.cudaError_t.cudaErrorInvalidValue)
    err, = cudart.cudaHostUnregister(addr_0)
    assertSuccess(err)

def test_cudart_class_reference():
    offset = 1
    width = 4
    height = 5
    depth = 6
    flags = 0
    numMipLevels = 1

    extent = cudart.cudaExtent()
    formatDesc = cudart.cudaChannelFormatDesc()
    externalMemoryMipmappedArrayDesc = cudart.cudaExternalMemoryMipmappedArrayDesc()

    # Get/set class attributes
    extent.width  = width
    extent.height = height
    extent.depth  = depth

    formatDesc.x = 8
    formatDesc.y = 0
    formatDesc.z = 0
    formatDesc.w = 0
    formatDesc.f = cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned

    externalMemoryMipmappedArrayDesc.offset     = offset
    externalMemoryMipmappedArrayDesc.formatDesc = formatDesc
    externalMemoryMipmappedArrayDesc.extent     = extent
    externalMemoryMipmappedArrayDesc.flags      = flags
    externalMemoryMipmappedArrayDesc.numLevels  = numMipLevels

    # Can manipulate child structure values directly
    externalMemoryMipmappedArrayDesc.extent.width  = width+1
    externalMemoryMipmappedArrayDesc.extent.height = height+1
    externalMemoryMipmappedArrayDesc.extent.depth  = depth+1
    assert(externalMemoryMipmappedArrayDesc.extent.width == width+1)
    assert(externalMemoryMipmappedArrayDesc.extent.height == height+1)
    assert(externalMemoryMipmappedArrayDesc.extent.depth == depth+1)

    externalMemoryMipmappedArrayDesc.formatDesc.x = 20
    externalMemoryMipmappedArrayDesc.formatDesc.y = 21
    externalMemoryMipmappedArrayDesc.formatDesc.z = 22
    externalMemoryMipmappedArrayDesc.formatDesc.w = 23
    externalMemoryMipmappedArrayDesc.formatDesc.f = cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat
    assert(externalMemoryMipmappedArrayDesc.formatDesc.x == 20)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.y == 21)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.z == 22)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.w == 23)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.f == cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat)

    # Can copy classes over
    externalMemoryMipmappedArrayDesc.extent = extent
    assert(externalMemoryMipmappedArrayDesc.extent.width == width)
    assert(externalMemoryMipmappedArrayDesc.extent.height == height)
    assert(externalMemoryMipmappedArrayDesc.extent.depth == depth)

    externalMemoryMipmappedArrayDesc.formatDesc = formatDesc
    assert(externalMemoryMipmappedArrayDesc.formatDesc.x == 8)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.y == 0)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.z == 0)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.w == 0)
    assert(externalMemoryMipmappedArrayDesc.formatDesc.f == cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned)

@pytest.mark.skipif(not supportsSparseTexturesDeviceFilter(), reason='Sparse Texture Device Filter')
def test_cudart_class_inline():
    extent = cudart.cudaExtent()
    extent.width  = 1000
    extent.height = 500
    extent.depth  = 0

    desc = cudart.cudaChannelFormatDesc()
    desc.x = 32
    desc.y = 32
    desc.z = 32
    desc.w = 32
    desc.f = cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat

    numChannels = 4
    numBytesPerChannel = desc.x/8
    numBytesPerTexel = numChannels * numBytesPerChannel

    flags = cudart.cudaArraySparse
    maxDim = max(extent.width, extent.height)
    numLevels = int(float(1.0) + math.log(maxDim, 2))

    err, mipmap = cudart.cudaMallocMipmappedArray(desc, extent, numLevels, flags)
    assertSuccess(err)

    err, sparseProp = cudart.cudaMipmappedArrayGetSparseProperties(mipmap)
    assertSuccess(err)

    # tileExtent
    # TODO: Will these values always be this same? Maybe need a more stable test?
    # TODO: Are these values even correct? Need to research the function some more.. Maybe need an easier API test
    assert(sparseProp.tileExtent.width == 64)
    assert(sparseProp.tileExtent.height == 64)
    assert(sparseProp.tileExtent.depth == 1)

    sparsePropNew = cudart.cudaArraySparseProperties()
    sparsePropNew.tileExtent.width = 15
    sparsePropNew.tileExtent.height = 16
    sparsePropNew.tileExtent.depth = 17

    # Check that we can copy inner structs
    sparseProp.tileExtent = sparsePropNew.tileExtent
    assert(sparseProp.tileExtent.width == 15)
    assert(sparseProp.tileExtent.height == 16)
    assert(sparseProp.tileExtent.depth == 17)

    assert(sparseProp.miptailFirstLevel == 3)
    assert(sparseProp.miptailSize == 196608)
    assert(sparseProp.flags == 0)

    err, = cudart.cudaFreeMipmappedArray(mipmap)
    assertSuccess(err)

    # TODO
    example = cudart.cudaExternalSemaphoreSignalNodeParams()
    example.extSemArray = [cudart.cudaExternalSemaphore_t(0), cudart.cudaExternalSemaphore_t(123), cudart.cudaExternalSemaphore_t(999)]
    a1 = cudart.cudaExternalSemaphoreSignalParams()
    a1.params.fence.value = 7
    a1.params.nvSciSync.fence = 999
    a1.params.keyedMutex.key = 9
    a1.flags = 1
    a2 = cudart.cudaExternalSemaphoreSignalParams()
    a2.params.fence.value = 7
    a2.params.nvSciSync.fence = 999
    a2.params.keyedMutex.key = 9
    a2.flags = 2
    a3 = cudart.cudaExternalSemaphoreSignalParams()
    a3.params.fence.value = 7
    a3.params.nvSciSync.fence = 999
    a3.params.keyedMutex.key = 9
    a3.flags = 3
    example.paramsArray = [a1]
    # Note: Setting is a pass by value. Changing the object does not reflect internal value
    a3.params.fence.value = 4
    a3.params.nvSciSync.fence = 4
    a3.params.keyedMutex.key = 4
    a3.flags = 4
    example.numExtSems = 3

def test_cudart_graphs():
    err, graph = cudart.cudaGraphCreate(0)
    assertSuccess(err)

    err, pGraphNode0 = cudart.cudaGraphAddEmptyNode(graph, None, 0)
    assertSuccess(err)
    err, pGraphNode1 = cudart.cudaGraphAddEmptyNode(graph, [pGraphNode0], 1)
    assertSuccess(err)
    err, pGraphNode2 = cudart.cudaGraphAddEmptyNode(graph, [pGraphNode0, pGraphNode1], 2)
    assertSuccess(err)

    err, nodes, numNodes = cudart.cudaGraphGetNodes(graph)
    err, nodes, numNodes = cudart.cudaGraphGetNodes(graph, numNodes)

    stream_legacy = cudart.cudaStream_t(cudart.cudaStreamLegacy)
    stream_per_thread = cudart.cudaStream_t(cudart.cudaStreamPerThread)
    err, stream_with_flags = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
    assertSuccess(err)

def test_cudart_list_access():
    err, prop = cudart.cudaGetDeviceProperties(0)
    prop.name = prop.name + b' '*(256-len(prop.name))

def test_cudart_class_setters():
    dim = cudart.dim3()

    dim.x = 1
    dim.y = 2
    dim.z = 3

    assert dim.x == 1
    assert dim.y == 2
    assert dim.z == 3

def test_cudart_both_type():
    err, mode = cudart.cudaThreadExchangeStreamCaptureMode(cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    assertSuccess(err)
    err, mode = cudart.cudaThreadExchangeStreamCaptureMode(cudart.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed)
    assertSuccess(err)
    assert(mode == cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    err, mode = cudart.cudaThreadExchangeStreamCaptureMode(cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)
    assertSuccess(err)
    assert(mode == cudart.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed)
    err, mode = cudart.cudaThreadExchangeStreamCaptureMode(cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    assertSuccess(err)
    assert(mode == cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)

def test_cudart_cudaGetDeviceProperties():
    err, prop = cudart.cudaGetDeviceProperties(0)
    assertSuccess(err)
    attrs = ['accessPolicyMaxWindowSize', 'asyncEngineCount', 'canMapHostMemory', 'canUseHostPointerForRegisteredMem', 'clockRate', 'computeMode', 'computePreemptionSupported', 'concurrentKernels', 'concurrentManagedAccess', 'cooperativeLaunch', 'cooperativeMultiDeviceLaunch', 'deviceOverlap', 'directManagedMemAccessFromHost', 'getPtr', 'globalL1CacheSupported', 'hostNativeAtomicSupported', 'integrated', 'isMultiGpuBoard', 'kernelExecTimeoutEnabled', 'l2CacheSize', 'localL1CacheSupported', 'luid', 'luidDeviceNodeMask', 'major', 'managedMemory', 'maxBlocksPerMultiProcessor', 'maxGridSize', 'maxSurface1D', 'maxSurface1DLayered', 'maxSurface2D', 'maxSurface2DLayered', 'maxSurface3D', 'maxSurfaceCubemap', 'maxSurfaceCubemapLayered', 'maxTexture1D', 'maxTexture1DLayered', 'maxTexture1DLinear', 'maxTexture1DMipmap', 'maxTexture2D', 'maxTexture2DGather', 'maxTexture2DLayered', 'maxTexture2DLinear', 'maxTexture2DMipmap', 'maxTexture3D', 'maxTexture3DAlt', 'maxTextureCubemap', 'maxTextureCubemapLayered', 'maxThreadsDim', 'maxThreadsPerBlock', 'maxThreadsPerMultiProcessor', 'memPitch', 'memoryBusWidth', 'memoryClockRate', 'minor', 'multiGpuBoardGroupID', 'multiProcessorCount', 'name', 'pageableMemoryAccess', 'pageableMemoryAccessUsesHostPageTables', 'pciBusID', 'pciDeviceID', 'pciDomainID', 'persistingL2CacheMaxSize', 'regsPerBlock', 'regsPerMultiprocessor', 'reservedSharedMemPerBlock', 'sharedMemPerBlock', 'sharedMemPerBlockOptin', 'sharedMemPerMultiprocessor', 'singleToDoublePrecisionPerfRatio', 'streamPrioritiesSupported', 'surfaceAlignment', 'tccDriver', 'textureAlignment', 'texturePitchAlignment', 'totalConstMem', 'totalGlobalMem', 'unifiedAddressing', 'uuid', 'warpSize']
    for attr in attrs:
        assert hasattr(prop, attr)
    assert len(prop.name.decode("utf-8")) != 0
    assert len(prop.uuid.bytes.hex()) != 0

    example = cudart.cudaExternalSemaphoreSignalNodeParams()
    example.extSemArray = [cudart.cudaExternalSemaphore_t(0), cudart.cudaExternalSemaphore_t(123), cudart.cudaExternalSemaphore_t(999)]
    a1 = cudart.cudaExternalSemaphoreSignalParams()
    a1.params.fence.value = 7
    a1.params.nvSciSync.fence = 999
    a1.params.keyedMutex.key = 9
    a1.flags = 1
    a2 = cudart.cudaExternalSemaphoreSignalParams()
    a2.params.fence.value = 7
    a2.params.nvSciSync.fence = 999
    a2.params.keyedMutex.key = 9
    a2.flags = 2
    a3 = cudart.cudaExternalSemaphoreSignalParams()
    a3.params.fence.value = 7
    a3.params.nvSciSync.fence = 999
    a3.params.keyedMutex.key = 9
    a3.flags = 3
    example.paramsArray = [a1]
    # Note: Setting is a pass by value. Changing the object does not reflect internal value
    a3.params.fence.value = 4
    a3.params.nvSciSync.fence = 4
    a3.params.keyedMutex.key = 4
    a3.flags = 4
    example.numExtSems = 3

@pytest.mark.skipif(driverVersionLessThan(11030) or not supportsMemoryPool(), reason='When new attributes were introduced')
def test_cudart_MemPool_attr():
    poolProps = cudart.cudaMemPoolProps()
    poolProps.allocType = cudart.cudaMemAllocationType.cudaMemAllocationTypePinned
    poolProps.location.id = 0
    poolProps.location.type = cudart.cudaMemLocationType.cudaMemLocationTypeDevice

    attr_list = [None] * 8
    err, pool = cudart.cudaMemPoolCreate(poolProps)
    assertSuccess(err)

    for idx, attr in enumerate([cudart.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies,
                                cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic,
                                cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemCurrent,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemHigh,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemCurrent,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemHigh]):
        err, attr_tmp = cudart.cudaMemPoolGetAttribute(pool, attr)
        assertSuccess(err)
        attr_list[idx] = attr_tmp

    for idxA, attr in enumerate([cudart.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies,
                                 cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic,
                                 cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies]):
        err, = cudart.cudaMemPoolSetAttribute(pool, attr, 0)
        assertSuccess(err)
    for idx, attr in enumerate([cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold]):
        err, = cudart.cudaMemPoolSetAttribute(pool, attr, cuda.cuuint64_t(9))
        assertSuccess(err)

    for idx, attr in enumerate([cudart.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies,
                                cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic,
                                cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies,
                                cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold]):
        err, attr_tmp = cudart.cudaMemPoolGetAttribute(pool, attr)
        assertSuccess(err)
        attr_list[idx] = attr_tmp
    assert(attr_list[0] == 0)
    assert(attr_list[1] == 0)
    assert(attr_list[2] == 0)
    assert(int(attr_list[3]) == 9)

    err, = cudart.cudaMemPoolDestroy(pool)
    assertSuccess(err)

def test_cudart_make_api():
    err, channelDesc = cudart.cudaCreateChannelDesc(32,0,0,0,cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat)
    assertSuccess(err)
    assert(channelDesc.x == 32)
    assert(channelDesc.y == 0)
    assert(channelDesc.z == 0)
    assert(channelDesc.w == 0)
    assert(channelDesc.f == cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat)

    # make_cudaPitchedPtr
    cudaPitchedPtr = cudart.make_cudaPitchedPtr(1,2,3,4)
    assert(cudaPitchedPtr.ptr == 1)
    assert(cudaPitchedPtr.pitch == 2)
    assert(cudaPitchedPtr.xsize == 3)
    assert(cudaPitchedPtr.ysize == 4)

    # make_cudaPos
    cudaPos = cudart.make_cudaPos(1,2,3)
    assert(cudaPos.x == 1)
    assert(cudaPos.y == 2)
    assert(cudaPos.z == 3)

    # make_cudaExtent
    cudaExtent = cudart.make_cudaExtent(1,2,3)
    assert(cudaExtent.width == 1)
    assert(cudaExtent.height == 2)
    assert(cudaExtent.depth == 3)

def test_cudart_cudaStreamGetCaptureInfo():
    # create stream
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)

    # validate that stream is not capturing
    err, status, *info = cudart.cudaStreamGetCaptureInfo(stream)
    assertSuccess(err)
    assert(status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone)

    # start capture
    err, = cudart.cudaStreamBeginCapture(
        stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
    )
    assertSuccess(err)

    # validate that stream is capturing now
    err, status, *info = cudart.cudaStreamGetCaptureInfo(stream)
    assertSuccess(err)
    assert(status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive)

    # clean up
    err, pgraph = cudart.cudaStreamEndCapture(stream)
    assertSuccess(err)

def test_cudart_cudaArrayGetInfo():
    # create channel descriptor
    x, y, z, w = 8, 0, 0, 0
    f = cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    err, desc = cudart.cudaCreateChannelDesc(
        x, y, z, w, f
    )
    assertSuccess(err)

    # allocate device array
    width = 10
    height = 0
    inFlags = 0
    err, arr = cudart.cudaMallocArray(desc, width, height, inFlags)
    assertSuccess(err)

    # get device array info
    err, desc, extent, outFlags = cudart.cudaArrayGetInfo(arr)
    assertSuccess(err)

    # validate descriptor, extent, flags
    assert(desc.x == x)
    assert(desc.y == y)
    assert(desc.z == z)
    assert(desc.w == w)
    assert(desc.f == f)
    assert(extent.width == width)
    assert(extent.height == height)
    assert(inFlags == outFlags)

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)
    
def test_cudart_cudaMemcpy2DToArray():
    # create host arrays
    size = int(1024 * np.uint8().itemsize)
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to arr
    err, = cudart.cudaMemcpy2DToArray(
        arr, 0, 0, h1, size, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assertSuccess(err)

    # arr to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, arr, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)

def test_cudart_cudaMemcpy2DToArray_DtoD():
    # allocate device memory
    size = 1024 * np.uint8().itemsize
    err, d1 = cudart.cudaMalloc(size)
    assertSuccess(err)
    err, d2 = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to d1
    err, = cudart.cudaMemcpy(d1, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assertSuccess(err)

    # d1 to arr
    err, = cudart.cudaMemcpy2DToArray(
        arr, 0, 0, d1, size, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    )
    assertSuccess(err)

    # arr to d2
    err, = cudart.cudaMemcpy2DFromArray(
        d2, size, arr, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    )
    assertSuccess(err)

    # d2 to h2
    err, = cudart.cudaMemcpy(h2, d2, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)
    err, = cudart.cudaFree(d2)
    assertSuccess(err)
    err, = cudart.cudaFree(d1)
    assertSuccess(err)

def test_cudart_cudaMemcpy2DArrayToArray():
    # create host arrays
    size = 1024 * np.uint8().itemsize
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device arrays
    err, a1 = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)
    err, a2 = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to a1
    err, = cudart.cudaMemcpy2DToArray(
        a1, 0, 0, h1, size, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assertSuccess(err)

    # a1 to a2
    err, = cudart.cudaMemcpy2DArrayToArray(
        a2, 0, 0, a1, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice    
    )
    assertSuccess(err)

    # a2 to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, a2, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(a2)
    assertSuccess(err)
    err, = cudart.cudaFreeArray(a1)
    assertSuccess(err)

def test_cudart_cudaMemcpyArrayToArray():
    # create host arrays
    size = 1024 * np.uint8().itemsize
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device arrays
    err, a1 = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)
    err, a2 = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to a1
    err, = cudart.cudaMemcpy2DToArray(
        a1, 0, 0, h1, size, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assertSuccess(err)

    # a1 to a2
    err, = cudart.cudaMemcpyArrayToArray(
        a2, 0, 0, a1, 0, 0, size,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice    
    )
    assertSuccess(err)

    # a2 to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, a2, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(a2)
    assertSuccess(err)
    err, = cudart.cudaFreeArray(a1)
    assertSuccess(err)

def test_cudart_cudaGetChannelDesc():
    # create channel descriptor
    x, y, z, w = 8, 0, 0, 0
    f = cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    err, desc = cudart.cudaCreateChannelDesc(
        x, y, z, w, f
    )
    assertSuccess(err)

    # allocate device array
    width = 10
    height = 0
    flags = 0
    err, arr = cudart.cudaMallocArray(desc, width, height, flags)
    assertSuccess(err)

    # get channel descriptor from array
    err, desc = cudart.cudaGetChannelDesc(arr)
    assertSuccess(err)

    # validate array channel descriptor
    assert(desc.x == x)
    assert(desc.y == y)
    assert(desc.z == z)
    assert(desc.w == w)
    assert(desc.f == f)

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)

def test_cudart_cudaGetTextureObjectTextureDesc():
    # create channel descriptor
    err, channelDesc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device arrays
    err, arr = cudart.cudaMallocArray(channelDesc, 1024, 0, 0)
    assertSuccess(err)

    # create descriptors for texture object
    resDesc = cudart.cudaResourceDesc()
    resDesc.res.array.array = arr
    inTexDesc = cudart.cudaTextureDesc()

    # create texture object
    err, texObject = cudart.cudaCreateTextureObject(resDesc, inTexDesc, None)
    assertSuccess(err)

    # get texture descriptor
    err, outTexDesc = cudart.cudaGetTextureObjectTextureDesc(texObject)
    assertSuccess(err)

    # validate texture descriptor
    for attr in dir(outTexDesc):
        if attr in ["borderColor", "getPtr"]:
            continue
        if not attr.startswith("_"):
            assert(getattr(outTexDesc, attr) == getattr(inTexDesc, attr))
    
    # clean up
    err, = cudart.cudaDestroyTextureObject(texObject)
    assertSuccess(err)

def test_cudart_cudaMemset3D():
    # create host arrays
    size = 1024 * np.uint8().itemsize
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # allocate device memory
    devExtent = cudart.make_cudaExtent(32, 32, 1)
    err, devPitchedPtr = cudart.cudaMalloc3D(devExtent)
    assertSuccess(err)

    # set memory
    memExtent = cudart.make_cudaExtent(devPitchedPtr.pitch, devPitchedPtr.ysize, 1)
    err, = cudart.cudaMemset3D(devPitchedPtr, 1, memExtent)
    assertSuccess(err)

    # D to h2
    err, = cudart.cudaMemcpy(
        h2, devPitchedPtr.ptr, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFree(devPitchedPtr.ptr)
    assertSuccess(err)

def test_cudart_cudaMemset3D_2D():
    # create host arrays
    size = 512 * np.uint8().itemsize
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # allocate device memory
    devExtent = cudart.make_cudaExtent(1024, 1, 1)
    err, devPitchedPtr = cudart.cudaMalloc3D(devExtent)
    assertSuccess(err)

    # set memory
    memExtent = cudart.make_cudaExtent(size, devPitchedPtr.ysize, 1)
    err, = cudart.cudaMemset3D(devPitchedPtr, 1, memExtent)
    assertSuccess(err)

    # D to h2
    err, = cudart.cudaMemcpy(
        h2, devPitchedPtr.ptr, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFree(devPitchedPtr.ptr)
    assertSuccess(err)

def test_cudart_cudaMemcpyToArray():
    # create host arrays
    size = 1024 * np.uint8().itemsize
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)
    
    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to arr
    err, = cudart.cudaMemcpyToArray(
        arr, 0, 0, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assertSuccess(err)

    # arr to h2
    err, = cudart.cudaMemcpyFromArray(
        h2, arr, 0, 0, size,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)

def test_cudart_cudaMemcpyToArray_DtoD():
    # allocate device memory
    size = int(1024 * np.uint8().itemsize)
    err, d1 = cudart.cudaMalloc(size)
    assertSuccess(err)
    err, d2 = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)
    
    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # h1 to d1
    err, = cudart.cudaMemcpy(d1, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assertSuccess(err)

    # d1 to arr
    err, = cudart.cudaMemcpyToArray(
        arr, 0, 0, d1, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    )
    assertSuccess(err)

    # arr to d2
    err, = cudart.cudaMemcpyFromArray(
        d2, arr, 0, 0, size,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    )
    assertSuccess(err)

    # d2 to h2
    err, = cudart.cudaMemcpy(h2, d2, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)
    err, = cudart.cudaFree(d2)
    assertSuccess(err)
    err, = cudart.cudaFree(d1)
    assertSuccess(err)

def test_cudart_cudaMemcpy3DAsync():
    # create host arrays
    size = int(1024 * np.uint8().itemsize)
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # create stream
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)

    # create memcpy params
    params = cudart.cudaMemcpy3DParms()
    params.srcPtr = cudart.make_cudaPitchedPtr(h1, size, 1, 1)
    params.dstArray = arr
    params.extent = cudart.make_cudaExtent(size, 1, 1)
    params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice

    # h1 to arr
    err, = cudart.cudaMemcpy3DAsync(params, stream)
    assertSuccess(err)

    # await results
    err, = cudart.cudaStreamSynchronize(stream)
    assertSuccess(err)

    # arr to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, arr, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)

def test_cudart_cudaGraphAddMemcpyNode1D():
    # allocate device memory
    size = 1024 * np.uint8().itemsize
    err, dptr = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # build graph
    err, graph = cudart.cudaGraphCreate(0)
    assertSuccess(err)

    # add nodes
    err, hToDNode = cudart.cudaGraphAddMemcpyNode1D(
        graph, [], 0, dptr, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assertSuccess(err)
    err, dToHNode = cudart.cudaGraphAddMemcpyNode1D(
        graph, [hToDNode], 1, h2, dptr, size,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # create stream
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)

    # execute graph
    err, execGraph = cudart.cudaGraphInstantiate(graph, 0)
    assertSuccess(err)
    err, = cudart.cudaGraphLaunch(execGraph, stream)

    # await results
    err, = cudart.cudaStreamSynchronize(stream)
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFree(dptr)
    assertSuccess(err)

def test_cudart_cudaGraphAddMemsetNode():
    # allocate device memory
    size = 1024 * np.uint8().itemsize
    err, dptr = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # build graph
    err, graph = cudart.cudaGraphCreate(0)
    assertSuccess(err)

    # set memset params
    params = cudart.cudaMemsetParams()
    params.dst = dptr
    params.pitch = size
    params.value = 1
    params.elementSize = 1
    params.width = size
    params.height = 1

    # add nodes
    err, setNode = cudart.cudaGraphAddMemsetNode(
        graph, [], 0, params
    )
    assertSuccess(err)
    err, cpyNode = cudart.cudaGraphAddMemcpyNode1D(
        graph, [setNode], 1, h2, dptr, size,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)


    # create stream
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)

    # execute graph
    err, execGraph = cudart.cudaGraphInstantiate(graph, 0)
    assertSuccess(err)
    err, = cudart.cudaGraphLaunch(execGraph, stream)
    assertSuccess(err)

    # await results
    err, = cudart.cudaStreamSynchronize(stream)
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFree(dptr)
    assertSuccess(err)

def test_cudart_cudaMemcpy3DPeer():
    # allocate device memory
    size = int(1024 * np.uint8().itemsize)
    err, dptr = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # create memcpy params
    params = cudart.cudaMemcpy3DPeerParms()
    params.srcPtr = cudart.make_cudaPitchedPtr(dptr, size, 1, 1)
    params.dstArray = arr
    params.extent = cudart.make_cudaExtent(size, 1, 1)

    # h1 to D
    err, = cudart.cudaMemcpy(dptr, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assertSuccess(err)

    # D to arr
    err, = cudart.cudaMemcpy3DPeer(params)
    assertSuccess(err)

    # arr to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, arr, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)
    err, = cudart.cudaFree(dptr)
    assertSuccess(err)

def test_cudart_cudaMemcpy3DPeerAsync():
    # allocate device memory
    size = 1024 * np.uint8().itemsize
    err, dptr = cudart.cudaMalloc(size)
    assertSuccess(err)

    # create host arrays
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert(np.array_equal(h1, h2) is False)

    # create channel descriptor
    err, desc = cudart.cudaCreateChannelDesc(
        8, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    )
    assertSuccess(err)

    # allocate device array
    err, arr = cudart.cudaMallocArray(desc, size, 0, 0)
    assertSuccess(err)

    # create stream
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)

    # create memcpy params
    params = cudart.cudaMemcpy3DPeerParms()
    params.srcPtr = cudart.make_cudaPitchedPtr(dptr, size, 1, 1)
    params.dstArray = arr
    params.extent = cudart.make_cudaExtent(size, 1, 1)

    # h1 to D
    err, = cudart.cudaMemcpy(dptr, h1, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assertSuccess(err)

    # D to arr
    err, = cudart.cudaMemcpy3DPeerAsync(params, stream)
    assertSuccess(err)

    # await results
    err, = cudart.cudaStreamSynchronize(stream)
    assertSuccess(err)

    # arr to h2
    err, = cudart.cudaMemcpy2DFromArray(
        h2, size, arr, 0, 0, size, 1,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assertSuccess(err)

    # validate h1 == h2
    assert(np.array_equal(h1, h2))

    # clean up
    err, = cudart.cudaFreeArray(arr)
    assertSuccess(err)
    err, = cudart.cudaFree(dptr)
    assertSuccess(err)

def test_profiler():
    err, = cudart.cudaProfilerStart()
    assertSuccess(err)
    err, = cudart.cudaProfilerStop()
    assertSuccess(err)

def test_cudart_eglFrame():
    frame = cudart.cudaEglFrame()
    # [<cudaArray_t 0x0>, <cudaArray_t 0x0>, <cudaArray_t 0x0>]
    assert(int(frame.frame.pArray[0]) == 0)
    assert(int(frame.frame.pArray[1]) == 0)
    assert(int(frame.frame.pArray[2]) == 0)
    frame.frame.pArray = [1,2,3]
    # [<cudaArray_t 0x1>, <cudaArray_t 0x2>, <cudaArray_t 0x3>]
    assert(int(frame.frame.pArray[0]) == 1)
    assert(int(frame.frame.pArray[1]) == 2)
    assert(int(frame.frame.pArray[2]) == 3)
    frame.frame.pArray = [1,2,cudart.cudaArray_t(4)]
    # [<cudaArray_t 0x1>, <cudaArray_t 0x2>, <cudaArray_t 0x4>]
    assert(int(frame.frame.pArray[0]) == 1)
    assert(int(frame.frame.pArray[1]) == 2)
    assert(int(frame.frame.pArray[2]) == 4)
    # frame.frame.pPitch
    # [ptr : 0x1
    # pitch : 2
    # xsize : 4
    # ysize : 0, ptr : 0x0
    # pitch : 0
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 0
    # xsize : 0
    # ysize : 0]
    assert(int(frame.frame.pPitch[0].ptr) == 1)
    assert(int(frame.frame.pPitch[0].pitch) == 2)
    assert(int(frame.frame.pPitch[0].xsize) == 4)
    assert(int(frame.frame.pPitch[0].ysize) == 0)
    assert(int(frame.frame.pPitch[1].ptr) == 0)
    assert(int(frame.frame.pPitch[1].pitch) == 0)
    assert(int(frame.frame.pPitch[1].xsize) == 0)
    assert(int(frame.frame.pPitch[1].ysize) == 0)
    assert(int(frame.frame.pPitch[2].ptr) == 0)
    assert(int(frame.frame.pPitch[2].pitch) == 0)
    assert(int(frame.frame.pPitch[2].xsize) == 0)
    assert(int(frame.frame.pPitch[2].ysize) == 0)
    frame.frame.pPitch = [cudart.cudaPitchedPtr(), cudart.cudaPitchedPtr(), cudart.cudaPitchedPtr()]
    # [ptr : 0x0
    # pitch : 0
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 0
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 0
    # xsize : 0
    # ysize : 0]
    assert(int(frame.frame.pPitch[0].ptr) == 0)
    assert(int(frame.frame.pPitch[0].pitch) == 0)
    assert(int(frame.frame.pPitch[0].xsize) == 0)
    assert(int(frame.frame.pPitch[0].ysize) == 0)
    assert(int(frame.frame.pPitch[1].ptr) == 0)
    assert(int(frame.frame.pPitch[1].pitch) == 0)
    assert(int(frame.frame.pPitch[1].xsize) == 0)
    assert(int(frame.frame.pPitch[1].ysize) == 0)
    assert(int(frame.frame.pPitch[2].ptr) == 0)
    assert(int(frame.frame.pPitch[2].pitch) == 0)
    assert(int(frame.frame.pPitch[2].xsize) == 0)
    assert(int(frame.frame.pPitch[2].ysize) == 0)
    x = frame.frame.pPitch[0]
    x.pitch = 123
    frame.frame.pPitch = [x,x,x]
    # [ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0]
    assert(int(frame.frame.pPitch[0].ptr) == 0)
    assert(int(frame.frame.pPitch[0].pitch) == 123)
    assert(int(frame.frame.pPitch[0].xsize) == 0)
    assert(int(frame.frame.pPitch[0].ysize) == 0)
    assert(int(frame.frame.pPitch[1].ptr) == 0)
    assert(int(frame.frame.pPitch[1].pitch) == 123)
    assert(int(frame.frame.pPitch[1].xsize) == 0)
    assert(int(frame.frame.pPitch[1].ysize) == 0)
    assert(int(frame.frame.pPitch[2].ptr) == 0)
    assert(int(frame.frame.pPitch[2].pitch) == 123)
    assert(int(frame.frame.pPitch[2].xsize) == 0)
    assert(int(frame.frame.pPitch[2].ysize) == 0)
    x.pitch = 1234
    # [ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0, ptr : 0x0
    # pitch : 123
    # xsize : 0
    # ysize : 0]
    assert(int(frame.frame.pPitch[0].ptr) == 0)
    assert(int(frame.frame.pPitch[0].pitch) == 123)
    assert(int(frame.frame.pPitch[0].xsize) == 0)
    assert(int(frame.frame.pPitch[0].ysize) == 0)
    assert(int(frame.frame.pPitch[1].ptr) == 0)
    assert(int(frame.frame.pPitch[1].pitch) == 123)
    assert(int(frame.frame.pPitch[1].xsize) == 0)
    assert(int(frame.frame.pPitch[1].ysize) == 0)
    assert(int(frame.frame.pPitch[2].ptr) == 0)
    assert(int(frame.frame.pPitch[2].pitch) == 123)
    assert(int(frame.frame.pPitch[2].xsize) == 0)
    assert(int(frame.frame.pPitch[2].ysize) == 0)

def cudart_func_stream_callback(use_host_api):
    class testStruct(ctypes.Structure):
        _fields_ = [('a', ctypes.c_int),
                    ('b', ctypes.c_int),
                    ('c', ctypes.c_int),]

    def task_callback_host(userData):
        data = testStruct.from_address(userData)
        assert(data.a == 1)
        assert(data.b == 2)
        assert(data.c == 3)
        return 0

    def task_callback_stream(stream, status, userData):
        data = testStruct.from_address(userData)
        assert(data.a == 1)
        assert(data.b == 2)
        assert(data.c == 3)
        return 0

    if use_host_api:
        callback_type = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        target_task = task_callback_host
    else:
        callback_type = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
        target_task = task_callback_stream

    # Construct ctype data
    c_callback = callback_type(target_task)
    c_data = testStruct(1, 2, 3)

    # ctypes is managing the pointer value for us
    if use_host_api:
        callback = cudart.cudaHostFn_t(_ptr=ctypes.addressof(c_callback))
    else:
        callback = cudart.cudaStreamCallback_t(_ptr=ctypes.addressof(c_callback))

    # Run
    err, stream = cudart.cudaStreamCreate()
    assertSuccess(err)
    if use_host_api:
        err, = cudart.cudaLaunchHostFunc(stream, callback, ctypes.addressof(c_data))
        assertSuccess(err)
    else:
        err, = cudart.cudaStreamAddCallback(stream, callback, ctypes.addressof(c_data), 0)
        assertSuccess(err)
    err, = cudart.cudaDeviceSynchronize()
    assertSuccess(err)


def test_cudart_func_callback():
    cudart_func_stream_callback(use_host_api=False)
    cudart_func_stream_callback(use_host_api=True)

@pytest.mark.skipif(driverVersionLessThan(12030)
                    or not supportsCudaAPI('cudaGraphConditionalHandleCreate'), reason='Conditional graph APIs required')
def test_cudart_conditional():
    err, graph = cudart.cudaGraphCreate(0)
    assertSuccess(err)
    err, handle = cudart.cudaGraphConditionalHandleCreate(graph, 0, 0)
    assertSuccess(err)

    params = cudart.cudaGraphNodeParams()
    params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
    params.conditional.handle = handle
    params.conditional.type = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
    params.conditional.size = 1

    assert(len(params.conditional.phGraph_out) == 1)
    assert(int(params.conditional.phGraph_out[0]) == 0)
    err, node = cudart.cudaGraphAddNode(graph, None, 0, params)
    assertSuccess(err)

    assert(len(params.conditional.phGraph_out) == 1)
    assert(int(params.conditional.phGraph_out[0]) != 0)
