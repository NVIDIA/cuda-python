# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import cython
from cuda.ccudart cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp, memcmp
from libcpp cimport bool
cimport cuda._cuda.ccuda as ccuda

cdef struct cudaArrayLocalState:
    ccuda.CUarray array
    cudaChannelFormatDesc desc
    size_t depth
    size_t height
    size_t width
    size_t elementSize
    size_t widthInBytes

ctypedef struct cudaStreamCallbackData_st:
    cudaStreamCallback_t callback
    void *userData

ctypedef cudaStreamCallbackData_st cudaStreamCallbackData

ctypedef struct cudaStreamHostCallbackData_st:
    cudaHostFn_t callback
    void *userData

ctypedef cudaStreamHostCallbackData_st cudaStreamHostCallbackData

cdef class cudaPythonGlobal:
    def __cinit__(self):
        self._lazyInitDriver = False
        self._numDevices = 0
        self._deviceList = NULL
        self._CUDART_VERSION = CUDART_VERSION

    def __dealloc__(self):
        if self._deviceList is not NULL:
            free(self._deviceList)
        for item in self._asyncCallbackDataMap:
            free(item.second)
        self._asyncCallbackDataMap.clear()

    cdef cudaError_t lazyInitDriver(self) except ?cudaErrorCallRequiresNewerDriver nogil:
        if self._lazyInitDriver:
            return cudaSuccess

        cdef cudaError_t err = cudaSuccess
        err = <cudaError_t>ccuda._cuInit(0)
        if err != cudaSuccess:
            return err
        err = <cudaError_t>ccuda._cuDeviceGetCount(&self._numDevices)
        if err != cudaSuccess:
            return err

        self._deviceList = <cudaPythonDevice *>calloc(self._numDevices, sizeof(cudaPythonDevice))
        if self._deviceList == NULL:
            return cudaErrorMemoryAllocation

        for deviceOrdinal in range(self._numDevices):
            err = initDevice(&self._deviceList[deviceOrdinal], deviceOrdinal)
            if err != cudaSuccess:
                free(self._deviceList)
                return err

        self._lazyInitDriver = True

    cdef cudaError_t lazyInitContextState(self) except ?cudaErrorCallRequiresNewerDriver nogil:
        cdef cudaError_t err = cudaSuccess
        cdef ccuda.CUcontext driverContext
        cdef cudaPythonDevice *device

        err = self.lazyInitDriver()
        if err != cudaSuccess:
            return err

        err = <cudaError_t>ccuda._cuCtxGetCurrent(&driverContext)
        if err != cudaSuccess:
            return err
        device = self.getDeviceFromPrimaryCtx(driverContext)

        # 1. Context + device
        if driverContext != NULL and device != NULL:
            err = initPrimaryContext(device)
            if err != cudaSuccess:
                return err

        # 2. Context + no device
        cdef unsigned int version
        if driverContext != NULL:
            # If the context exists, but is non-primary, make sure it can be used with the CUDA 3.2 API,
            # then return immediately
            err = <cudaError_t>ccuda._cuCtxGetApiVersion(driverContext, &version)
            if err == cudaErrorContextIsDestroyed:
                return cudaErrorIncompatibleDriverContext
            elif err != cudaSuccess:
                return err
            elif version < 3020:
                return cudaErrorIncompatibleDriverContext
            return cudaSuccess

        # 3. No context + device
        # (impossible)

        # 4. No context + no device
        # Default to first device
        device = self.getDevice(0)
        err = initPrimaryContext(device)
        if err != cudaSuccess:
            return err
        err = <cudaError_t> ccuda._cuCtxSetCurrent(device.primaryContext)
        return err

    cdef cudaPythonDevice* getDevice(self, int deviceOrdinal) noexcept nogil:
        if deviceOrdinal < 0 or deviceOrdinal >= m_global._numDevices:
            return NULL
        return &self._deviceList[deviceOrdinal]

    cdef cudaPythonDevice* getDeviceFromDriver(self, ccuda.CUdevice driverDevice) noexcept nogil:
        for i in range(self._numDevices):
            if self._deviceList[i].driverDevice == driverDevice:
                return &self._deviceList[i]
        return NULL

    cdef cudaPythonDevice* getDeviceFromPrimaryCtx(self, ccuda.CUcontext context) noexcept nogil:
        if context == NULL:
            return NULL
        for i in range(self._numDevices):
            if self._deviceList[i].primaryContext == context:
                return &self._deviceList[i]
        return NULL

cdef cudaPythonGlobal m_global = cudaPythonGlobal()


cdef cudaError_t initDevice(cudaPythonDevice *device, int deviceOrdinal) except ?cudaErrorCallRequiresNewerDriver nogil:
    # ccuda.CUcontext primaryContext
    device[0].primaryContext = NULL
    # bool primaryContextRetained
    device[0].primaryContextRetained = False
    # int deviceOrdinal
    device[0].deviceOrdinal = deviceOrdinal

    # ccuda.CUdevice driverDevice
    err = ccuda._cuDeviceGet(&device[0].driverDevice, deviceOrdinal)
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    # cudaDeviceProp deviceProperties
    err = ccuda._cuDeviceGetName(device[0].deviceProperties.name, sizeof(device[0].deviceProperties.name), <ccuda.CUdevice>deviceOrdinal)
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceTotalMem_v2(&(device[0].deviceProperties.totalGlobalMem), <ccuda.CUdevice>deviceOrdinal)
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceTotalMem_v2(&(device[0].deviceProperties.totalGlobalMem), <ccuda.CUdevice>deviceOrdinal)
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.major), ccuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.minor), ccuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.deviceOverlap), ccuda.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.asyncEngineCount), ccuda.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.multiProcessorCount), ccuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.kernelExecTimeoutEnabled), ccuda.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.integrated), ccuda.CU_DEVICE_ATTRIBUTE_INTEGRATED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.canMapHostMemory), ccuda.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture1D), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture1DMipmap), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture1DLinear), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2D[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2D[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DMipmap[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DMipmap[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLinear[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLinear[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLinear[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DGather[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DGather[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3D[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3D[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3D[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3DAlt[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3DAlt[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture3DAlt[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTextureCubemap), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture1DLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture1DLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTexture2DLayered[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTextureCubemapLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxTextureCubemapLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface1D), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface2D[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface2D[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface3D[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface3D[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface3D[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface1DLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface1DLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface2DLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface2DLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurface2DLayered[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurfaceCubemap), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurfaceCubemapLayered[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxSurfaceCubemapLayered[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.concurrentKernels), ccuda.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.ECCEnabled), ccuda.CU_DEVICE_ATTRIBUTE_ECC_ENABLED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.pciBusID), ccuda.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.pciDeviceID), ccuda.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.pciDomainID), ccuda.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.tccDriver), ccuda.CU_DEVICE_ATTRIBUTE_TCC_DRIVER, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.unifiedAddressing), ccuda.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.memoryClockRate), ccuda.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.memoryBusWidth), ccuda.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.l2CacheSize), ccuda.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.persistingL2CacheMaxSize), ccuda.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxThreadsPerMultiProcessor), ccuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    cdef int surfaceAlignment
    err = ccuda._cuDeviceGetAttribute(&(surfaceAlignment), ccuda.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.surfaceAlignment = surfaceAlignment

    cdef int texturePitchAlignment
    err = ccuda._cuDeviceGetAttribute(&texturePitchAlignment, ccuda.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.texturePitchAlignment = texturePitchAlignment

    cdef int sharedMemPerBlock
    err = ccuda._cuDeviceGetAttribute(&sharedMemPerBlock, ccuda.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.sharedMemPerBlock = sharedMemPerBlock

    cdef int sharedMemPerBlockOptin
    err = ccuda._cuDeviceGetAttribute(&sharedMemPerBlockOptin, ccuda.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.sharedMemPerBlockOptin = sharedMemPerBlockOptin

    cdef int sharedMemPerMultiprocessor
    err = ccuda._cuDeviceGetAttribute(&sharedMemPerMultiprocessor, ccuda.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.sharedMemPerMultiprocessor = sharedMemPerMultiprocessor

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.regsPerBlock), ccuda.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.regsPerMultiprocessor), ccuda.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.warpSize), ccuda.CU_DEVICE_ATTRIBUTE_WARP_SIZE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    cdef int memPitch
    err = ccuda._cuDeviceGetAttribute(&memPitch, ccuda.CU_DEVICE_ATTRIBUTE_MAX_PITCH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.memPitch = memPitch

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxThreadsPerBlock), ccuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxThreadsDim[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxThreadsDim[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxThreadsDim[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxGridSize[0]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxGridSize[1]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxGridSize[2]), ccuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    cdef int totalConstMem
    err = ccuda._cuDeviceGetAttribute(&totalConstMem, ccuda.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.totalConstMem = totalConstMem

    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.clockRate), ccuda.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    cdef int textureAlignment
    err = ccuda._cuDeviceGetAttribute(&textureAlignment, ccuda.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.textureAlignment = textureAlignment
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.streamPrioritiesSupported), ccuda.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.globalL1CacheSupported), ccuda.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.localL1CacheSupported), ccuda.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.managedMemory), ccuda.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.isMultiGpuBoard), ccuda.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.multiGpuBoardGroupID), ccuda.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.hostNativeAtomicSupported), ccuda.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.singleToDoublePrecisionPerfRatio), ccuda.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.pageableMemoryAccess), ccuda.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.concurrentManagedAccess), ccuda.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.computePreemptionSupported), ccuda.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.canUseHostPointerForRegisteredMem), ccuda.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.cooperativeLaunch), ccuda.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.cooperativeMultiDeviceLaunch), ccuda.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.pageableMemoryAccessUsesHostPageTables), ccuda.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.directManagedMemAccessFromHost), ccuda.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    err = ccuda._cuDeviceGetUuid(<ccuda.CUuuid_st*>(&(device[0].deviceProperties.uuid)), <ccuda.CUdevice>deviceOrdinal)
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.maxBlocksPerMultiProcessor), ccuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    err = ccuda._cuDeviceGetAttribute(&(device[0].deviceProperties.accessPolicyMaxWindowSize), ccuda.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError

    cdef int reservedSharedMemPerBlock
    err = ccuda._cuDeviceGetAttribute(&reservedSharedMemPerBlock, ccuda.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, <ccuda.CUdevice>(deviceOrdinal))
    if err != ccuda.cudaError_enum.CUDA_SUCCESS:
        return cudaErrorInitializationError
    device[0].deviceProperties.reservedSharedMemPerBlock = reservedSharedMemPerBlock

    return cudaSuccess


cdef cudaError_t initPrimaryContext(cudaPythonDevice *device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess

    # If we have context retained we need to check if it is not reset
    cdef unsigned int version
    if device[0].primaryContextRetained:
        err = <cudaError_t>ccuda._cuCtxGetApiVersion(device[0].primaryContext, &version)
        if err == cudaErrorDeviceUninitialized:
            err = <cudaError_t>ccuda.cuDevicePrimaryCtxRelease(device[0].driverDevice)
            if err != cudaSuccess:
                return err
            device[0].primaryContextRetained = False
        elif err != cudaSuccess:
            return err

    # If we don't or it is invalid we need to recreate it
    if not device[0].primaryContextRetained:
        err = <cudaError_t>ccuda._cuDevicePrimaryCtxRetain(&device[0].primaryContext, device[0].driverDevice)
        if err != cudaSuccess:
            return err
        device[0].primaryContextRetained = True
    return err

cdef cudaError_t resetPrimaryContext(cudaPythonDevice* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int version

    err = <cudaError_t>ccuda._cuCtxGetApiVersion(device[0].primaryContext, &version)
    if err == cudaSuccess:
        if not device[0].primaryContextRetained:
            err = <cudaError_t>ccuda._cuDevicePrimaryCtxRetain(&device[0].primaryContext, device[0].driverDevice)
            if err != cudaSuccess:
                return err
            device[0].primaryContextRetained = True
        ccuda._cuDevicePrimaryCtxReset_v2(device[0].driverDevice)
        return cudaSuccess
    elif err == cudaErrorDeviceUninitialized:
        return cudaSuccess
    else:
        return err


cdef cudaPythonGlobal globalGetInstance():
    return m_global


cdef cudaError_t _setLastError(cudaError_t err) except ?cudaErrorCallRequiresNewerDriver nogil:
    if err != cudaSuccess:
        m_global._lastError = err


cdef int case_desc(const cudaChannelFormatDesc* d, int x, int y, int z, int w, int f) except ?cudaErrorCallRequiresNewerDriver nogil:
    return d[0].x == x and d[0].y == y and d[0].z == z and d[0].w == w and d[0].f == f


cdef cudaError_t getDescInfo(const cudaChannelFormatDesc* d, int *numberOfChannels, ccuda.CUarray_format *format) except ?cudaErrorCallRequiresNewerDriver nogil:
    # Check validity
    if d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSigned,
                  cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        if (d[0].x != 8) and (d[0].x != 16) and (d[0].x != 32):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindFloat,):
        if (d[0].x != 16) and (d[0].x != 32):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindNV12,):
        if (d[0].x != 8) or (d[0].y != 8) or (d[0].z != 8) or (d[0].w != 0):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4,):
        if (d[0].x != 8):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4,):
        if (d[0].x != 16):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7,
                    cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB,):
        if (d[0].x != 8):
            return cudaErrorInvalidChannelDescriptor
    elif d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H,
                    cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H,):
        if (d[0].x != 16) or (d[0].y != 16) or (d[0].z != 16) or (d[0].w != 0):
            return cudaErrorInvalidChannelDescriptor
    else:
        return cudaErrorInvalidChannelDescriptor

    # If Y is non-zero, it must match X
    # If Z is non-zero, it must match Y
    # If W is non-zero, it must match Z
    if (((d[0].y != 0) and (d[0].y != d[0].x)) or
        ((d[0].z != 0) and (d[0].z != d[0].y)) or
        ((d[0].w != 0) and (d[0].w != d[0].z))):
        return cudaErrorInvalidChannelDescriptor
    if case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindSigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindUnsigned):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32
    elif case_desc(d, 16, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 16, 16, 16, 16, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_HALF
    elif case_desc(d, 32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 32, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 32, 32, 32, 32, cudaChannelFormatKind.cudaChannelFormatKindFloat):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_FLOAT
    elif case_desc(d, 8, 8, 8, 0, cudaChannelFormatKind.cudaChannelFormatKindNV12):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_NV12
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC1_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC1_UNORM_SRGB
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC2_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC2_UNORM_SRGB
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC3_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC3_UNORM_SRGB
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC4_UNORM
    elif case_desc(d, 8, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4):
        numberOfChannels[0] = 1
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC4_SNORM
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC5_UNORM
    elif case_desc(d, 8, 8, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5):
        numberOfChannels[0] = 2
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC5_SNORM
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC6H_UF16
    elif case_desc(d, 16, 16, 16, 0, cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H):
        numberOfChannels[0] = 3
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC6H_SF16
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC7_UNORM
    elif case_desc(d, 8, 8, 8, 8, cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB):
        numberOfChannels[0] = 4
        format[0] = ccuda.CUarray_format_enum.CU_AD_FORMAT_BC7_UNORM_SRGB
    else:
        return cudaErrorInvalidChannelDescriptor

    if d[0].f in (cudaChannelFormatKind.cudaChannelFormatKindNV12,
                  cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H,
                  cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H,):
        if numberOfChannels[0] != 3:
            return cudaErrorInvalidChannelDescriptor
    else:
        if (numberOfChannels[0] != 1) and (numberOfChannels[0] != 2) and (numberOfChannels[0] != 4):
            return cudaErrorInvalidChannelDescriptor
    return cudaSuccess


@cython.show_performance_hints(False)
cdef void cudaStreamRtCallbackWrapper(ccuda.CUstream stream, ccuda.CUresult status, void *data) nogil:
    cdef cudaStreamCallbackData *cbData = <cudaStreamCallbackData *>data
    cdef cudaError_t err = <cudaError_t>status
    with gil:
        cbData.callback(stream, err, cbData.userData)
    free(cbData)


cdef cudaError_t streamAddCallbackCommon(
  cudaStream_t stream,
  cudaStreamCallback_t callback,
  void *userData,
  unsigned int flags
) except ?cudaErrorCallRequiresNewerDriver nogil:
    if callback == NULL:
        return cudaErrorInvalidValue

    cdef cudaStreamCallbackData *cbData = NULL
    cdef cudaError_t err = cudaSuccess
    cbData = <cudaStreamCallbackData *>malloc(sizeof(cbData[0]))

    if cbData == NULL:
        return cudaErrorMemoryAllocation

    cbData.callback = callback
    cbData.userData = userData
    err = <cudaError_t>ccuda._cuStreamAddCallback(stream, <ccuda.CUstreamCallback>cudaStreamRtCallbackWrapper, <void *>cbData, flags)
    if err != cudaSuccess:
        free(cbData)
    return err


@cython.show_performance_hints(False)
cdef void cudaStreamRtHostCallbackWrapper(void *data) nogil:
    cdef cudaStreamHostCallbackData *cbData = <cudaStreamHostCallbackData *>data
    with gil:
        cbData.callback(cbData.userData)
    free(cbData)


cdef cudaError_t streamAddHostCallbackCommon(
  cudaStream_t stream,
  cudaHostFn_t callback,
  void *userData
) except ?cudaErrorCallRequiresNewerDriver nogil:
    if callback == NULL:
        return cudaErrorInvalidValue

    cdef cudaStreamHostCallbackData *cbData = NULL
    cdef cudaError_t err = cudaSuccess
    cbData = <cudaStreamHostCallbackData *>malloc(sizeof(cbData[0]))

    if cbData == NULL:
        return cudaErrorMemoryAllocation

    cbData.callback = callback
    cbData.userData = userData
    err = <cudaError_t>ccuda._cuLaunchHostFunc(<ccuda.CUstream>stream, <ccuda.CUhostFn>cudaStreamRtHostCallbackWrapper, <void *>cbData)
    if err != cudaSuccess:
        free(cbData)
    return err


cdef cudaError_t toRuntimeStreamCaptureStatus(ccuda.CUstreamCaptureStatus driverCaptureStatus, cudaStreamCaptureStatus *runtimeStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    if driverCaptureStatus == ccuda.CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_NONE:
        runtimeStatus[0] = cudaStreamCaptureStatus.cudaStreamCaptureStatusNone
    elif driverCaptureStatus == ccuda.CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        runtimeStatus[0] = cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    elif driverCaptureStatus == ccuda.CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
        runtimeStatus[0] = cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated
    else:
         return cudaErrorUnknown
    return cudaSuccess


cdef cudaError_t streamGetCaptureInfoCommon(
        cudaStream_t stream,
        cudaStreamCaptureStatus* captureStatus_out,
        unsigned long long *id_out,
        cudaGraph_t *graph_out,
        const cudaGraphNode_t **dependencies_out,
        size_t *numDependencies_out)  except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess

    if captureStatus_out == NULL:
        return cudaErrorInvalidValue

    cdef ccuda.CUstreamCaptureStatus driverCaptureStatus

    err = <cudaError_t>ccuda._cuStreamGetCaptureInfo_v2(stream, &driverCaptureStatus, <ccuda.cuuint64_t*>id_out,
            graph_out, dependencies_out, numDependencies_out)
    if err != cudaSuccess:
        return err

    return toRuntimeStreamCaptureStatus(driverCaptureStatus, captureStatus_out)


cdef cudaError_t streamGetCaptureInfoCommon_v3(
        cudaStream_t stream,
        cudaStreamCaptureStatus* captureStatus_out,
        unsigned long long *id_out,
        cudaGraph_t *graph_out,
        const cudaGraphNode_t **dependencies_out,
        const cudaGraphEdgeData** edgeData_out,
        size_t *numDependencies_out)  except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess

    if captureStatus_out == NULL:
        return cudaErrorInvalidValue

    cdef ccuda.CUstreamCaptureStatus driverCaptureStatus

    err = <cudaError_t>ccuda._cuStreamGetCaptureInfo_v3(stream, &driverCaptureStatus, <ccuda.cuuint64_t*>id_out,
            graph_out, dependencies_out, <const ccuda.CUgraphEdgeData**>edgeData_out, numDependencies_out)
    if err != cudaSuccess:
        return err

    return toRuntimeStreamCaptureStatus(driverCaptureStatus, captureStatus_out)


cdef ccuda.CUDA_MEMCPY3D_v2 memCopy3DInit(ccuda.CUmemorytype_enum dstType, ccuda.CUmemorytype_enum srcType) noexcept nogil:
    cdef ccuda.CUDA_MEMCPY3D_v2 cp
    memset(&cp, 0, sizeof(cp))
    cp.dstMemoryType = dstType
    cp.srcMemoryType = srcType
    cp.WidthInBytes = 0
    cp.Height = 1
    cp.Depth = 1
    return cp


cdef ccuda.CUDA_MEMCPY2D_v2 memCopy2DInit(ccuda.CUmemorytype_enum dstType, ccuda.CUmemorytype_enum srcType) noexcept nogil:
    cdef ccuda.CUDA_MEMCPY2D_v2 cp
    memset(&cp, 0, sizeof(cp))
    cp.dstMemoryType = dstType
    cp.srcMemoryType = srcType
    cp.WidthInBytes = 0
    cp.Height = 1
    return cp


cdef cudaError_t bytesPerElement(size_t *bytes, int numberOfChannels, ccuda.CUarray_format format) except ?cudaErrorCallRequiresNewerDriver nogil:
    if format in (ccuda.CU_AD_FORMAT_FLOAT,
                  ccuda.CU_AD_FORMAT_UNSIGNED_INT32,
                  ccuda.CU_AD_FORMAT_SIGNED_INT32):
        bytes[0] = numberOfChannels * 4
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_HALF,
                    ccuda.CU_AD_FORMAT_SIGNED_INT16,
                    ccuda.CU_AD_FORMAT_UNSIGNED_INT16):
        bytes[0] = numberOfChannels * 2
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_SIGNED_INT8,
                    ccuda.CU_AD_FORMAT_UNSIGNED_INT8,
                    ccuda.CU_AD_FORMAT_NV12):
        bytes[0] = numberOfChannels
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_SNORM_INT8X1,
                    ccuda.CU_AD_FORMAT_UNORM_INT8X1):
        bytes[0] = 1
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_SNORM_INT8X2,
                    ccuda.CU_AD_FORMAT_UNORM_INT8X2,
                    ccuda.CU_AD_FORMAT_SNORM_INT16X1,
                    ccuda.CU_AD_FORMAT_UNORM_INT16X1):
        bytes[0] = 2
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_SNORM_INT8X4,
                    ccuda.CU_AD_FORMAT_UNORM_INT8X4,
                    ccuda.CU_AD_FORMAT_SNORM_INT16X2,
                    ccuda.CU_AD_FORMAT_UNORM_INT16X2):
        bytes[0] = 4
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_SNORM_INT16X4,
                    ccuda.CU_AD_FORMAT_UNORM_INT16X4):
        bytes[0] = 8
        return cudaSuccess
    elif format in (ccuda.CU_AD_FORMAT_BC2_UNORM,
                    ccuda.CU_AD_FORMAT_BC2_UNORM_SRGB,
                    ccuda.CU_AD_FORMAT_BC3_UNORM,
                    ccuda.CU_AD_FORMAT_BC3_UNORM_SRGB,
                    ccuda.CU_AD_FORMAT_BC5_UNORM,
                    ccuda.CU_AD_FORMAT_BC5_SNORM,
                    ccuda.CU_AD_FORMAT_BC6H_UF16,
                    ccuda.CU_AD_FORMAT_BC6H_SF16,
                    ccuda.CU_AD_FORMAT_BC7_UNORM,
                    ccuda.CU_AD_FORMAT_BC7_UNORM_SRGB):
        bytes[0] = 16
        return cudaSuccess
    return cudaErrorInvalidChannelDescriptor


cdef cudaError_t getChannelFormatDescFromDriverDesc(
    cudaChannelFormatDesc* pRuntimeDesc, size_t* pDepth, size_t* pHeight, size_t* pWidth,
    const ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2* pDriverDesc) except ?cudaErrorCallRequiresNewerDriver nogil:

    cdef int channel_size = 0
    if pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT8:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT32:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
        channel_size = 32
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SIGNED_INT8:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SIGNED_INT16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SIGNED_INT32:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSigned
        channel_size = 32
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_HALF:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindFloat
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_FLOAT:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindFloat
        channel_size = 32
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_NV12:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindNV12
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT8X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT8X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT8X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT8X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT8X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT8X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT16X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT16X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_UNORM_INT16X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT16X1:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT16X2:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_SNORM_INT16X4:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC1_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC1_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC2_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC2_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC3_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC3_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC4_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC4_SNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC5_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC5_SNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC6H_UF16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC6H_SF16:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H
        channel_size = 16
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC7_UNORM:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7
        channel_size = 8
    elif pDriverDesc[0].Format == ccuda.CU_AD_FORMAT_BC7_UNORM_SRGB:
        pRuntimeDesc[0].f = cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB
        channel_size = 8
    else:
        return cudaErrorInvalidChannelDescriptor

    # populate bits per channel
    pRuntimeDesc[0].x = 0
    pRuntimeDesc[0].y = 0
    pRuntimeDesc[0].z = 0
    pRuntimeDesc[0].w = 0

    if pDriverDesc[0].NumChannels >= 4:
        pRuntimeDesc[0].w = channel_size
    if pDriverDesc[0].NumChannels >= 3:
        pRuntimeDesc[0].z = channel_size
    if pDriverDesc[0].NumChannels >= 2:
        pRuntimeDesc[0].y = channel_size
    if pDriverDesc[0].NumChannels >= 1:
        pRuntimeDesc[0].x = channel_size

    if pDriverDesc[0].NumChannels not in (4, 3, 2, 1):
        return cudaErrorInvalidChannelDescriptor

    # populate dimensions
    if pDepth != NULL:
        pDepth[0]  = pDriverDesc[0].Depth
    if pHeight != NULL:
        pHeight[0] = pDriverDesc[0].Height
    if pWidth != NULL:
        pWidth[0]  = pDriverDesc[0].Width
    return cudaSuccess


cdef cudaError_t getArrayBlockExtent(cudaExtent *blockExtent, ccuda.CUarray_format format) except ?cudaErrorCallRequiresNewerDriver nogil:
    if format in (ccuda.CU_AD_FORMAT_FLOAT,
                  ccuda.CU_AD_FORMAT_UNSIGNED_INT32,
                  ccuda.CU_AD_FORMAT_SIGNED_INT32,
                  ccuda.CU_AD_FORMAT_HALF,
                  ccuda.CU_AD_FORMAT_SIGNED_INT16,
                  ccuda.CU_AD_FORMAT_UNSIGNED_INT16,
                  ccuda.CU_AD_FORMAT_SIGNED_INT8,
                  ccuda.CU_AD_FORMAT_UNSIGNED_INT8,
                  ccuda.CU_AD_FORMAT_NV12,
                  ccuda.CU_AD_FORMAT_SNORM_INT8X1,
                  ccuda.CU_AD_FORMAT_UNORM_INT8X1,
                  ccuda.CU_AD_FORMAT_SNORM_INT8X2,
                  ccuda.CU_AD_FORMAT_UNORM_INT8X2,
                  ccuda.CU_AD_FORMAT_SNORM_INT16X1,
                  ccuda.CU_AD_FORMAT_UNORM_INT16X1,
                  ccuda.CU_AD_FORMAT_SNORM_INT8X4,
                  ccuda.CU_AD_FORMAT_UNORM_INT8X4,
                  ccuda.CU_AD_FORMAT_SNORM_INT16X2,
                  ccuda.CU_AD_FORMAT_UNORM_INT16X2,
                  ccuda.CU_AD_FORMAT_SNORM_INT16X4,
                  ccuda.CU_AD_FORMAT_UNORM_INT16X4):
        blockExtent[0].width = 1
        blockExtent[0].height = 1
        blockExtent[0].depth = 1
    elif format in (ccuda.CU_AD_FORMAT_BC1_UNORM,
                    ccuda.CU_AD_FORMAT_BC1_UNORM_SRGB,
                    ccuda.CU_AD_FORMAT_BC4_UNORM,
                    ccuda.CU_AD_FORMAT_BC4_SNORM,
                    ccuda.CU_AD_FORMAT_BC2_UNORM,
                    ccuda.CU_AD_FORMAT_BC2_UNORM_SRGB,
                    ccuda.CU_AD_FORMAT_BC3_UNORM,
                    ccuda.CU_AD_FORMAT_BC3_UNORM_SRGB,
                    ccuda.CU_AD_FORMAT_BC5_UNORM,
                    ccuda.CU_AD_FORMAT_BC5_SNORM,
                    ccuda.CU_AD_FORMAT_BC6H_UF16,
                    ccuda.CU_AD_FORMAT_BC6H_SF16,
                    ccuda.CU_AD_FORMAT_BC7_UNORM,
                    ccuda.CU_AD_FORMAT_BC7_UNORM_SRGB):
        blockExtent[0].width = 4
        blockExtent[0].height = 4
        blockExtent[0].depth = 1
    else:
        return cudaErrorInvalidChannelDescriptor
    return cudaSuccess


cdef cudaError_t getLocalState(cudaArrayLocalState *state, cudaArray_const_t thisArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    memset(&arrayState, 0, sizeof(arrayState))
    arrayState.array = <ccuda.CUarray>thisArray

    cdef cudaExtent compBlockExtent
    compBlockExtent.width = 1
    compBlockExtent.height = 1
    compBlockExtent.depth = 1
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 driverDesc
    memset(&driverDesc, 0, sizeof(driverDesc))
    err = <cudaError_t>ccuda._cuArray3DGetDescriptor_v2(&driverDesc, <ccuda.CUarray>arrayState.array)
    if err != cudaSuccess:
        return err
    err = getChannelFormatDescFromDriverDesc(&arrayState.desc, &arrayState.depth, &arrayState.height, &arrayState.width, &driverDesc)
    if err != cudaSuccess:
        return err
    err = bytesPerElement(&arrayState.elementSize, driverDesc.NumChannels, driverDesc.Format)
    if err != cudaSuccess:
        return err
    err = getArrayBlockExtent(&compBlockExtent, driverDesc.Format)
    if err != cudaSuccess:
        return err
    arrayState.widthInBytes = <size_t>((arrayState.width + compBlockExtent.width - 1) / compBlockExtent.width) * arrayState.elementSize

    state[0] = arrayState
    return cudaSuccess


cdef cudaError_t copyFromHost2D(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t spitch, size_t width, size_t height, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef cudaArrayLocalState arrayState
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST)

    cp.dstArray      = arrayState.array
    cp.dstXInBytes   = wOffset
    cp.dstY          = hOffset

    cp.srcHost       = src
    cp.srcPitch      = spitch
    cp.srcXInBytes   = 0
    cp.srcY          = 0

    cp.Height        = height
    cp.WidthInBytes  = width

    err = driverMemcpy3D(&cp, stream, async)
    return err


cdef cudaError_t copyFromDevice2D(ccuda.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t srcOffset,
        size_t spitch, size_t width, size_t height, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef cudaArrayLocalState arrayState
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY, type)

    cp.dstArray      = arrayState.array
    cp.dstXInBytes   = wOffset
    cp.dstY          = hOffset

    cp.srcDevice     = <ccuda.CUdeviceptr_v2>src
    cp.srcPitch      = spitch
    cp.srcXInBytes   = srcOffset % spitch
    cp.srcY          = <size_t>(srcOffset / spitch)

    cp.Height        = height
    cp.WidthInBytes  = width

    err = driverMemcpy3D(&cp, stream, async)
    if err != cudaSuccess:
        return err

    return cudaSuccess


cdef cudaError_t copyToHost2D(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, char *dst, size_t dpitch, size_t width,
        size_t height, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)

    cp.dstHost       = dst
    cp.dstPitch      = dpitch
    cp.dstXInBytes   = 0
    cp.dstY          = 0

    cp.srcArray      = arrayState.array
    cp.srcXInBytes   = wOffset
    cp.srcY          = hOffset

    cp.Height        = height
    cp.WidthInBytes  = width

    err = driverMemcpy3D(&cp, stream, async)
    if err != cudaSuccess:
        return err

    return cudaSuccess


cdef cudaError_t copyToDevice2D(ccuda.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *dst, size_t dstOffset, size_t dpitch,
        size_t width, size_t height, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:

    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(type, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)

    cp.dstDevice     = <ccuda.CUdeviceptr_v2>dst
    cp.dstPitch      = dpitch
    cp.dstXInBytes   = dstOffset % dpitch
    cp.dstY          = <size_t>(dstOffset / dpitch)

    cp.srcArray      = arrayState.array
    cp.srcXInBytes   = wOffset
    cp.srcY          = hOffset

    cp.Height        = height
    cp.WidthInBytes  = width

    err = driverMemcpy3D(&cp, stream, async)
    if err != cudaSuccess:
        return err

    return cudaSuccess


cdef cudaError_t copyToArray2D(cudaArray_const_t thisArray, size_t hOffsetSrc, size_t wOffsetSrc, cudaArray_t dst,
        size_t hOffsetDst, size_t wOffsetDst, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)

    cp.dstArray      = <ccuda.CUarray>dst
    cp.dstXInBytes   = wOffsetDst
    cp.dstY          = hOffsetDst

    cp.srcArray      = arrayState.array
    cp.srcXInBytes   = wOffsetSrc
    cp.srcY          = hOffsetSrc

    cp.Height        = height
    cp.WidthInBytes  = width

    err = driverMemcpy3D(&cp, NULL, False)
    if err != cudaSuccess:
        return err

    return cudaSuccess


cdef cudaError_t copyToArray(cudaArray_const_t thisArray, size_t hOffsetSrc, size_t wOffsetSrc, cudaArray_t dst, size_t hOffsetDst,
        size_t wOffsetDst, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef void *tmp
    cdef cudaError_t err = cudaSuccess
    err = cudaMalloc(&tmp, count)
    if err != cudaSuccess:
        return err

    err = cudaMemcpyFromArray(tmp, thisArray, wOffsetSrc, hOffsetSrc, count, cudaMemcpyDeviceToDevice)
    if err != cudaSuccess:
        return err
    err = cudaMemcpyToArray(dst, wOffsetDst, hOffsetDst, tmp, count, cudaMemcpyDeviceToDevice)
    if err != cudaSuccess:
        return err
    err = cudaFree(tmp)
    if err != cudaSuccess:
        return err
    return cudaSuccess


cdef cudaError_t memcpyArrayToArray(cudaArray_t dst, size_t hOffsetDst, size_t wOffsetDst,
                                    cudaArray_const_t src, size_t hOffsetSrc, size_t wOffsetSrc,
                                    size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    if count == 0:
        return cudaSuccess
    if kind != cudaMemcpyDeviceToDevice and kind != cudaMemcpyDefault:
        return cudaErrorInvalidMemcpyDirection
    return copyToArray(src, hOffsetSrc, wOffsetSrc, dst, hOffsetDst, wOffsetDst, count)


cdef cudaError_t getChannelDesc(cudaArray_const_t thisArray, cudaChannelFormatDesc *outDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)  
    if err != cudaSuccess:
        return err
    outDesc[0] = arrayState.desc
    return cudaSuccess


cdef cudaError_t getFormat(cudaArray_const_t thisArray, int &numberOfChannels, ccuda.CUarray_format *format) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    return getDescInfo(&arrayState.desc, <int*>&numberOfChannels, <ccuda.CUarray_format*>format)


cdef cudaError_t getDriverResDescFromResDesc(ccuda.CUDA_RESOURCE_DESC *rdDst, const cudaResourceDesc *rdSrc,
                                             ccuda.CUDA_TEXTURE_DESC *tdDst, const cudaTextureDesc *tdSrc,
                                             ccuda.CUDA_RESOURCE_VIEW_DESC *rvdDst, const cudaResourceViewDesc *rvdSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef int i = 0
    cdef int numChannels = 0
    cdef ccuda.CUarray_format format
    cdef ccuda.CUarray hArray = NULL
    cdef cudaError_t err = cudaSuccess
    i = 0

    memset(rdDst, 0, sizeof(rdDst[0]))

    if rdSrc[0].resType == cudaResourceType.cudaResourceTypeArray:
        rdDst[0].resType          = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY
        rdDst[0].res.array.hArray = <ccuda.CUarray>rdSrc[0].res.array.array
        err = getFormat(rdSrc[0].res.array.array, numChannels, &format)
        if err != cudaSuccess:
            return err
    elif rdSrc[0].resType == cudaResourceType.cudaResourceTypeMipmappedArray:
        rdDst[0].resType                    = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
        rdDst[0].res.mipmap.hMipmappedArray = <ccuda.CUmipmappedArray>rdSrc[0].res.mipmap.mipmap
        err = <cudaError_t>ccuda._cuMipmappedArrayGetLevel(&hArray, rdDst[0].res.mipmap.hMipmappedArray, 0)
        if err != cudaSuccess:
            return err
        err = getFormat(<cudaArray_t>hArray, numChannels, &format)
        if err != cudaSuccess:
            return err
    elif rdSrc[0].resType == cudaResourceType.cudaResourceTypeLinear:
        rdDst[0].resType                = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR
        rdDst[0].res.linear.devPtr      = <ccuda.CUdeviceptr_v2>rdSrc[0].res.linear.devPtr
        rdDst[0].res.linear.sizeInBytes = rdSrc[0].res.linear.sizeInBytes
        err = getDescInfo(&rdSrc[0].res.linear.desc, <int*>&numChannels, <ccuda.CUarray_format*>&format)
        if err != cudaSuccess:
            return err
        rdDst[0].res.linear.format      = format
        rdDst[0].res.linear.numChannels = numChannels
    elif rdSrc[0].resType == cudaResourceType.cudaResourceTypePitch2D:
        rdDst[0].resType                  = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D
        rdDst[0].res.pitch2D.devPtr       = <ccuda.CUdeviceptr_v2>rdSrc[0].res.pitch2D.devPtr
        rdDst[0].res.pitch2D.pitchInBytes = rdSrc[0].res.pitch2D.pitchInBytes
        rdDst[0].res.pitch2D.width        = rdSrc[0].res.pitch2D.width
        rdDst[0].res.pitch2D.height       = rdSrc[0].res.pitch2D.height
        err = getDescInfo(&rdSrc[0].res.linear.desc, <int*>&numChannels, <ccuda.CUarray_format*>&format)
        if err != cudaSuccess:
            return err
        rdDst[0].res.pitch2D.format       = format
        rdDst[0].res.pitch2D.numChannels  = numChannels
    else:
        return cudaErrorInvalidValue


    rdDst[0].flags = 0

    if tdDst and tdSrc:
        memset(tdDst, 0, sizeof(tdDst[0]))

        while (i < 3):
            tdDst[0].addressMode[i] = <ccuda.CUaddress_mode>tdSrc[0].addressMode[i]
            i += 1

        tdDst[0].filterMode          = <ccuda.CUfilter_mode>tdSrc[0].filterMode
        tdDst[0].mipmapFilterMode    = <ccuda.CUfilter_mode>tdSrc[0].mipmapFilterMode
        tdDst[0].mipmapLevelBias     = tdSrc[0].mipmapLevelBias
        tdDst[0].minMipmapLevelClamp = tdSrc[0].minMipmapLevelClamp
        tdDst[0].maxMipmapLevelClamp = tdSrc[0].maxMipmapLevelClamp
        tdDst[0].maxAnisotropy       = tdSrc[0].maxAnisotropy
        i = 0
        while (i < 4):
            tdDst[0].borderColor[i] = tdSrc[0].borderColor[i]
            i += 1

        if tdSrc[0].sRGB:
            tdDst[0].flags |= ccuda.CU_TRSF_SRGB
        else:
            tdDst[0].flags |= 0

        if tdSrc[0].normalizedCoords:
            tdDst[0].flags |= ccuda.CU_TRSF_NORMALIZED_COORDINATES
        else:
            tdDst[0].flags |= 0

        if tdSrc[0].disableTrilinearOptimization:
            tdDst[0].flags |= ccuda.CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION
        else:
            tdDst[0].flags |= 0

        if tdSrc[0].seamlessCubemap:
            tdDst[0].flags |= ccuda.CU_TRSF_SEAMLESS_CUBEMAP
        else:
            tdDst[0].flags |= 0

        if format in (ccuda.CU_AD_FORMAT_SNORM_INT8X1,
                      ccuda.CU_AD_FORMAT_SNORM_INT8X2,
                      ccuda.CU_AD_FORMAT_SNORM_INT8X4,
                      ccuda.CU_AD_FORMAT_UNORM_INT8X1,
                      ccuda.CU_AD_FORMAT_UNORM_INT8X2,
                      ccuda.CU_AD_FORMAT_UNORM_INT8X4,
                      ccuda.CU_AD_FORMAT_SNORM_INT16X1,
                      ccuda.CU_AD_FORMAT_SNORM_INT16X2,
                      ccuda.CU_AD_FORMAT_SNORM_INT16X4,
                      ccuda.CU_AD_FORMAT_UNORM_INT16X1,
                      ccuda.CU_AD_FORMAT_UNORM_INT16X2,
                      ccuda.CU_AD_FORMAT_UNORM_INT16X4,
                      ccuda.CU_AD_FORMAT_BC1_UNORM,
                      ccuda.CU_AD_FORMAT_BC1_UNORM_SRGB,
                      ccuda.CU_AD_FORMAT_BC2_UNORM,
                      ccuda.CU_AD_FORMAT_BC2_UNORM_SRGB,
                      ccuda.CU_AD_FORMAT_BC3_UNORM,
                      ccuda.CU_AD_FORMAT_BC3_UNORM_SRGB,
                      ccuda.CU_AD_FORMAT_BC4_UNORM,
                      ccuda.CU_AD_FORMAT_BC4_SNORM,
                      ccuda.CU_AD_FORMAT_BC5_UNORM,
                      ccuda.CU_AD_FORMAT_BC5_SNORM,
                      ccuda.CU_AD_FORMAT_BC7_UNORM,
                      ccuda.CU_AD_FORMAT_BC7_UNORM_SRGB):
            if tdSrc[0].readMode != cudaTextureReadMode.cudaReadModeNormalizedFloat:
                return cudaErrorInvalidNormSetting
        elif format in (ccuda.CU_AD_FORMAT_SIGNED_INT8,
                        ccuda.CU_AD_FORMAT_SIGNED_INT16,
                        ccuda.CU_AD_FORMAT_UNSIGNED_INT8,
                        ccuda.CU_AD_FORMAT_UNSIGNED_INT16):
            if tdSrc[0].readMode == cudaReadModeElementType:
                if tdSrc[0].filterMode == cudaTextureFilterMode.cudaFilterModeLinear:
                    return cudaErrorInvalidFilterSetting
                tdDst[0].flags |= ccuda.CU_TRSF_READ_AS_INTEGER
        elif format == ccuda.CU_AD_FORMAT_NV12:
            return cudaErrorInvalidValue
        elif format == ccuda.CU_AD_FORMAT_SIGNED_INT32 or format == ccuda.CU_AD_FORMAT_UNSIGNED_INT32:
            if tdSrc[0].filterMode == cudaTextureFilterMode.cudaFilterModeLinear:
                return cudaErrorInvalidFilterSetting
            if tdSrc[0].readMode == cudaTextureReadMode.cudaReadModeNormalizedFloat:
                return cudaErrorInvalidNormSetting
        else:
            if tdSrc[0].readMode == cudaTextureReadMode.cudaReadModeNormalizedFloat:
                return cudaErrorInvalidNormSetting

    if rvdDst and rvdSrc:
        memset(rvdDst, 0, sizeof(rvdDst[0]))

        rvdDst[0].format           = <ccuda.CUresourceViewFormat>rvdSrc[0].format
        rvdDst[0].width            = rvdSrc[0].width
        rvdDst[0].height           = rvdSrc[0].height
        rvdDst[0].depth            = rvdSrc[0].depth
        rvdDst[0].firstMipmapLevel = rvdSrc[0].firstMipmapLevel
        rvdDst[0].lastMipmapLevel  = rvdSrc[0].lastMipmapLevel
        rvdDst[0].firstLayer       = rvdSrc[0].firstLayer
        rvdDst[0].lastLayer        = rvdSrc[0].lastLayer

    return cudaSuccess


cdef cudaError_t getResDescFromDriverResDesc(cudaResourceDesc *rdDst, const ccuda.CUDA_RESOURCE_DESC *rdSrc,
                                             cudaTextureDesc *tdDst, const ccuda.CUDA_TEXTURE_DESC *tdSrc,
                                             cudaResourceViewDesc *rvdDst, const ccuda.CUDA_RESOURCE_VIEW_DESC *rvdSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef int i = 0
    cdef int numChannels = 0
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 ad
    cdef ccuda.CUarray hArray

    memset(rdDst, 0, sizeof(rdDst[0]))
    memset(&ad, 0, sizeof(ad))
    memset(&hArray, 0, sizeof(hArray))

    if rdSrc[0].resType == ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY:
        rdDst[0].resType         = cudaResourceType.cudaResourceTypeArray
        rdDst[0].res.array.array = <cudaArray_t>rdSrc[0].res.array.hArray
        err = getFormat(rdDst[0].res.array.array, numChannels, &ad.Format)
        if err != cudaSuccess:
            return err
    elif rdSrc[0].resType == ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
        rdDst[0].resType = cudaResourceType.cudaResourceTypeMipmappedArray
        rdDst[0].res.mipmap.mipmap = <cudaMipmappedArray_t>rdSrc[0].res.mipmap.hMipmappedArray
        err = <cudaError_t>ccuda._cuMipmappedArrayGetLevel(&hArray, rdSrc[0].res.mipmap.hMipmappedArray, 0)
        if err != cudaSuccess:
            return err
        err = getFormat(<cudaArray_t>hArray, numChannels, &ad.Format)
        if err != cudaSuccess:
            return err
    elif rdSrc[0].resType == ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR:
        rdDst[0].resType                = cudaResourceType.cudaResourceTypeLinear
        rdDst[0].res.linear.devPtr      = <void *>rdSrc[0].res.linear.devPtr
        rdDst[0].res.linear.sizeInBytes = rdSrc[0].res.linear.sizeInBytes
        ad.Format      = rdSrc[0].res.linear.format
        ad.NumChannels = rdSrc[0].res.linear.numChannels
        err = getChannelFormatDescFromDriverDesc(&rdDst[0].res.linear.desc,
                                                 NULL, NULL, NULL,
                                                 &ad)
        if err != cudaSuccess:
            return err
    elif rdSrc[0].resType == ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D:
        rdDst[0].resType                  = cudaResourceType.cudaResourceTypePitch2D
        rdDst[0].res.pitch2D.devPtr       = <void *>rdSrc[0].res.pitch2D.devPtr
        rdDst[0].res.pitch2D.pitchInBytes = rdSrc[0].res.pitch2D.pitchInBytes
        rdDst[0].res.pitch2D.width        = rdSrc[0].res.pitch2D.width
        rdDst[0].res.pitch2D.height       = rdSrc[0].res.pitch2D.height
        ad.Format      = rdSrc[0].res.linear.format
        ad.NumChannels = rdSrc[0].res.linear.numChannels
        err = getChannelFormatDescFromDriverDesc(&rdDst[0].res.linear.desc,
                                                NULL, NULL, NULL,
                                                &ad)
        if err != cudaSuccess:
            return err
    else:
        return cudaErrorInvalidValue

    if tdDst and tdSrc:
        memset(tdDst, 0, sizeof(tdDst[0]))
        i = 0
        while i < 3:
            tdDst[0].addressMode[i] = <cudaTextureAddressMode>tdSrc[0].addressMode[i]
            i += 1

        tdDst[0].filterMode          = <cudaTextureFilterMode>tdSrc[0].filterMode
        tdDst[0].mipmapFilterMode    = <cudaTextureFilterMode>tdSrc[0].mipmapFilterMode
        tdDst[0].mipmapLevelBias     = tdSrc[0].mipmapLevelBias
        tdDst[0].minMipmapLevelClamp = tdSrc[0].minMipmapLevelClamp
        tdDst[0].maxMipmapLevelClamp = tdSrc[0].maxMipmapLevelClamp
        tdDst[0].maxAnisotropy       = tdSrc[0].maxAnisotropy
        i = 0
        while i < 4:
            tdDst[0].borderColor[i] = tdSrc[0].borderColor[i]
            i += 1

        if tdSrc[0].flags & ccuda.CU_TRSF_SRGB:
            tdDst[0].sRGB                         = 1
        else:
            tdDst[0].sRGB                         = 0

        if tdSrc[0].flags & ccuda.CU_TRSF_NORMALIZED_COORDINATES:
            tdDst[0].normalizedCoords             = 1
        else:
            tdDst[0].normalizedCoords             = 0

        if tdSrc[0].flags & ccuda.CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION:
            tdDst[0].disableTrilinearOptimization = 1
        else:
            tdDst[0].disableTrilinearOptimization = 0

        if tdSrc[0].flags & ccuda.CU_TRSF_SEAMLESS_CUBEMAP:
            tdDst[0].seamlessCubemap |= 1
        else:
            tdDst[0].seamlessCubemap |= 0

        if ad.Format in (ccuda.CU_AD_FORMAT_SNORM_INT8X1,
                         ccuda.CU_AD_FORMAT_SNORM_INT8X2,
                         ccuda.CU_AD_FORMAT_SNORM_INT8X4,
                         ccuda.CU_AD_FORMAT_UNORM_INT8X1,
                         ccuda.CU_AD_FORMAT_UNORM_INT8X2,
                         ccuda.CU_AD_FORMAT_UNORM_INT8X4,
                         ccuda.CU_AD_FORMAT_SNORM_INT16X1,
                         ccuda.CU_AD_FORMAT_SNORM_INT16X2,
                         ccuda.CU_AD_FORMAT_SNORM_INT16X4,
                         ccuda.CU_AD_FORMAT_UNORM_INT16X1,
                         ccuda.CU_AD_FORMAT_UNORM_INT16X2,
                         ccuda.CU_AD_FORMAT_UNORM_INT16X4,
                         ccuda.CU_AD_FORMAT_BC1_UNORM,
                         ccuda.CU_AD_FORMAT_BC1_UNORM_SRGB,
                         ccuda.CU_AD_FORMAT_BC2_UNORM,
                         ccuda.CU_AD_FORMAT_BC2_UNORM_SRGB,
                         ccuda.CU_AD_FORMAT_BC3_UNORM,
                         ccuda.CU_AD_FORMAT_BC3_UNORM_SRGB,
                         ccuda.CU_AD_FORMAT_BC4_UNORM,
                         ccuda.CU_AD_FORMAT_BC4_SNORM,
                         ccuda.CU_AD_FORMAT_BC5_UNORM,
                         ccuda.CU_AD_FORMAT_BC5_SNORM,
                         ccuda.CU_AD_FORMAT_BC7_UNORM,
                         ccuda.CU_AD_FORMAT_BC7_UNORM_SRGB):
            tdDst[0].readMode = cudaTextureReadMode.cudaReadModeNormalizedFloat
        elif ad.Format in (ccuda.CU_AD_FORMAT_SIGNED_INT8,
                           ccuda.CU_AD_FORMAT_SIGNED_INT16,
                           ccuda.CU_AD_FORMAT_UNSIGNED_INT8,
                           ccuda.CU_AD_FORMAT_UNSIGNED_INT16):
            with gil:
                if (tdSrc[0].flags & ccuda.CU_TRSF_READ_AS_INTEGER):
                    tdDst[0].readMode = cudaTextureReadMode.cudaReadModeElementType
                else:
                    tdDst[0].readMode = cudaTextureReadMode.cudaReadModeNormalizedFloat
        else:
            tdDst[0].readMode = cudaTextureReadMode.cudaReadModeElementType

    if rvdDst and rvdSrc:
        memset(rvdDst, 0, sizeof(rvdDst[0]))

        rvdDst[0].format           = <cudaResourceViewFormat>rvdSrc[0].format
        rvdDst[0].width            = rvdSrc[0].width
        rvdDst[0].height           = rvdSrc[0].height
        rvdDst[0].depth            = rvdSrc[0].depth
        rvdDst[0].firstMipmapLevel = rvdSrc[0].firstMipmapLevel
        rvdDst[0].lastMipmapLevel  = rvdSrc[0].lastMipmapLevel
        rvdDst[0].firstLayer       = rvdSrc[0].firstLayer
        rvdDst[0].lastLayer        = rvdSrc[0].lastLayer

    return cudaSuccess


cdef cudaError_t memsetPtr(char *mem, int c, size_t count, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if count == 0:
        return cudaSuccess

    if not async:
        return <cudaError_t>ccuda._cuMemsetD8_v2(<ccuda.CUdeviceptr_v2>mem, <unsigned char>c, count)
    else:
        return <cudaError_t>ccuda._cuMemsetD8Async(<ccuda.CUdeviceptr_v2>mem, <unsigned char>c, count, sid)


cdef cudaError_t memset2DPtr(char *mem, size_t pitch, int c, size_t width, size_t height, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if width == 0 or height == 0:
        return cudaSuccess

    if not async:
        return <cudaError_t>ccuda._cuMemsetD2D8_v2(<ccuda.CUdeviceptr_v2>mem, pitch, <unsigned char>c, width, height)
    else:
        return <cudaError_t>ccuda._cuMemsetD2D8Async(<ccuda.CUdeviceptr_v2>mem, pitch, <unsigned char>c, width, height, sid)


cdef cudaError_t copyFromHost(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t count, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef size_t copied = 0
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST)

    if (wOffset > 0) and (count >= arrayState.widthInBytes - wOffset):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcHost       = src
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = 0
        cp.srcY          = 0

        cp.Height        = 1
        cp.WidthInBytes  = arrayState.widthInBytes - wOffset

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied >= arrayState.widthInBytes):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcHost       = src + copied
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = 0
        cp.srcY          = 0

        cp.Height        = <size_t>((count - copied) / arrayState.widthInBytes)
        cp.WidthInBytes  = arrayState.widthInBytes

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied > 0):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcHost       = src + copied
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = 0
        cp.srcY          = 0

        cp.Height        = 1
        cp.WidthInBytes  = count - copied

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    return cudaSuccess


cdef cudaError_t copyFromDevice(ccuda.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t srcOffset, size_t count, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef size_t copied = 0
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY, type)

    if (wOffset > 0) and (count >= arrayState.widthInBytes - wOffset):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcDevice     = <ccuda.CUdeviceptr_v2>src
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = srcOffset
        cp.srcY          = 0

        cp.Height        = 1
        cp.WidthInBytes  = arrayState.widthInBytes - wOffset

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied >= arrayState.widthInBytes):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcDevice     = <ccuda.CUdeviceptr_v2>(src + copied)
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = srcOffset
        cp.srcY          = 0

        cp.Height        = <size_t>((count - copied) / arrayState.widthInBytes)
        cp.WidthInBytes  = arrayState.widthInBytes

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0


        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied > 0):
        cp.dstArray      = arrayState.array
        cp.dstXInBytes   = wOffset
        cp.dstY          = hOffset

        cp.srcDevice     = <ccuda.CUdeviceptr_v2>(src + copied)
        cp.srcPitch      = arrayState.widthInBytes
        cp.srcXInBytes   = srcOffset
        cp.srcY          = 0

        cp.Height        = 1
        cp.WidthInBytes  = count - copied

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    return cudaSuccess


cdef cudaError_t copyToHost(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, char *dst, size_t count, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef size_t copied = 0
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)

    if (wOffset > 0) and (count >= arrayState.widthInBytes - wOffset):
        cp.dstHost       = dst
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = 0
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = 1
        cp.WidthInBytes  = arrayState.widthInBytes - wOffset

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied >= arrayState.widthInBytes):
        cp.dstHost       = dst + copied
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = 0
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = <size_t>((count - copied) / arrayState.widthInBytes)
        cp.WidthInBytes  = arrayState.widthInBytes

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied > 0):
        cp.dstHost       = dst + copied
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = 0
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = 1
        cp.WidthInBytes  = count - copied

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    return cudaSuccess


cdef cudaError_t driverMemcpy3DPeer(ccuda.CUDA_MEMCPY3D_PEER *cp, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if async:
        return <cudaError_t>ccuda._cuMemcpy3DPeerAsync(cp, stream)
    else:
        return <cudaError_t>ccuda._cuMemcpy3DPeer(cp)


cdef cudaError_t driverMemcpy3D(ccuda.CUDA_MEMCPY3D_v2 *cp, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if async:
        return <cudaError_t>ccuda._cuMemcpy3DAsync_v2(cp, stream)
    else:
        return <cudaError_t>ccuda._cuMemcpy3D_v2(cp)


cdef cudaError_t memcpy3D(const cudaMemcpy3DParms *p, bool peer, int srcDevice, int dstDevice, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef ccuda.CUDA_MEMCPY3D_v2 cd
    cdef ccuda.CUDA_MEMCPY3D_PEER cdPeer
    cdef cudaPythonDevice *srcDev
    cdef cudaPythonDevice *dstDev

    memset(&cdPeer, 0, sizeof(cdPeer))

    cdef cudaError_t err = toDriverMemCopy3DParams(p, &cd)
    if err != cudaSuccess:
        return err

    # Execute the copy
    if peer:
        srcDev = m_global.getDevice(srcDevice)
        dstDev = m_global.getDevice(dstDevice)
        if srcDev == NULL or dstDev == NULL:
            return cudaErrorInvalidDevice
        cdPeer.srcXInBytes = cd.srcXInBytes
        cdPeer.srcY = cd.srcY
        cdPeer.srcZ = cd.srcZ
        cdPeer.srcLOD = cd.srcLOD
        cdPeer.srcMemoryType = cd.srcMemoryType
        cdPeer.srcHost = cd.srcHost
        cdPeer.srcDevice = cd.srcDevice
        cdPeer.srcArray = cd.srcArray
        cdPeer.srcContext = srcDev.primaryContext
        cdPeer.srcPitch = cd.srcPitch
        cdPeer.srcHeight = cd.srcHeight
        cdPeer.dstXInBytes = cd.dstXInBytes
        cdPeer.dstY = cd.dstY
        cdPeer.dstZ = cd.dstZ
        cdPeer.dstLOD = cd.dstLOD
        cdPeer.dstMemoryType = cd.dstMemoryType
        cdPeer.dstHost = cd.dstHost
        cdPeer.dstDevice = cd.dstDevice
        cdPeer.dstArray = cd.dstArray
        cdPeer.dstContext = dstDev.primaryContext
        cdPeer.dstPitch = cd.dstPitch
        cdPeer.dstHeight = cd.dstHeight
        cdPeer.WidthInBytes = cd.WidthInBytes
        cdPeer.Height = cd.Height
        cdPeer.Depth = cd.Depth
        err = driverMemcpy3DPeer(&cdPeer, sid, async)
    else:
        err = driverMemcpy3D(&cd, sid, async)
    return err


cdef cudaError_t copyToDevice(ccuda.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *dst, size_t dstOffset, size_t count, ccuda.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaArrayLocalState arrayState
    cdef cudaError_t err = cudaSuccess
    memset(&arrayState, 0, sizeof(arrayState))
    err = getLocalState(&arrayState, thisArray)
    if err != cudaSuccess:
        return err
    cdef size_t copied = 0
    cdef ccuda.CUDA_MEMCPY3D_v2 cp = memCopy3DInit(type, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)

    if (wOffset > 0) and (count >= arrayState.widthInBytes - wOffset):
        cp.dstDevice     = <ccuda.CUdeviceptr_v2>dst
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = dstOffset
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = 1
        cp.WidthInBytes  = arrayState.widthInBytes - wOffset

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied >= arrayState.widthInBytes):
        cp.dstDevice     = <ccuda.CUdeviceptr_v2>(dst + copied)
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = dstOffset
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = <size_t>((count - copied) / arrayState.widthInBytes)
        cp.WidthInBytes  = arrayState.widthInBytes

        copied  += cp.Height * cp.WidthInBytes
        hOffset += cp.Height
        wOffset  = 0

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    if (count - copied > 0):
        cp.dstDevice     = <ccuda.CUdeviceptr_v2>(dst + copied)
        cp.dstPitch      = arrayState.widthInBytes
        cp.dstXInBytes   = dstOffset
        cp.dstY          = 0

        cp.srcArray      = arrayState.array
        cp.srcXInBytes   = wOffset
        cp.srcY          = hOffset

        cp.Height        = 1
        cp.WidthInBytes  = count - copied

        err = driverMemcpy3D(&cp, stream, async)
        if err != cudaSuccess:
            return err

    return cudaSuccess


cdef cudaError_t copy1DConvertTo3DParams(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaMemcpy3DParms *p) except ?cudaErrorCallRequiresNewerDriver nogil:
    memset(p, 0, sizeof(cudaMemcpy3DParms))
    p[0].extent.width = count
    p[0].extent.height = 1
    p[0].extent.depth = 1
    p[0].dstPtr.ptr = dst
    p[0].srcPtr.ptr = <void *>src
    p[0].kind = kind


cdef void toDriverMemsetNodeParams(const cudaMemsetParams *pRuntimeParams, ccuda.CUDA_MEMSET_NODE_PARAMS *pDriverParams) noexcept nogil:
    pDriverParams[0].dst = <ccuda.CUdeviceptr_v2>pRuntimeParams[0].dst
    pDriverParams[0].pitch = pRuntimeParams[0].pitch
    pDriverParams[0].value = pRuntimeParams[0].value
    pDriverParams[0].elementSize = pRuntimeParams[0].elementSize
    pDriverParams[0].width = pRuntimeParams[0].width
    pDriverParams[0].height = pRuntimeParams[0].height


cdef cudaError_t getElementSize(size_t *elementSize, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR driverDesc
    cdef cudaError_t err = cudaSuccess
    err = <cudaError_t>ccuda._cuArray3DGetDescriptor_v2(&driverDesc, <ccuda.CUarray>array)
    if err != cudaSuccess:
        return err
    if (driverDesc.Format == ccuda.CU_AD_FORMAT_FLOAT or
        driverDesc.Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT32 or
        driverDesc.Format == ccuda.CU_AD_FORMAT_SIGNED_INT32):
        elementSize[0] = driverDesc.NumChannels * 4
        return cudaSuccess
    elif (driverDesc.Format == ccuda.CU_AD_FORMAT_HALF or
          driverDesc.Format == ccuda.CU_AD_FORMAT_SIGNED_INT16 or
          driverDesc.Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT16):
        elementSize[0] = driverDesc.NumChannels * 2
        return cudaSuccess
    elif (driverDesc.Format == ccuda.CU_AD_FORMAT_SIGNED_INT8 or
          driverDesc.Format == ccuda.CU_AD_FORMAT_UNSIGNED_INT8 or
          driverDesc.Format == ccuda.CU_AD_FORMAT_NV12):
        elementSize[0] = driverDesc.NumChannels
        return cudaSuccess
    return cudaErrorInvalidChannelDescriptor


cdef cudaError_t toDriverMemCopy3DParams(const cudaMemcpy3DParms *p, ccuda.CUDA_MEMCPY3D *cd) except ?cudaErrorCallRequiresNewerDriver nogil:
    memset(cd, 0, sizeof(ccuda.CUDA_MEMCPY3D))
    cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
    cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
    cd[0].WidthInBytes = 0
    cd[0].Height = 1
    cd[0].Depth = 1
    cdef size_t srcElementSize = 0
    cdef size_t dstElementSize = 0
    cdef cudaError_t err = cudaSuccess

    cdef cudaExtent srcBlockExtent
    cdef cudaExtent dstBlockExtent
    cdef cudaExtent copyBlockExtent
    cdef ccuda.CUarray_format srcFmt
    cdef ccuda.CUarray_format dstFmt
    cdef int numChannels = 0
    srcBlockExtent.width = srcBlockExtent.height = srcBlockExtent.depth = 1
    dstBlockExtent.width = dstBlockExtent.height = dstBlockExtent.depth = 1
    copyBlockExtent.width = copyBlockExtent.height = copyBlockExtent.depth = 1

    if p[0].extent.width == 0 or p[0].extent.height == 0 or p[0].extent.depth == 0:
        return cudaSuccess

    if p[0].kind == cudaMemcpyHostToHost:
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST
    elif p[0].kind == cudaMemcpyHostToDevice:
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
    elif p[0].kind == cudaMemcpyDeviceToHost:
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST
    elif p[0].kind == cudaMemcpyDeviceToDevice:
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
    elif p[0].kind == cudaMemcpyDefault:
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED
    else:
        return cudaErrorInvalidMemcpyDirection

    if p[0].srcArray:
        err = getFormat(p[0].srcArray, numChannels, &srcFmt)
        if err != cudaSuccess:
            return err
        err = getArrayBlockExtent(&srcBlockExtent, srcFmt)
        if err != cudaSuccess:
            return err
        copyBlockExtent = srcBlockExtent
    if p[0].dstArray:
        err = getFormat(p[0].dstArray, numChannels, &dstFmt)
        if err != cudaSuccess:
            return err
        err = getArrayBlockExtent(&dstBlockExtent, dstFmt)
        if err != cudaSuccess:
            return err
        if not p[0].srcArray:
            copyBlockExtent = dstBlockExtent

    if p[0].srcArray:
        if NULL != p[0].srcPtr.ptr or ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST == cd[0].srcMemoryType:
            return cudaErrorInvalidValue
        cd[0].srcMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY
        cd[0].srcArray = <ccuda.CUarray>p[0].srcArray
        err = getElementSize(&srcElementSize, p[0].srcArray)
        if err != cudaSuccess:
            return err
    else:
        if NULL == p[0].srcPtr.ptr:
            return cudaErrorInvalidValue
        if (p[0].extent.height > 1 or p[0].extent.depth > 1) and (p[0].extent.width > p[0].srcPtr.pitch):
            return cudaErrorInvalidPitchValue
        if p[0].extent.depth > 1:
            adjustedSrcHeight = p[0].srcPtr.ysize * copyBlockExtent.height
            if p[0].extent.height > adjustedSrcHeight:
                return cudaErrorInvalidPitchValue

        if ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST == cd[0].srcMemoryType:
            cd[0].srcHost = p[0].srcPtr.ptr
        else:
            cd[0].srcDevice = <ccuda.CUdeviceptr_v2>(p[0].srcPtr.ptr)
        cd[0].srcPitch = p[0].srcPtr.pitch
        cd[0].srcHeight = p[0].srcPtr.ysize

    if p[0].dstArray:
        if NULL != p[0].dstPtr.ptr:
            return cudaErrorInvalidValue
        cd[0].dstMemoryType = ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY
        cd[0].dstArray = <ccuda.CUarray>p[0].dstArray
        err = getElementSize(&dstElementSize, p[0].dstArray)
        if err != cudaSuccess:
            return err
    else:
        if NULL == p[0].dstPtr.ptr:
            return cudaErrorInvalidValue
        if (p[0].extent.height > 1 or p[0].extent.depth > 1) and (p[0].extent.width > p[0].dstPtr.pitch):
            return cudaErrorInvalidPitchValue
        if p[0].extent.depth > 1:
            adjustedDstHeight = p[0].dstPtr.ysize * copyBlockExtent.height
            if p[0].extent.height > adjustedDstHeight:
                return cudaErrorInvalidPitchValue

        if ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST == cd[0].dstMemoryType:
            cd[0].dstHost = p[0].dstPtr.ptr
        else:
            cd[0].dstDevice = <ccuda.CUdeviceptr_v2>(p[0].dstPtr.ptr)
        cd[0].dstPitch = p[0].dstPtr.pitch
        cd[0].dstHeight = p[0].dstPtr.ysize

    if srcElementSize and dstElementSize and srcElementSize != dstElementSize:
        return cudaErrorInvalidValue

    cdef size_t elementSize = sizeof(char)
    if srcElementSize:
        elementSize = srcElementSize
    if dstElementSize:
        elementSize = dstElementSize
    srcElementSize = elementSize
    dstElementSize = elementSize

    # Determine the extent of the transfer
    cd[0].WidthInBytes = <size_t>((p[0].extent.width + copyBlockExtent.width - 1) / copyBlockExtent.width)  * elementSize
    cd[0].Height       = <size_t>((p[0].extent.height + copyBlockExtent.height - 1) / copyBlockExtent.height)
    cd[0].Depth        = p[0].extent.depth

    # Populate bloated src copy origin
    cd[0].srcXInBytes  = <size_t>(p[0].srcPos.x / srcBlockExtent.width) * elementSize
    cd[0].srcY         = <size_t>(p[0].srcPos.y / srcBlockExtent.height)
    cd[0].srcZ         = p[0].srcPos.z

    # Populate bloated dst copy origin
    cd[0].dstXInBytes  = <size_t>(p[0].dstPos.x / dstBlockExtent.width) * elementSize
    cd[0].dstY         = <size_t>(p[0].dstPos.y / dstBlockExtent.height)
    cd[0].dstZ         = p[0].dstPos.z

    return cudaSuccess


cdef cudaError_t mallocArray(cudaArray_t *arrayPtr, const cudaChannelFormatDesc *desc,
        size_t depth, size_t height, size_t width, int corr2D, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    if arrayPtr == NULL:
        return cudaErrorInvalidValue

    cdef ccuda.CUarray array = NULL
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 ad
    cdef cudaError_t err = cudaSuccess
    arrayPtr[0] = NULL
    if (((width == 0)) or
        ((height == 0) and (depth != 0) and not (flags & cudaArrayLayered)) or
        ((flags & cudaArrayLayered) and (depth == 0)) or
        ((flags & cudaArrayCubemap) and not (flags & cudaArrayLayered) and ((width != height) or (depth != 6))) or
        ((flags & cudaArrayLayered) and (flags & cudaArrayCubemap) and ((width != height) or (depth % 6 != 0)))):
        return cudaErrorInvalidValue
    else:
        memset(&ad, 0, sizeof(ad))
        err = getDescInfo(desc, <int*>&ad.NumChannels, <ccuda.CUarray_format*>&ad.Format)
        if err != cudaSuccess:
            return err
        ad.Height = <unsigned int>height
        ad.Width  = <unsigned int>width
        ad.Depth  = <unsigned int>(depth - corr2D)
        ad.Flags  = flags
        err = <cudaError_t>ccuda._cuArray3DCreate_v2(&array, &ad)
        if err != cudaSuccess:
            return err

        arrayPtr[0] = <cudaArray_t>array
    return cudaSuccess


cdef cudaError_t memcpy2DToArray(cudaArray_t dst, size_t hOffset, size_t wOffset, const char *src,
                                 size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
                                 cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if width == 0 or height == 0:
        return cudaSuccess
    if height > 1 and width > spitch:
        return cudaErrorInvalidPitchValue

    cdef cudaError_t err = cudaSuccess
    if kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
       err = copyFromHost2D(dst, hOffset, wOffset, src, spitch, width, height, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
       err = copyFromDevice2D(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, dst, hOffset, wOffset, src, 0, spitch, width, height, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
       err = copyFromDevice2D(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED, dst, hOffset, wOffset, src, 0, spitch, width, height, sid, async)
    else:
        return cudaErrorInvalidMemcpyDirection
    return err


cdef cudaError_t memcpy2DPtr(char *dst, size_t dpitch, const char *src, size_t spitch, size_t width,
                             size_t height, cudaMemcpyKind kind,
                             cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if width == 0 or height == 0:
        return cudaSuccess
    if height > 1 and width > dpitch:
        return cudaErrorInvalidPitchValue
    if height > 1 and width > spitch:
        return cudaErrorInvalidPitchValue

    cdef cudaError_t err = cudaSuccess
    cdef ccuda.CUDA_MEMCPY2D_v2 cp
    memset(&cp, 0, sizeof(cp))

    if kind == cudaMemcpyKind.cudaMemcpyHostToHost:
        cp = memCopy2DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST)
        cp.dstHost = dst
        cp.srcHost = src
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        cp = memCopy2DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE)
        cp.dstHost = dst
        cp.srcDevice = <ccuda.CUdeviceptr_v2>src
    elif kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
        cp = memCopy2DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST)
        cp.dstDevice = <ccuda.CUdeviceptr_v2>dst
        cp.srcHost = src
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        cp = memCopy2DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE)
        cp.dstDevice = <ccuda.CUdeviceptr_v2>dst
        cp.srcDevice = <ccuda.CUdeviceptr_v2>src
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        cp = memCopy2DInit(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED, ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED)
        cp.dstDevice = <ccuda.CUdeviceptr_v2>dst
        cp.srcDevice = <ccuda.CUdeviceptr_v2>src
    else:
        err = cudaErrorInvalidMemcpyDirection

    if err != cudaSuccess:
        return err

    cp.dstPitch      = dpitch
    cp.srcPitch      = spitch
    cp.WidthInBytes  = width
    cp.Height        = height

    if async:
        err = <cudaError_t>ccuda._cuMemcpy2DAsync_v2(&cp, sid)
    else:
        err = <cudaError_t>ccuda._cuMemcpy2DUnaligned_v2(&cp)
    return err


cdef cudaError_t memcpyDispatch(void *dst, const void *src, size_t size, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    if size == 0:
        return cudaSuccess

    cdef cudaError_t err = cudaSuccess
    if kind == cudaMemcpyKind.cudaMemcpyHostToHost:
        return memcpy2DPtr(<char*>dst, size, <const char*>src, size, size, 1, kind, NULL, 0)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        err = <cudaError_t>ccuda._cuMemcpyDtoH_v2(dst, <ccuda.CUdeviceptr_v2>src, size)
    elif kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
        err = <cudaError_t>ccuda._cuMemcpyHtoD_v2(<ccuda.CUdeviceptr_v2>dst, src, size)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        err = <cudaError_t>ccuda._cuMemcpyDtoD_v2(<ccuda.CUdeviceptr_v2>dst, <ccuda.CUdeviceptr_v2>src, size)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        err = <cudaError_t>ccuda._cuMemcpy(<ccuda.CUdeviceptr_v2>dst, <ccuda.CUdeviceptr_v2>src, size)
    else:
        return cudaErrorInvalidMemcpyDirection


cdef cudaError_t mallocHost(size_t size, void **mem, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    if size == 0:
        if mem == NULL:
            return cudaErrorInvalidValue
        mem[0] = NULL
        return cudaSuccess
    else:
        return <cudaError_t>ccuda._cuMemHostAlloc(mem, size, flags)


cdef cudaError_t mallocPitch(size_t width, size_t height, size_t depth, void **mem, size_t *pitch) except ?cudaErrorCallRequiresNewerDriver nogil:
    height *= depth

    if width == 0 or height == 0:
        if mem == NULL or pitch == NULL:
            return cudaErrorInvalidValue
        mem[0]   = NULL
        pitch[0] = 0
    else:
        return <cudaError_t>ccuda._cuMemAllocPitch_v2(<ccuda.CUdeviceptr_v2*>mem, pitch, width, height, 4)
    return cudaSuccess


cdef cudaError_t mallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const cudaChannelFormatDesc *desc,
                                      size_t depth, size_t height, size_t width, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    if mipmappedArray == NULL:
        return cudaErrorInvalidValue

    cdef ccuda.CUmipmappedArray mipmap = NULL
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 ad
    memset(&ad, 0, sizeof(ad))

    mipmappedArray[0] = NULL
    if (((width == 0)) or
        ((height == 0) and (depth != 0) and not (flags & cudaArrayLayered)) or
        ((flags & cudaArrayLayered) and (depth == 0)) or
        ((flags & cudaArrayCubemap) and not (flags & cudaArrayLayered) and ((width != height) or (depth != 6))) or
        ((flags & cudaArrayLayered) and (flags & cudaArrayCubemap) and ((width != height) or (depth % 6 != 0)))):
        return cudaErrorInvalidValue
    else:
        err = getDescInfo(desc, <int*>&ad.NumChannels, &ad.Format)
        if err != cudaSuccess:
            return err
        ad.Height = <unsigned int>height
        ad.Width  = <unsigned int>width
        ad.Depth  = <unsigned int>depth
        ad.Flags  = flags
        err = <cudaError_t>ccuda._cuMipmappedArrayCreate(&mipmap, &ad, numLevels)
        if err != cudaSuccess:
            return err
        mipmappedArray[0] = <cudaMipmappedArray_t>mipmap
    return cudaSuccess


cdef cudaError_t memcpyAsyncDispatch(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    if size == 0:
        return cudaSuccess
    elif kind == cudaMemcpyKind.cudaMemcpyHostToHost:
        return memcpy2DPtr(<char*>dst, size, <const char*>src, size, size, 1, kind, stream, True)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        return <cudaError_t>ccuda._cuMemcpyDtoHAsync_v2(dst, <ccuda.CUdeviceptr_v2>src, size, stream)
    elif kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
        return<cudaError_t>ccuda._cuMemcpyHtoDAsync_v2(<ccuda.CUdeviceptr_v2>dst, src, size, stream)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        return<cudaError_t>ccuda._cuMemcpyDtoDAsync_v2(<ccuda.CUdeviceptr_v2>dst, <ccuda.CUdeviceptr_v2>src, size, stream)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        return<cudaError_t>ccuda._cuMemcpyAsync(<ccuda.CUdeviceptr_v2>dst, <ccuda.CUdeviceptr_v2>src, size, stream)
    return cudaErrorInvalidMemcpyDirection


cdef cudaError_t toCudartMemCopy3DParams(const ccuda.CUDA_MEMCPY3D_v2 *cd, cudaMemcpy3DParms *p) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaExtent srcBlockExtent
    cdef cudaExtent dstBlockExtent
    cdef cudaExtent copyBlockExtent
    cdef ccuda.CUarray_format srcFmt
    cdef ccuda.CUarray_format dstFmt
    cdef int numChannels = 0
    srcBlockExtent.width = srcBlockExtent.height = srcBlockExtent.depth = 1
    dstBlockExtent.width = dstBlockExtent.height = dstBlockExtent.depth = 1
    copyBlockExtent.width = copyBlockExtent.height = copyBlockExtent.depth = 1

    memset(p, 0, sizeof(cudaMemcpy3DParms))
    p[0].srcPtr.xsize = 0
    p[0].dstPtr.xsize = 0

    if (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST and cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST):
        p[0].kind = cudaMemcpyHostToHost

        p[0].srcPtr.ptr = <void*>cd[0].srcHost
        p[0].srcPtr.pitch = cd[0].srcPitch
        p[0].srcPtr.ysize = cd[0].srcHeight

        p[0].dstPtr.ptr = cd[0].dstHost
        p[0].dstPtr.pitch = cd[0].dstPitch
        p[0].dstPtr.ysize = cd[0].dstHeight
    elif (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST
            and (cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE
                or cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)):
        p[0].kind = cudaMemcpyHostToDevice

        p[0].srcPtr.ptr = <void*>cd[0].srcHost
        p[0].srcPtr.pitch = cd[0].srcPitch
        p[0].srcPtr.ysize = cd[0].srcHeight

        if (cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
            p[0].dstArray = <cudaArray_t>cd[0].dstArray
        else:
            p[0].dstPtr.ptr = <void*>cd[0].dstDevice
            p[0].dstPtr.pitch = cd[0].dstPitch
            p[0].dstPtr.ysize = cd[0].dstHeight
    elif ((cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE or cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)
            and cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST):
        p[0].kind = cudaMemcpyDeviceToHost

        if (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
            p[0].srcArray = <cudaArray_t>cd[0].srcArray
        else:
            p[0].srcPtr.ptr = <void*>cd[0].srcDevice
            p[0].srcPtr.pitch = cd[0].srcPitch
            p[0].srcPtr.ysize = cd[0].srcHeight

        p[0].dstPtr.ptr = cd[0].dstHost
        p[0].dstPtr.pitch = cd[0].dstPitch
        p[0].dstPtr.ysize = cd[0].dstHeight
    elif ((cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE or cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)
            and (cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE or cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY)):
        p[0].kind = cudaMemcpyDeviceToDevice

        if (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
            p[0].srcArray = <cudaArray_t>cd[0].srcArray
        else:
            p[0].srcPtr.ptr = <void*>cd[0].srcDevice
            p[0].srcPtr.pitch = cd[0].srcPitch
            p[0].srcPtr.ysize = cd[0].srcHeight

        if (cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
            p[0].dstArray = <cudaArray_t>cd[0].dstArray
        else:
            p[0].dstPtr.ptr = <void*>cd[0].dstDevice
            p[0].dstPtr.pitch = cd[0].dstPitch
            p[0].dstPtr.ysize = cd[0].dstHeight
    elif (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED and cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED):
        p[0].kind = cudaMemcpyDefault

        p[0].srcPtr.ptr = <void*>cd[0].srcDevice
        p[0].srcPtr.pitch = cd[0].srcPitch
        p[0].srcPtr.ysize = cd[0].srcHeight

        p[0].dstPtr.ptr = <void*>cd[0].dstDevice
        p[0].dstPtr.pitch = cd[0].dstPitch
        p[0].dstPtr.ysize = cd[0].dstHeight
    elif (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED and cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
        p[0].kind = cudaMemcpyDefault

        p[0].srcPtr.ptr = <void*>cd[0].srcDevice
        p[0].srcPtr.pitch = cd[0].srcPitch
        p[0].srcPtr.ysize = cd[0].srcHeight

        p[0].dstArray = <cudaArray_t>cd[0].dstArray
    elif (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY and cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED):
        p[0].kind = cudaMemcpyDefault

        p[0].srcArray = <cudaArray_t>cd[0].srcArray

        p[0].dstPtr.ptr = <void*>cd[0].dstDevice
        p[0].dstPtr.pitch = cd[0].dstPitch
        p[0].dstPtr.ysize = cd[0].dstHeight
    else:
        return cudaErrorUnknown

    cdef size_t srcElementSize = 0
    cdef size_t dstElementSize = 0
    cdef cudaError_t err = cudaSuccess

    if (cd[0].srcMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
        err = getFormat(<cudaArray_t>cd[0].srcArray, numChannels, &srcFmt)
        if err != cudaSuccess:
            return err
        err = getArrayBlockExtent(&srcBlockExtent, srcFmt)
        if err != cudaSuccess:
            return err
        err = getElementSize(&srcElementSize, <cudaArray_t>cd[0].srcArray)
        if err != cudaSuccess:
            return err
        copyBlockExtent = srcBlockExtent

    if (cd[0].dstMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY):
        err = getFormat(<cudaArray_t>cd[0].dstArray, numChannels, &dstFmt)
        if err != cudaSuccess:
            return err
        err = getArrayBlockExtent(&dstBlockExtent, dstFmt)
        if err != cudaSuccess:
            return err
        err = getElementSize(&dstElementSize, <cudaArray_t>cd[0].dstArray)
        if err != cudaSuccess:
            return err
        if cd[0].srcMemoryType != ccuda.CUmemorytype_enum.CU_MEMORYTYPE_ARRAY:
            copyBlockExtent = dstBlockExtent

    if (srcElementSize and dstElementSize and srcElementSize != dstElementSize):
        return cudaErrorInvalidValue

    cdef size_t elementSize = sizeof(char)
    if (srcElementSize):
        elementSize = srcElementSize
    if (dstElementSize):
        elementSize = dstElementSize
    srcElementSize = elementSize
    dstElementSize = elementSize

    p[0].extent.width = <size_t>(cd[0].WidthInBytes / elementSize) * copyBlockExtent.width
    p[0].extent.height = cd[0].Height * copyBlockExtent.height
    p[0].extent.depth = cd[0].Depth

    p[0].srcPos.x = <size_t>(cd[0].srcXInBytes / elementSize) * srcBlockExtent.width
    p[0].srcPos.y = cd[0].srcY * srcBlockExtent.height
    p[0].srcPos.z = cd[0].srcZ

    p[0].dstPos.x = <size_t>(cd[0].dstXInBytes / elementSize) * dstBlockExtent.width
    p[0].dstPos.y = cd[0].dstY * dstBlockExtent.height
    p[0].dstPos.z = cd[0].dstZ
    return cudaSuccess


cdef cudaError_t memcpy2DFromArray(char *dst, size_t dpitch, cudaArray_const_t src, size_t hOffset,
        size_t wOffset, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    if width == 0 or height == 0:
        return cudaSuccess
    if height > 1 and width > dpitch:
        return cudaErrorInvalidPitchValue

    if kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        err = copyToHost2D(src, hOffset, wOffset, dst, dpitch, width, height, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        err = copyToDevice2D(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, src, hOffset, wOffset, dst, 0, dpitch, width, height, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        err = copyToDevice2D(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED, src, hOffset, wOffset, dst, 0, dpitch, width, height, sid, async)
    else:
        return cudaErrorInvalidMemcpyDirection
    return err


cdef cudaError_t memcpy2DArrayToArray(cudaArray_t dst, size_t hOffsetDst, size_t wOffsetDst,
                                      cudaArray_const_t src, size_t hOffsetSrc, size_t wOffsetSrc,
                                      size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    if width == 0 or height == 0:
        return cudaSuccess
    if kind != cudaMemcpyKind.cudaMemcpyDeviceToDevice and kind != cudaMemcpyKind.cudaMemcpyDefault:
        return cudaErrorInvalidMemcpyDirection
    return copyToArray2D(src, hOffsetSrc, wOffsetSrc, dst, hOffsetDst, wOffsetDst, width, height)


cdef cudaError_t memset3DPtr(cudaPitchedPtr p, int val, cudaExtent e, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if e.width == 0 or e.height == 0 or e.depth == 0:
        return cudaSuccess

    if (e.height > 1 or e.depth > 1) and e.width > p.pitch:
        return cudaErrorInvalidValue

    if e.depth > 0 and e.height > p.ysize:
        return cudaErrorInvalidValue

    cdef char *ptr = <char*>p.ptr
    cdef size_t d
    cdef cudaError_t err = cudaSuccess

    if e.width >= p.xsize and e.height == p.ysize and e.width == p.pitch:
        return memsetPtr(ptr, val, e.width * e.height * e.depth, sid, async)
    elif e.height == p.ysize:
        return memset2DPtr(ptr, p.pitch, val, e.width, e.height * e.depth, sid, async)
    else:
        d = 0
        while (d != e.depth):
            err = memset2DPtr(ptr, p.pitch, val, e.width, e.height, sid, async)
            if err != cudaSuccess:
                return err
            ptr += p.pitch * p.ysize
            d += 1
    return cudaSuccess


cdef cudaError_t memcpyToArray(cudaArray_t dst, size_t hOffset, size_t wOffset, const char *src,
                               size_t count, cudaMemcpyKind kind,
                               cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if count == 0:
        return cudaSuccess

    if kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
        return copyFromHost(dst, hOffset, wOffset, src, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        return copyFromDevice(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, dst, hOffset, wOffset, src, 0, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        return copyFromDevice(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED, dst, hOffset, wOffset, src, 0, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyHostToHost or kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        return cudaErrorInvalidMemcpyDirection
    return cudaSuccess


cdef cudaError_t memcpyFromArray(char *dst, cudaArray_const_t src, size_t hOffset, size_t wOffset,
                                 size_t count, cudaMemcpyKind kind,
                                 cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil:
    if count == 0:
        return cudaSuccess

    if kind == cudaMemcpyKind.cudaMemcpyDeviceToHost:
        return copyToHost(src, hOffset, wOffset, dst, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
        return copyToDevice(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE, src, hOffset, wOffset, dst, 0, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyDefault:
        return copyToDevice(ccuda.CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED, src, hOffset, wOffset, dst, 0, count, sid, async)
    elif kind == cudaMemcpyKind.cudaMemcpyHostToDevice or kind == cudaMemcpyKind.cudaMemcpyHostToHost:
        return cudaErrorInvalidMemcpyDirection
    return cudaSuccess


cdef cudaError_t toDriverCudaResourceDesc(ccuda.CUDA_RESOURCE_DESC *_driver_pResDesc, const cudaResourceDesc *pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef int numChannels
    cdef ccuda.CUarray_format format

    if pResDesc[0].resType == cudaResourceType.cudaResourceTypeArray:
        _driver_pResDesc[0].resType          = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY
        _driver_pResDesc[0].res.array.hArray = <ccuda.CUarray>pResDesc[0].res.array.array
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeMipmappedArray:
        _driver_pResDesc[0].resType                    = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
        _driver_pResDesc[0].res.mipmap.hMipmappedArray = <ccuda.CUmipmappedArray>pResDesc[0].res.mipmap.mipmap
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeLinear:
        _driver_pResDesc[0].resType                = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR
        _driver_pResDesc[0].res.linear.devPtr      = <ccuda.CUdeviceptr>pResDesc[0].res.linear.devPtr
        _driver_pResDesc[0].res.linear.sizeInBytes = pResDesc[0].res.linear.sizeInBytes
        err = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        _driver_pResDesc[0].res.linear.format      = format
        _driver_pResDesc[0].res.linear.numChannels = numChannels
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypePitch2D:
        _driver_pResDesc[0].resType                  = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D
        _driver_pResDesc[0].res.pitch2D.devPtr       = <ccuda.CUdeviceptr>pResDesc[0].res.pitch2D.devPtr
        _driver_pResDesc[0].res.pitch2D.pitchInBytes = pResDesc[0].res.pitch2D.pitchInBytes
        _driver_pResDesc[0].res.pitch2D.width        = pResDesc[0].res.pitch2D.width
        _driver_pResDesc[0].res.pitch2D.height       = pResDesc[0].res.pitch2D.height
        err = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        _driver_pResDesc[0].res.pitch2D.format       = format
        _driver_pResDesc[0].res.pitch2D.numChannels  = numChannels
    else:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    _driver_pResDesc[0].flags = 0

    return err


cdef cudaError_t getDriverEglFrame(ccuda.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int i = 0

    err = getDescInfo(&eglFrame.planeDesc[0].channelDesc, <int*>&cuEglFrame[0].numChannels, &cuEglFrame[0].cuFormat)
    if err != cudaSuccess:
        return err
    for i in range(eglFrame.planeCount):
        if eglFrame.frameType == cudaEglFrameTypeArray:
            cuEglFrame[0].frame.pArray[i] = <ccuda.CUarray>eglFrame.frame.pArray[i]
        else:
            cuEglFrame[0].frame.pPitch[i] = eglFrame.frame.pPitch[i].ptr
    cuEglFrame[0].width = eglFrame.planeDesc[0].width
    cuEglFrame[0].height = eglFrame.planeDesc[0].height
    cuEglFrame[0].depth = eglFrame.planeDesc[0].depth
    cuEglFrame[0].pitch = eglFrame.planeDesc[0].pitch
    cuEglFrame[0].planeCount = eglFrame.planeCount
    if eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUYV422:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY422:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422
    elif eglFrame.eglColorFormat == cudaEglColorFormatARGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatRGBA:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA
    elif eglFrame.eglColorFormat == cudaEglColorFormatABGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBGRA:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA
    elif eglFrame.eglColorFormat == cudaEglColorFormatL:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L
    elif eglFrame.eglColorFormat == cudaEglColorFormatR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R
    elif eglFrame.eglColorFormat == cudaEglColorFormatA:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A
    elif eglFrame.eglColorFormat == cudaEglColorFormatRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG
    elif eglFrame.eglColorFormat == cudaEglColorFormatAYUV:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatVYUY_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatUYVY_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUYV_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVYU_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUVA_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatAYUV_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerRGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerBGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerGRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerGBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10RGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10BGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10GRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10GBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12RGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12BGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12GRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12GBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14RGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14BGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14GRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer14GBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20RGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20BGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20GRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer20GBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspRGGB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspBGGR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspGRBG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerIspGBRG:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU444Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU422Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerBCCR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerRCCB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerCRBC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayerCBRC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer10CCCC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12BCCR:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12RCCB:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CRBC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CBRC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC
    elif eglFrame.eglColorFormat == cudaEglColorFormatBayer12CCCC:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC
    elif eglFrame.eglColorFormat == cudaEglColorFormatY:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUV420Planar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVU420Planar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar_2020:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_422SemiPlanar_709:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709
    elif eglFrame.eglColorFormat == cudaEglColorFormatY_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatYUVA:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA
    elif eglFrame.eglColorFormat == cudaEglColorFormatYVYU:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU
    elif eglFrame.eglColorFormat == cudaEglColorFormatVYUY:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat =  ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER
    elif eglFrame.eglColorFormat == cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER:
        cuEglFrame[0].eglColorFormat = ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER
    else:
        return cudaErrorInvalidValue
    if eglFrame.frameType == cudaEglFrameTypeArray:
        cuEglFrame[0].frameType = ccuda.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY
    elif eglFrame.frameType == cudaEglFrameTypePitch:
        cuEglFrame[0].frameType = ccuda.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH
    else:
        return cudaErrorInvalidValue


@cython.show_performance_hints(False)
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, ccuda.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef unsigned int i
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 ad
    cdef cudaPitchedPtr pPtr
    memset(eglFrame, 0, sizeof(eglFrame[0]))
    memset(&ad, 0, sizeof(ad))
    for i in range(cueglFrame.planeCount):
        ad.Depth = cueglFrame.depth
        ad.Flags = 0
        ad.Format = cueglFrame.cuFormat
        ad.Height = cueglFrame.height
        ad.NumChannels = cueglFrame.numChannels
        ad.Width = cueglFrame.width

        err = getChannelFormatDescFromDriverDesc(&eglFrame[0].planeDesc[i].channelDesc, NULL, NULL, NULL, &ad)
        if err != cudaSuccess:
            return err

        eglFrame[0].planeDesc[i].depth = cueglFrame.depth
        eglFrame[0].planeDesc[i].numChannels = cueglFrame.numChannels
        if i == 0:
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = <unsigned int>(cueglFrame.height / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709 or 
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = <unsigned int>(cueglFrame.height / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709):
            eglFrame[0].planeDesc[i].width = <unsigned int>(cueglFrame.width / 2)
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].pitch = <unsigned int>(cueglFrame.pitch / 2)
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
        elif (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER or
              cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER):
            eglFrame[0].planeDesc[i].height = cueglFrame.height
            eglFrame[0].planeDesc[i].width = cueglFrame.width
            eglFrame[0].planeDesc[i].pitch = cueglFrame.pitch
            eglFrame[0].planeDesc[1].channelDesc.y = 8
            if (cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER or
                cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER):
                eglFrame[0].planeDesc[1].channelDesc.y = 16
        if cueglFrame.frameType == ccuda.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY:
            eglFrame[0].frame.pArray[i] = <cudaArray_t>cueglFrame.frame.pArray[i]
        else:
            pPtr = make_cudaPitchedPtr(cueglFrame.frame.pPitch[i], eglFrame[0].planeDesc[i].pitch,
                    eglFrame[0].planeDesc[i].width, eglFrame[0].planeDesc[i].height)
            eglFrame[0].frame.pPitch[i] = pPtr

    eglFrame[0].planeCount = cueglFrame.planeCount
    if cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_422:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUYV422
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_422:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY422
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ARGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatARGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RGBA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatRGBA
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_ABGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatABGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BGRA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBGRA
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_L:
        eglFrame[0].eglColorFormat = cudaEglColorFormatL
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_R:
        eglFrame[0].eglColorFormat = cudaEglColorFormatR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_A:
        eglFrame[0].eglColorFormat = cudaEglColorFormatA
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_RG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV:
        eglFrame[0].eglColorFormat = cudaEglColorFormatAYUV
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatVYUY_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_UYVY_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatUYVY_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUYV_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUYV_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVYU_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUVA_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_AYUV_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatAYUV_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerRGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerBGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerGRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerGBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10RGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10BGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10GRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10GBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12RGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12BGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12GRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12GBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14RGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14BGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14GRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER14_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer14GBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20RGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20BGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20GRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER20_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer20GBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspRGGB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspBGGR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspGRBG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerIspGBRG
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU444_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU444Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU422_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU422Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_BCCR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerBCCR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_RCCB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerRCCB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CRBC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerCRBC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER_CBRC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayerCBRC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER10_CCCC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer10CCCC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_BCCR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12BCCR
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_RCCB:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12RCCB
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CRBC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CRBC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CBRC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CBRC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_BAYER12_CCCC:
        eglFrame[0].eglColorFormat = cudaEglColorFormatBayer12CCCC
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420SemiPlanar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420SemiPlanar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUV420Planar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVU420Planar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar_2020
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_422SemiPlanar_709
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YUVA:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYUVA
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_YVYU:
        eglFrame[0].eglColorFormat = cudaEglColorFormatYVYU
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_VYUY:
        eglFrame[0].eglColorFormat = cudaEglColorFormatVYUY
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar_ER
    elif cueglFrame.eglColorFormat == ccuda.CUeglColorFormat_enum.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER:
        eglFrame[0].eglColorFormat = cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER
    else:
        return cudaErrorInvalidValue
    if cueglFrame.frameType == ccuda.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_ARRAY:
        eglFrame[0].frameType = cudaEglFrameTypeArray
    elif cueglFrame.frameType == ccuda.CUeglFrameType_enum.CU_EGL_FRAME_TYPE_PITCH:
        eglFrame[0].frameType = cudaEglFrameTypePitch
    else:
        return cudaErrorInvalidValue


cdef cudaError_t toDriverGraphNodeParams(const cudaGraphNodeParams *rtParams, ccuda.CUgraphNodeParams *driverParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err
    cdef ccuda.CUcontext context
    memset(driverParams, 0, sizeof(driverParams[0]))

    if rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeKernel:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_KERNEL
        err = toDriverKernelNodeParams(<const cudaKernelNodeParams *>&rtParams[0].kernel, <ccuda.CUDA_KERNEL_NODE_PARAMS *>&driverParams[0].kernel)
        if err != cudaSuccess:
            return err
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEMCPY
        err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        err = toDriverMemCopy3DParams(&rtParams[0].memcpy.copyParams, &driverParams[0].memcpy.copyParams)
        if err != cudaSuccess:
            return err
        driverParams[0].memcpy.copyCtx = context
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeMemset:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEMSET
        err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        toDriverMemsetNodeParams(<const cudaMemsetParams *>&rtParams[0].memset, <ccuda.CUDA_MEMSET_NODE_PARAMS *>&driverParams[0].memset)
        driverParams[0].memset.ctx = context
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeHost:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_HOST
        toDriverHostNodeParams(<const cudaHostNodeParams *>&rtParams[0].host, <ccuda.CUDA_HOST_NODE_PARAMS *>&driverParams[0].host)
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeGraph:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_GRAPH
        driverParams[0].graph.graph = rtParams[0].graph.graph
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeEmpty:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_EMPTY
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeWaitEvent:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_WAIT_EVENT
        driverParams[0].eventWait.event = rtParams[0].eventWait.event
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeEventRecord:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_EVENT_RECORD
        driverParams[0].eventRecord.event = rtParams[0].eventRecord.event
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreSignal:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL
        driverParams[0].extSemSignal = (<ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 *>(&rtParams[0].extSemSignal))[0]
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreWait:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT
        driverParams[0].extSemWait = (<ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 *>&rtParams[0].extSemWait)[0]
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeMemAlloc:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEM_ALLOC
        driverParams[0].alloc = (<ccuda.CUDA_MEM_ALLOC_NODE_PARAMS_v2 *>&rtParams[0].alloc)[0]
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeMemFree:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEM_FREE
        driverParams[0].free.dptr = <ccuda.CUdeviceptr>rtParams[0].free.dptr
    elif rtParams[0].type == cudaGraphNodeType.cudaGraphNodeTypeConditional:
        driverParams[0].type = ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_CONDITIONAL
        # RT params mirror the driver params except the RT struct lacks the ctx at the end.
        memcpy(&driverParams[0].conditional, &rtParams[0].conditional, sizeof(rtParams[0].conditional))
        err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        driverParams[0].conditional.ctx = context
    else:
        return cudaErrorInvalidValue
    return cudaSuccess


cdef void toCudartGraphNodeOutParams(const ccuda.CUgraphNodeParams *driverParams, cudaGraphNodeParams *rtParams) noexcept nogil:
    if driverParams[0].type == ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEM_ALLOC:
        rtParams[0].alloc.dptr = <void *>driverParams[0].alloc.dptr
    elif driverParams[0].type == ccuda.CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_CONDITIONAL:
        rtParams[0].conditional.phGraph_out = <cudaGraph_t *>driverParams[0].conditional.phGraph_out


cdef cudaError_t toDriverKernelNodeParams(const cudaKernelNodeParams nodeParams[0], ccuda.CUDA_KERNEL_NODE_PARAMS *driverNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    cdef ccuda.CUcontext context
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    driverNodeParams[0].func = <ccuda.CUfunction>nodeParams[0].func
    driverNodeParams[0].kern = NULL
    driverNodeParams[0].ctx = context
    driverNodeParams[0].gridDimX = nodeParams[0].gridDim.x
    driverNodeParams[0].gridDimY = nodeParams[0].gridDim.y
    driverNodeParams[0].gridDimZ = nodeParams[0].gridDim.z
    driverNodeParams[0].blockDimX = nodeParams[0].blockDim.x
    driverNodeParams[0].blockDimY = nodeParams[0].blockDim.y
    driverNodeParams[0].blockDimZ = nodeParams[0].blockDim.z
    driverNodeParams[0].sharedMemBytes = nodeParams[0].sharedMemBytes
    driverNodeParams[0].kernelParams = nodeParams[0].kernelParams
    driverNodeParams[0].extra = nodeParams[0].extra
    return err


cdef void toDriverHostNodeParams(const cudaHostNodeParams *pRuntimeNodeParams, ccuda.CUDA_HOST_NODE_PARAMS *pDriverNodeParams) noexcept nogil:
    pDriverNodeParams[0].fn = pRuntimeNodeParams[0].fn
    pDriverNodeParams[0].userData = pRuntimeNodeParams[0].userData


@cython.show_performance_hints(False)
cdef void cudaAsyncNotificationCallbackWrapper(cudaAsyncNotificationInfo_t *info, void *data, cudaAsyncCallbackHandle_t handle) nogil:
    cdef cudaAsyncCallbackData *cbData = <cudaAsyncCallbackData *>data
    with gil:
        cbData.callback(info, cbData.userData, handle)


cdef cudaError_t DeviceRegisterAsyncNotificationCommon(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaAsyncCallbackData *cbData = NULL
    cdef cudaError_t err = cudaSuccess
    cbData = <cudaAsyncCallbackData *>malloc(sizeof(cbData[0]))

    if cbData == NULL:
        return cudaErrorMemoryAllocation

    cbData.callback = callbackFunc
    cbData.userData = userData
    err = <cudaError_t>ccuda._cuDeviceRegisterAsyncNotification(<ccuda.CUdevice>device, <ccuda.CUasyncCallback>cudaAsyncNotificationCallbackWrapper, <void*>cbData, <ccuda.CUasyncCallbackHandle*>callback)
    if err != cudaSuccess:
        free(cbData)

    m_global._asyncCallbackDataMap[callback[0]] = cbData

    return err

cdef cudaError_t DeviceUnregisterAsyncNotificationCommon(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef cudaError_t err = cudaSuccess
    err = <cudaError_t>ccuda._cuDeviceUnregisterAsyncNotification(<ccuda.CUdevice>device, <ccuda.CUasyncCallbackHandle>callback)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    free(m_global._asyncCallbackDataMap[callback])
    m_global._asyncCallbackDataMap.erase(callback)

    return err
