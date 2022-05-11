# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda._cuda.ccuda as ccuda

cdef CUresult cuGetErrorString(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGetErrorString(error, pStr)

cdef CUresult cuGetErrorName(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGetErrorName(error, pStr)

cdef CUresult cuInit(unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuInit(Flags)

cdef CUresult cuDriverGetVersion(int* driverVersion) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDriverGetVersion(driverVersion)

cdef CUresult cuDeviceGet(CUdevice* device, int ordinal) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGet(device, ordinal)

cdef CUresult cuDeviceGetCount(int* count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetCount(count)

cdef CUresult cuDeviceGetName(char* name, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetName(name, length, dev)

cdef CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetUuid(uuid, dev)

cdef CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetUuid_v2(uuid, dev)

cdef CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetLuid(luid, deviceNodeMask, dev)

cdef CUresult cuDeviceTotalMem(size_t* numbytes, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceTotalMem_v2(numbytes, dev)

cdef CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format pformat, unsigned numChannels, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, pformat, numChannels, dev)

cdef CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetAttribute(pi, attrib, dev)

cdef CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags)

cdef CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceSetMemPool(dev, pool)

cdef CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetMemPool(pool, dev)

cdef CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetDefaultMemPool(pool_out, dev)

cdef CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFlushGPUDirectRDMAWrites(target, scope)

cdef CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetProperties(prop, dev)

cdef CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceComputeCapability(major, minor, dev)

cdef CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDevicePrimaryCtxRetain(pctx, dev)

cdef CUresult cuDevicePrimaryCtxRelease(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDevicePrimaryCtxRelease_v2(dev)

cdef CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDevicePrimaryCtxSetFlags_v2(dev, flags)

cdef CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDevicePrimaryCtxGetState(dev, flags, active)

cdef CUresult cuDevicePrimaryCtxReset(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDevicePrimaryCtxReset_v2(dev)

cdef CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType typename, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetExecAffinitySupport(pi, typename, dev)

cdef CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxCreate_v2(pctx, flags, dev)

cdef CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev)

cdef CUresult cuCtxDestroy(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxDestroy_v2(ctx)

cdef CUresult cuCtxPushCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxPushCurrent_v2(ctx)

cdef CUresult cuCtxPopCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxPopCurrent_v2(pctx)

cdef CUresult cuCtxSetCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxSetCurrent(ctx)

cdef CUresult cuCtxGetCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetCurrent(pctx)

cdef CUresult cuCtxGetDevice(CUdevice* device) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetDevice(device)

cdef CUresult cuCtxGetFlags(unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetFlags(flags)

cdef CUresult cuCtxSynchronize() nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxSynchronize()

cdef CUresult cuCtxSetLimit(CUlimit limit, size_t value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxSetLimit(limit, value)

cdef CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetLimit(pvalue, limit)

cdef CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetCacheConfig(pconfig)

cdef CUresult cuCtxSetCacheConfig(CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxSetCacheConfig(config)

cdef CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetSharedMemConfig(pConfig)

cdef CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxSetSharedMemConfig(config)

cdef CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetApiVersion(ctx, version)

cdef CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)

cdef CUresult cuCtxResetPersistingL2Cache() nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxResetPersistingL2Cache()

cdef CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType typename) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxGetExecAffinity(pExecAffinity, typename)

cdef CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxAttach(pctx, flags)

cdef CUresult cuCtxDetach(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxDetach(ctx)

cdef CUresult cuModuleLoad(CUmodule* module, const char* fname) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleLoad(module, fname)

cdef CUresult cuModuleLoadData(CUmodule* module, const void* image) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleLoadData(module, image)

cdef CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleLoadDataEx(module, image, numOptions, options, optionValues)

cdef CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleLoadFatBinary(module, fatCubin)

cdef CUresult cuModuleUnload(CUmodule hmod) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleUnload(hmod)

cdef CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleGetFunction(hfunc, hmod, name)

cdef CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* numbytes, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleGetGlobal_v2(dptr, numbytes, hmod, name)

cdef CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleGetTexRef(pTexRef, hmod, name)

cdef CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleGetSurfRef(pSurfRef, hmod, name)

cdef CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLinkCreate_v2(numOptions, options, optionValues, stateOut)

cdef CUresult cuLinkAddData(CUlinkState state, CUjitInputType typename, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLinkAddData_v2(state, typename, data, size, name, numOptions, options, optionValues)

cdef CUresult cuLinkAddFile(CUlinkState state, CUjitInputType typename, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLinkAddFile_v2(state, typename, path, numOptions, options, optionValues)

cdef CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLinkComplete(state, cubinOut, sizeOut)

cdef CUresult cuLinkDestroy(CUlinkState state) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLinkDestroy(state)

cdef CUresult cuMemGetInfo(size_t* free, size_t* total) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetInfo_v2(free, total)

cdef CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAlloc_v2(dptr, bytesize)

cdef CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)

cdef CUresult cuMemFree(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemFree_v2(dptr)

cdef CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetAddressRange_v2(pbase, psize, dptr)

cdef CUresult cuMemAllocHost(void** pp, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAllocHost_v2(pp, bytesize)

cdef CUresult cuMemFreeHost(void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemFreeHost(p)

cdef CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemHostAlloc(pp, bytesize, Flags)

cdef CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemHostGetDevicePointer_v2(pdptr, p, Flags)

cdef CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemHostGetFlags(pFlags, p)

cdef CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAllocManaged(dptr, bytesize, flags)

cdef CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetByPCIBusId(dev, pciBusId)

cdef CUresult cuDeviceGetPCIBusId(char* pciBusId, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetPCIBusId(pciBusId, length, dev)

cdef CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuIpcGetEventHandle(pHandle, event)

cdef CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuIpcOpenEventHandle(phEvent, handle)

cdef CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuIpcGetMemHandle(pHandle, dptr)

cdef CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuIpcOpenMemHandle_v2(pdptr, handle, Flags)

cdef CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuIpcCloseMemHandle(dptr)

cdef CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemHostRegister_v2(p, bytesize, Flags)

cdef CUresult cuMemHostUnregister(void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemHostUnregister(p)

cdef CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy(dst, src, ByteCount)

cdef CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)

cdef CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)

cdef CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)

cdef CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)

cdef CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)

cdef CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)

cdef CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)

cdef CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)

cdef CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)

cdef CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy2D_v2(pCopy)

cdef CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy2DUnaligned_v2(pCopy)

cdef CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy3D_v2(pCopy)

cdef CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy3DPeer(pCopy)

cdef CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyAsync(dst, src, ByteCount, hStream)

cdef CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)

cdef CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)

cdef CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)

cdef CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)

cdef CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)

cdef CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)

cdef CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy2DAsync_v2(pCopy, hStream)

cdef CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy3DAsync_v2(pCopy, hStream)

cdef CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemcpy3DPeerAsync(pCopy, hStream)

cdef CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD8_v2(dstDevice, uc, N)

cdef CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD16_v2(dstDevice, us, N)

cdef CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD32_v2(dstDevice, ui, N)

cdef CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)

cdef CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)

cdef CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)

cdef CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD8Async(dstDevice, uc, N, hStream)

cdef CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD16Async(dstDevice, us, N, hStream)

cdef CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD32Async(dstDevice, ui, N, hStream)

cdef CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)

cdef CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)

cdef CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)

cdef CUresult cuArrayCreate(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayCreate_v2(pHandle, pAllocateArray)

cdef CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)

cdef CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayGetSparseProperties(sparseProperties, array)

cdef CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap)

cdef CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayGetMemoryRequirements(memoryRequirements, array, device)

cdef CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)

cdef CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayGetPlane(pPlaneArray, hArray, planeIdx)

cdef CUresult cuArrayDestroy(CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArrayDestroy(hArray)

cdef CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArray3DCreate_v2(pHandle, pAllocateArray)

cdef CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)

cdef CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)

cdef CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)

cdef CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMipmappedArrayDestroy(hMipmappedArray)

cdef CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAddressReserve(ptr, size, alignment, addr, flags)

cdef CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAddressFree(ptr, size)

cdef CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemCreate(handle, size, prop, flags)

cdef CUresult cuMemRelease(CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemRelease(handle)

cdef CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemMap(ptr, size, offset, handle, flags)

cdef CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemMapArrayAsync(mapInfoList, count, hStream)

cdef CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemUnmap(ptr, size)

cdef CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemSetAccess(ptr, size, desc, count)

cdef CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetAccess(flags, location, ptr)

cdef CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags)

cdef CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemImportFromShareableHandle(handle, osHandle, shHandleType)

cdef CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetAllocationGranularity(granularity, prop, option)

cdef CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetAllocationPropertiesFromHandle(prop, handle)

cdef CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemRetainAllocationHandle(handle, addr)

cdef CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemFreeAsync(dptr, hStream)

cdef CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAllocAsync(dptr, bytesize, hStream)

cdef CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolTrimTo(pool, minBytesToKeep)

cdef CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolSetAttribute(pool, attr, value)

cdef CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolGetAttribute(pool, attr, value)

cdef CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolSetAccess(pool, map, count)

cdef CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolGetAccess(flags, memPool, location)

cdef CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolCreate(pool, poolProps)

cdef CUresult cuMemPoolDestroy(CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolDestroy(pool)

cdef CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream)

cdef CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags)

cdef CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags)

cdef CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolExportPointer(shareData_out, ptr)

cdef CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPoolImportPointer(ptr_out, pool, shareData)

cdef CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuPointerGetAttribute(data, attribute, ptr)

cdef CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemPrefetchAsync(devPtr, count, dstDevice, hStream)

cdef CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemAdvise(devPtr, count, advice, device)

cdef CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)

cdef CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)

cdef CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuPointerSetAttribute(value, attribute, ptr)

cdef CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuPointerGetAttributes(numAttributes, attributes, data, ptr)

cdef CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamCreate(phStream, Flags)

cdef CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamCreateWithPriority(phStream, flags, priority)

cdef CUresult cuStreamGetPriority(CUstream hStream, int* priority) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetPriority(hStream, priority)

cdef CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetFlags(hStream, flags)

cdef CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetCtx(hStream, pctx)

cdef CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWaitEvent(hStream, hEvent, Flags)

cdef CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamAddCallback(hStream, callback, userData, flags)

cdef CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamBeginCapture_v2(hStream, mode)

cdef CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuThreadExchangeStreamCaptureMode(mode)

cdef CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamEndCapture(hStream, phGraph)

cdef CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamIsCapturing(hStream, captureStatus)

cdef CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out)

cdef CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)

cdef CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags)

cdef CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamAttachMemAsync(hStream, dptr, length, flags)

cdef CUresult cuStreamQuery(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamQuery(hStream)

cdef CUresult cuStreamSynchronize(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamSynchronize(hStream)

cdef CUresult cuStreamDestroy(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamDestroy_v2(hStream)

cdef CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamCopyAttributes(dst, src)

cdef CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamGetAttribute(hStream, attr, value_out)

cdef CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamSetAttribute(hStream, attr, value)

cdef CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventCreate(phEvent, Flags)

cdef CUresult cuEventRecord(CUevent hEvent, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventRecord(hEvent, hStream)

cdef CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventRecordWithFlags(hEvent, hStream, flags)

cdef CUresult cuEventQuery(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventQuery(hEvent)

cdef CUresult cuEventSynchronize(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventSynchronize(hEvent)

cdef CUresult cuEventDestroy(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventDestroy_v2(hEvent)

cdef CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventElapsedTime(pMilliseconds, hStart, hEnd)

cdef CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuImportExternalMemory(extMem_out, memHandleDesc)

cdef CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)

cdef CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)

cdef CUresult cuDestroyExternalMemory(CUexternalMemory extMem) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDestroyExternalMemory(extMem)

cdef CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuImportExternalSemaphore(extSem_out, semHandleDesc)

cdef CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDestroyExternalSemaphore(extSem)

cdef CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWaitValue32(stream, addr, value, flags)

cdef CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWaitValue64(stream, addr, value, flags)

cdef CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWriteValue32(stream, addr, value, flags)

cdef CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWriteValue64(stream, addr, value, flags)

cdef CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamBatchMemOp(stream, count, paramArray, flags)

cdef CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWaitValue32_v2(stream, addr, value, flags)

cdef CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWaitValue64_v2(stream, addr, value, flags)

cdef CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWriteValue32_v2(stream, addr, value, flags)

cdef CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamWriteValue64_v2(stream, addr, value, flags)

cdef CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuStreamBatchMemOp_v2(stream, count, paramArray, flags)

cdef CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncGetAttribute(pi, attrib, hfunc)

cdef CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncSetAttribute(hfunc, attrib, value)

cdef CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncSetCacheConfig(hfunc, config)

cdef CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncSetSharedMemConfig(hfunc, config)

cdef CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncGetModule(hmod, hfunc)

cdef CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)

cdef CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)

cdef CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)

cdef CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchHostFunc(hStream, fn, userData)

cdef CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncSetBlockShape(hfunc, x, y, z)

cdef CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuFuncSetSharedSize(hfunc, numbytes)

cdef CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuParamSetSize(hfunc, numbytes)

cdef CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuParamSeti(hfunc, offset, value)

cdef CUresult cuParamSetf(CUfunction hfunc, int offset, float value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuParamSetf(hfunc, offset, value)

cdef CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuParamSetv(hfunc, offset, ptr, numbytes)

cdef CUresult cuLaunch(CUfunction f) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunch(f)

cdef CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchGrid(f, grid_width, grid_height)

cdef CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuLaunchGridAsync(f, grid_width, grid_height, hStream)

cdef CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuParamSetTexRef(hfunc, texunit, hTexRef)

cdef CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphCreate(phGraph, flags)

cdef CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphKernelNodeGetParams(hNode, nodeParams)

cdef CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphKernelNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)

cdef CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemcpyNodeGetParams(hNode, nodeParams)

cdef CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemcpyNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)

cdef CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemsetNodeGetParams(hNode, nodeParams)

cdef CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemsetNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphHostNodeGetParams(hNode, nodeParams)

cdef CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphHostNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph)

cdef CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphChildGraphNodeGetGraph(hNode, phGraph)

cdef CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)

cdef CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event)

cdef CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphEventRecordNodeGetEvent(hNode, event_out)

cdef CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphEventRecordNodeSetEvent(hNode, event)

cdef CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event)

cdef CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphEventWaitNodeGetEvent(hNode, event_out)

cdef CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphEventWaitNodeSetEvent(hNode, event)

cdef CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)

cdef CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)

cdef CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)

cdef CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphBatchMemOpNodeSetParams(hNode, nodeParams)

cdef CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)

cdef CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)

cdef CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemAllocNodeGetParams(hNode, params_out)

cdef CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr)

cdef CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphMemFreeNodeGetParams(hNode, dptr_out)

cdef CUresult cuDeviceGraphMemTrim(CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGraphMemTrim(device)

cdef CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetGraphMemAttribute(device, attr, value)

cdef CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceSetGraphMemAttribute(device, attr, value)

cdef CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphClone(phGraphClone, originalGraph)

cdef CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)

cdef CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* typename) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeGetType(hNode, typename)

cdef CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphGetNodes(hGraph, nodes, numNodes)

cdef CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)

cdef CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, size_t* numEdges) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphGetEdges(hGraph, from_, to, numEdges)

cdef CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeGetDependencies(hNode, dependencies, numDependencies)

cdef CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)

cdef CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphAddDependencies(hGraph, from_, to, numDependencies)

cdef CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphRemoveDependencies(hGraph, from_, to, numDependencies)

cdef CUresult cuGraphDestroyNode(CUgraphNode hNode) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphDestroyNode(hNode)

cdef CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)

cdef CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags)

cdef CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams)

cdef CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)

cdef CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)

cdef CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams)

cdef CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph)

cdef CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)

cdef CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)

cdef CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)

cdef CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)

cdef CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)

cdef CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)

cdef CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphUpload(hGraphExec, hStream)

cdef CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphLaunch(hGraphExec, hStream)

cdef CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecDestroy(hGraphExec)

cdef CUresult cuGraphDestroy(CUgraph hGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphDestroy(hGraph)

cdef CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)

cdef CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphKernelNodeCopyAttributes(dst, src)

cdef CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphKernelNodeGetAttribute(hNode, attr, value_out)

cdef CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphKernelNodeSetAttribute(hNode, attr, value)

cdef CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphDebugDotPrint(hGraph, path, flags)

cdef CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)

cdef CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuUserObjectRetain(object, count)

cdef CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuUserObjectRelease(object, count)

cdef CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphRetainUserObject(graph, object, count, flags)

cdef CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphReleaseUserObject(graph, object, count)

cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)

cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)

cdef CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)

cdef CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)

cdef CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)

cdef CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetArray(hTexRef, hArray, Flags)

cdef CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)

cdef CUresult cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, numbytes)

cdef CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)

cdef CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)

cdef CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetAddressMode(hTexRef, dim, am)

cdef CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetFilterMode(hTexRef, fm)

cdef CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetMipmapFilterMode(hTexRef, fm)

cdef CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetMipmapLevelBias(hTexRef, bias)

cdef CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)

cdef CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)

cdef CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetBorderColor(hTexRef, pBorderColor)

cdef CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefSetFlags(hTexRef, Flags)

cdef CUresult cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetAddress_v2(pdptr, hTexRef)

cdef CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetArray(phArray, hTexRef)

cdef CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)

cdef CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetAddressMode(pam, hTexRef, dim)

cdef CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetFilterMode(pfm, hTexRef)

cdef CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)

cdef CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetMipmapFilterMode(pfm, hTexRef)

cdef CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetMipmapLevelBias(pbias, hTexRef)

cdef CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)

cdef CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)

cdef CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetBorderColor(pBorderColor, hTexRef)

cdef CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefGetFlags(pFlags, hTexRef)

cdef CUresult cuTexRefCreate(CUtexref* pTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefCreate(pTexRef)

cdef CUresult cuTexRefDestroy(CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexRefDestroy(hTexRef)

cdef CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSurfRefSetArray(hSurfRef, hArray, Flags)

cdef CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSurfRefGetArray(phArray, hSurfRef)

cdef CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef CUresult cuTexObjectDestroy(CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexObjectDestroy(texObject)

cdef CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexObjectGetResourceDesc(pResDesc, texObject)

cdef CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexObjectGetTextureDesc(pTexDesc, texObject)

cdef CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)

cdef CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSurfObjectCreate(pSurfObject, pResDesc)

cdef CUresult cuSurfObjectDestroy(CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSurfObjectDestroy(surfObject)

cdef CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuSurfObjectGetResourceDesc(pResDesc, surfObject)

cdef CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)

cdef CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxEnablePeerAccess(peerContext, Flags)

cdef CUresult cuCtxDisablePeerAccess(CUcontext peerContext) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuCtxDisablePeerAccess(peerContext)

cdef CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)

cdef CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsUnregisterResource(resource)

cdef CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)

cdef CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)

cdef CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)

cdef CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsResourceSetMapFlags_v2(resource, flags)

cdef CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsMapResources(count, resources, hStream)

cdef CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsUnmapResources(count, resources, hStream)

cdef CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGetProcAddress(symbol, pfn, cudaVersion, flags)

cdef CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuModuleGetLoadingMode(mode)

cdef CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags)

cdef CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGetExportTable(ppExportTable, pExportTableId)

cdef CUresult cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuProfilerInitialize(configFile, outputFile, outputMode)

cdef CUresult cuProfilerStart() nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuProfilerStart()

cdef CUresult cuProfilerStop() nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuProfilerStop()

cdef CUresult cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuVDPAUGetDevice(pDevice, vdpDevice, vdpGetProcAddress)

cdef CUresult cuVDPAUCtxCreate(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuVDPAUCtxCreate_v2(pCtx, flags, device, vdpDevice, vdpGetProcAddress)

cdef CUresult cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsVDPAURegisterVideoSurface(pCudaResource, vdpSurface, flags)

cdef CUresult cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsVDPAURegisterOutputSurface(pCudaResource, vdpSurface, flags)

cdef CUresult cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsEGLRegisterImage(pCudaResource, image, flags)

cdef CUresult cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamConsumerConnect(conn, stream)

cdef CUresult cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamConsumerConnectWithFlags(conn, stream, flags)

cdef CUresult cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamConsumerDisconnect(conn)

cdef CUresult cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, timeout)

cdef CUresult cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream)

cdef CUresult cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamProducerConnect(conn, stream, width, height)

cdef CUresult cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamProducerDisconnect(conn)

cdef CUresult cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamProducerPresentFrame(conn, eglframe, pStream)

cdef CUresult cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEGLStreamProducerReturnFrame(conn, eglframe, pStream)

cdef CUresult cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsResourceGetMappedEglFrame(eglFrame, resource, index, mipLevel)

cdef CUresult cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuEventCreateFromEGLSync(phEvent, eglSync, flags)

cdef CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags)

cdef CUresult cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGraphicsGLRegisterImage(pCudaResource, image, target, Flags)

cdef CUresult cuGLGetDevices(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) nogil except ?CUDA_ERROR_NOT_FOUND:
    return ccuda._cuGLGetDevices_v2(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)
