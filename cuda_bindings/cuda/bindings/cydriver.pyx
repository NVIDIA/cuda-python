# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from ._internal cimport driver as _driver

cdef CUresult cuGetErrorString(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGetErrorString(error, pStr)


cdef CUresult cuGetErrorName(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGetErrorName(error, pStr)


cdef CUresult cuInit(unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuInit(Flags)


cdef CUresult cuDriverGetVersion(int* driverVersion) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDriverGetVersion(driverVersion)


cdef CUresult cuDeviceGet(CUdevice* device, int ordinal) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGet(device, ordinal)


cdef CUresult cuDeviceGetCount(int* count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetCount(count)


cdef CUresult cuDeviceGetName(char* name, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetName(name, len, dev)


cdef CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetUuid_v2(uuid, dev)


cdef CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetLuid(luid, deviceNodeMask, dev)


cdef CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceTotalMem_v2(bytes, dev)


cdef CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev)


cdef CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetAttribute(pi, attrib, dev)


cdef CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags)


cdef CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceSetMemPool(dev, pool)


cdef CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetMemPool(pool, dev)


cdef CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetDefaultMemPool(pool_out, dev)


cdef CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetExecAffinitySupport(pi, type, dev)


cdef CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFlushGPUDirectRDMAWrites(target, scope)


cdef CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetProperties(prop, dev)


cdef CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceComputeCapability(major, minor, dev)


cdef CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevicePrimaryCtxRetain(pctx, dev)


cdef CUresult cuDevicePrimaryCtxRelease(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevicePrimaryCtxRelease_v2(dev)


cdef CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevicePrimaryCtxSetFlags_v2(dev, flags)


cdef CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevicePrimaryCtxGetState(dev, flags, active)


cdef CUresult cuDevicePrimaryCtxReset(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevicePrimaryCtxReset_v2(dev)


cdef CUresult cuCtxCreate(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxCreate_v4(pctx, ctxCreateParams, flags, dev)


cdef CUresult cuCtxDestroy(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxDestroy_v2(ctx)


cdef CUresult cuCtxPushCurrent(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxPushCurrent_v2(ctx)


cdef CUresult cuCtxPopCurrent(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxPopCurrent_v2(pctx)


cdef CUresult cuCtxSetCurrent(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSetCurrent(ctx)


cdef CUresult cuCtxGetCurrent(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetCurrent(pctx)


cdef CUresult cuCtxGetDevice(CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetDevice(device)


cdef CUresult cuCtxGetFlags(unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetFlags(flags)


cdef CUresult cuCtxSetFlags(unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSetFlags(flags)


cdef CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetId(ctx, ctxId)


cdef CUresult cuCtxSynchronize() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSynchronize()


cdef CUresult cuCtxSetLimit(CUlimit limit, size_t value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSetLimit(limit, value)


cdef CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetLimit(pvalue, limit)


cdef CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetCacheConfig(pconfig)


cdef CUresult cuCtxSetCacheConfig(CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSetCacheConfig(config)


cdef CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetApiVersion(ctx, version)


cdef CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)


cdef CUresult cuCtxResetPersistingL2Cache() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxResetPersistingL2Cache()


cdef CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetExecAffinity(pExecAffinity, type)


cdef CUresult cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxRecordEvent(hCtx, hEvent)


cdef CUresult cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxWaitEvent(hCtx, hEvent)


cdef CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxAttach(pctx, flags)


cdef CUresult cuCtxDetach(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxDetach(ctx)


cdef CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetSharedMemConfig(pConfig)


cdef CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSetSharedMemConfig(config)


cdef CUresult cuModuleLoad(CUmodule* module, const char* fname) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleLoad(module, fname)


cdef CUresult cuModuleLoadData(CUmodule* module, const void* image) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleLoadData(module, image)


cdef CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleLoadDataEx(module, image, numOptions, options, optionValues)


cdef CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleLoadFatBinary(module, fatCubin)


cdef CUresult cuModuleUnload(CUmodule hmod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleUnload(hmod)


cdef CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetLoadingMode(mode)


cdef CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetFunction(hfunc, hmod, name)


cdef CUresult cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetFunctionCount(count, mod)


cdef CUresult cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleEnumerateFunctions(functions, numFunctions, mod)


cdef CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetGlobal_v2(dptr, bytes, hmod, name)


cdef CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLinkCreate_v2(numOptions, options, optionValues, stateOut)


cdef CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues)


cdef CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues)


cdef CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLinkComplete(state, cubinOut, sizeOut)


cdef CUresult cuLinkDestroy(CUlinkState state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLinkDestroy(state)


cdef CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetTexRef(pTexRef, hmod, name)


cdef CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuModuleGetSurfRef(pSurfRef, hmod, name)


cdef CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult cuLibraryUnload(CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryUnload(library)


cdef CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetKernel(pKernel, library, name)


cdef CUresult cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetKernelCount(count, lib)


cdef CUresult cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryEnumerateKernels(kernels, numKernels, lib)


cdef CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetModule(pMod, library)


cdef CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetFunction(pFunc, kernel)


cdef CUresult cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetLibrary(pLib, kernel)


cdef CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetGlobal(dptr, bytes, library, name)


cdef CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetManaged(dptr, bytes, library, name)


cdef CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLibraryGetUnifiedFunction(fptr, library, symbol)


cdef CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetAttribute(pi, attrib, kernel, dev)


cdef CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelSetAttribute(attrib, val, kernel, dev)


cdef CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelSetCacheConfig(kernel, config, dev)


cdef CUresult cuKernelGetName(const char** name, CUkernel hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetName(name, hfunc)


cdef CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetParamInfo(kernel, paramIndex, paramOffset, paramSize)


cdef CUresult cuMemGetInfo(size_t* free, size_t* total) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetInfo_v2(free, total)


cdef CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAlloc_v2(dptr, bytesize)


cdef CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)


cdef CUresult cuMemFree(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemFree_v2(dptr)


cdef CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetAddressRange_v2(pbase, psize, dptr)


cdef CUresult cuMemAllocHost(void** pp, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAllocHost_v2(pp, bytesize)


cdef CUresult cuMemFreeHost(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemFreeHost(p)


cdef CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemHostAlloc(pp, bytesize, Flags)


cdef CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemHostGetDevicePointer_v2(pdptr, p, Flags)


cdef CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemHostGetFlags(pFlags, p)


cdef CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAllocManaged(dptr, bytesize, flags)


cdef CUresult cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void* userData, CUasyncCallbackHandle* callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)


cdef CUresult cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceUnregisterAsyncNotification(device, callback)


cdef CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetByPCIBusId(dev, pciBusId)


cdef CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetPCIBusId(pciBusId, len, dev)


cdef CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuIpcGetEventHandle(pHandle, event)


cdef CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuIpcOpenEventHandle(phEvent, handle)


cdef CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuIpcGetMemHandle(pHandle, dptr)


cdef CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuIpcOpenMemHandle_v2(pdptr, handle, Flags)


cdef CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuIpcCloseMemHandle(dptr)


cdef CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemHostRegister_v2(p, bytesize, Flags)


cdef CUresult cuMemHostUnregister(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemHostUnregister(p)


cdef CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy(dst, src, ByteCount)


cdef CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)


cdef CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)


cdef CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)


cdef CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)


cdef CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)


cdef CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)


cdef CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)


cdef CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)


cdef CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)


cdef CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy2D_v2(pCopy)


cdef CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy2DUnaligned_v2(pCopy)


cdef CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3D_v2(pCopy)


cdef CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3DPeer(pCopy)


cdef CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyAsync(dst, src, ByteCount, hStream)


cdef CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)


cdef CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)


cdef CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)


cdef CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)


cdef CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)


cdef CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)


cdef CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy2DAsync_v2(pCopy, hStream)


cdef CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3DAsync_v2(pCopy, hStream)


cdef CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3DPeerAsync(pCopy, hStream)


cdef CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD8_v2(dstDevice, uc, N)


cdef CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD16_v2(dstDevice, us, N)


cdef CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD32_v2(dstDevice, ui, N)


cdef CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)


cdef CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)


cdef CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)


cdef CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD8Async(dstDevice, uc, N, hStream)


cdef CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD16Async(dstDevice, us, N, hStream)


cdef CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD32Async(dstDevice, ui, N, hStream)


cdef CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)


cdef CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)


cdef CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)


cdef CUresult cuArrayCreate(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayCreate_v2(pHandle, pAllocateArray)


cdef CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)


cdef CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayGetSparseProperties(sparseProperties, array)


cdef CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap)


cdef CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayGetMemoryRequirements(memoryRequirements, array, device)


cdef CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)


cdef CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayGetPlane(pPlaneArray, hArray, planeIdx)


cdef CUresult cuArrayDestroy(CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArrayDestroy(hArray)


cdef CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArray3DCreate_v2(pHandle, pAllocateArray)


cdef CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)


cdef CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)


cdef CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)


cdef CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMipmappedArrayDestroy(hMipmappedArray)


cdef CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags)


cdef CUresult cuMemBatchDecompressAsync(CUmemDecompressParams* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemBatchDecompressAsync(paramsArray, count, flags, errorIndex, stream)


cdef CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAddressReserve(ptr, size, alignment, addr, flags)


cdef CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAddressFree(ptr, size)


cdef CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemCreate(handle, size, prop, flags)


cdef CUresult cuMemRelease(CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemRelease(handle)


cdef CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemMap(ptr, size, offset, handle, flags)


cdef CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemMapArrayAsync(mapInfoList, count, hStream)


cdef CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemUnmap(ptr, size)


cdef CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemSetAccess(ptr, size, desc, count)


cdef CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetAccess(flags, location, ptr)


cdef CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags)


cdef CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemImportFromShareableHandle(handle, osHandle, shHandleType)


cdef CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetAllocationGranularity(granularity, prop, option)


cdef CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetAllocationPropertiesFromHandle(prop, handle)


cdef CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemRetainAllocationHandle(handle, addr)


cdef CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemFreeAsync(dptr, hStream)


cdef CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAllocAsync(dptr, bytesize, hStream)


cdef CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolTrimTo(pool, minBytesToKeep)


cdef CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolSetAttribute(pool, attr, value)


cdef CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolGetAttribute(pool, attr, value)


cdef CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolSetAccess(pool, map, count)


cdef CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolGetAccess(flags, memPool, location)


cdef CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolCreate(pool, poolProps)


cdef CUresult cuMemPoolDestroy(CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolDestroy(pool)


cdef CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream)


cdef CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags)


cdef CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags)


cdef CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolExportPointer(shareData_out, ptr)


cdef CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPoolImportPointer(ptr_out, pool, shareData)


cdef CUresult cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastCreate(mcHandle, prop)


cdef CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastAddDevice(mcHandle, dev)


cdef CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastBindAddr(mcHandle, mcOffset, memptr, size, flags)


cdef CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastUnbind(mcHandle, dev, mcOffset, size)


cdef CUresult cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastGetGranularity(granularity, prop, option)


cdef CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuPointerGetAttribute(data, attribute, ptr)


cdef CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream)


cdef CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemAdvise_v2(devPtr, count, advice, location)


cdef CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)


cdef CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)


cdef CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuPointerSetAttribute(value, attribute, ptr)


cdef CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuPointerGetAttributes(numAttributes, attributes, data, ptr)


cdef CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamCreate(phStream, Flags)


cdef CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamCreateWithPriority(phStream, flags, priority)


cdef CUresult cuStreamGetPriority(CUstream hStream, int* priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetPriority(hStream, priority)


cdef CUresult cuStreamGetDevice(CUstream hStream, CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetDevice(hStream, device)


cdef CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetFlags(hStream, flags)


cdef CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetId(hStream, streamId)


cdef CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetCtx(hStream, pctx)


cdef CUresult cuStreamGetCtx_v2(CUstream hStream, CUcontext* pCtx, CUgreenCtx* pGreenCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetCtx_v2(hStream, pCtx, pGreenCtx)


cdef CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamWaitEvent(hStream, hEvent, Flags)


cdef CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamAddCallback(hStream, callback, userData, flags)


cdef CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamBeginCapture_v2(hStream, mode)


cdef CUresult cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamBeginCaptureToGraph(hStream, hGraph, dependencies, dependencyData, numDependencies, mode)


cdef CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuThreadExchangeStreamCaptureMode(mode)


cdef CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamEndCapture(hStream, phGraph)


cdef CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamIsCapturing(hStream, captureStatus)


cdef CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)


cdef CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, dependencyData, numDependencies, flags)


cdef CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamAttachMemAsync(hStream, dptr, length, flags)


cdef CUresult cuStreamQuery(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamQuery(hStream)


cdef CUresult cuStreamSynchronize(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamSynchronize(hStream)


cdef CUresult cuStreamDestroy(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamDestroy_v2(hStream)


cdef CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamCopyAttributes(dst, src)


cdef CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetAttribute(hStream, attr, value_out)


cdef CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamSetAttribute(hStream, attr, value)


cdef CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventCreate(phEvent, Flags)


cdef CUresult cuEventRecord(CUevent hEvent, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventRecord(hEvent, hStream)


cdef CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventRecordWithFlags(hEvent, hStream, flags)


cdef CUresult cuEventQuery(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventQuery(hEvent)


cdef CUresult cuEventSynchronize(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventSynchronize(hEvent)


cdef CUresult cuEventDestroy(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventDestroy_v2(hEvent)


cdef CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd)


cdef CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuImportExternalMemory(extMem_out, memHandleDesc)


cdef CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)


cdef CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)


cdef CUresult cuDestroyExternalMemory(CUexternalMemory extMem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDestroyExternalMemory(extMem)


cdef CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuImportExternalSemaphore(extSem_out, semHandleDesc)


cdef CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDestroyExternalSemaphore(extSem)


cdef CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamWaitValue32_v2(stream, addr, value, flags)


cdef CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamWaitValue64_v2(stream, addr, value, flags)


cdef CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamWriteValue32_v2(stream, addr, value, flags)


cdef CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamWriteValue64_v2(stream, addr, value, flags)


cdef CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamBatchMemOp_v2(stream, count, paramArray, flags)


cdef CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncGetAttribute(pi, attrib, hfunc)


cdef CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncSetAttribute(hfunc, attrib, value)


cdef CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncSetCacheConfig(hfunc, config)


cdef CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncGetModule(hmod, hfunc)


cdef CUresult cuFuncGetName(const char** name, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncGetName(name, hfunc)


cdef CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncGetParamInfo(func, paramIndex, paramOffset, paramSize)


cdef CUresult cuFuncIsLoaded(CUfunctionLoadingState* state, CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncIsLoaded(state, function)


cdef CUresult cuFuncLoad(CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncLoad(function)


cdef CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)


cdef CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchKernelEx(config, f, kernelParams, extra)


cdef CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)


cdef CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)


cdef CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchHostFunc(hStream, fn, userData)


cdef CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncSetBlockShape(hfunc, x, y, z)


cdef CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncSetSharedSize(hfunc, bytes)


cdef CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuParamSetSize(hfunc, numbytes)


cdef CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuParamSeti(hfunc, offset, value)


cdef CUresult cuParamSetf(CUfunction hfunc, int offset, float value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuParamSetf(hfunc, offset, value)


cdef CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuParamSetv(hfunc, offset, ptr, numbytes)


cdef CUresult cuLaunch(CUfunction f) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunch(f)


cdef CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchGrid(f, grid_width, grid_height)


cdef CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchGridAsync(f, grid_width, grid_height, hStream)


cdef CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuParamSetTexRef(hfunc, texunit, hTexRef)


cdef CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncSetSharedMemConfig(hfunc, config)


cdef CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphCreate(phGraph, flags)


cdef CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphKernelNodeGetParams_v2(hNode, nodeParams)


cdef CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphKernelNodeSetParams_v2(hNode, nodeParams)


cdef CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)


cdef CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemcpyNodeGetParams(hNode, nodeParams)


cdef CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemcpyNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)


cdef CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemsetNodeGetParams(hNode, nodeParams)


cdef CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemsetNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphHostNodeGetParams(hNode, nodeParams)


cdef CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphHostNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph)


cdef CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphChildGraphNodeGetGraph(hNode, phGraph)


cdef CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)


cdef CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphEventRecordNodeGetEvent(hNode, event_out)


cdef CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphEventRecordNodeSetEvent(hNode, event)


cdef CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphEventWaitNodeGetEvent(hNode, event_out)


cdef CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphEventWaitNodeSetEvent(hNode, event)


cdef CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)


cdef CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)


cdef CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)


cdef CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphBatchMemOpNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemAllocNodeGetParams(hNode, params_out)


cdef CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr)


cdef CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphMemFreeNodeGetParams(hNode, dptr_out)


cdef CUresult cuDeviceGraphMemTrim(CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGraphMemTrim(device)


cdef CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetGraphMemAttribute(device, attr, value)


cdef CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceSetGraphMemAttribute(device, attr, value)


cdef CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphClone(phGraphClone, originalGraph)


cdef CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)


cdef CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetType(hNode, type)


cdef CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphGetNodes(hGraph, nodes, numNodes)


cdef CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)


cdef CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphGetEdges_v2(hGraph, from_, to, edgeData, numEdges)


cdef CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData, size_t* numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetDependencies_v2(hNode, dependencies, edgeData, numDependencies)


cdef CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, CUgraphEdgeData* edgeData, size_t* numDependentNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, edgeData, numDependentNodes)


cdef CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddDependencies_v2(hGraph, from_, to, edgeData, numDependencies)


cdef CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphRemoveDependencies_v2(hGraph, from_, to, edgeData, numDependencies)


cdef CUresult cuGraphDestroyNode(CUgraphNode hNode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphDestroyNode(hNode)


cdef CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags)


cdef CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams)


cdef CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecGetFlags(hGraphExec, flags)


cdef CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)


cdef CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)


cdef CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph)


cdef CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)


cdef CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)


cdef CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)


cdef CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)


cdef CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphUpload(hGraphExec, hStream)


cdef CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphLaunch(hGraphExec, hStream)


cdef CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecDestroy(hGraphExec)


cdef CUresult cuGraphDestroy(CUgraph hGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphDestroy(hGraph)


cdef CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo)


cdef CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphKernelNodeCopyAttributes(dst, src)


cdef CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphKernelNodeGetAttribute(hNode, attr, value_out)


cdef CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphKernelNodeSetAttribute(hNode, attr, value)


cdef CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphDebugDotPrint(hGraph, path, flags)


cdef CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)


cdef CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuUserObjectRetain(object, count)


cdef CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuUserObjectRelease(object, count)


cdef CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphRetainUserObject(graph, object, count, flags)


cdef CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphReleaseUserObject(graph, object, count)


cdef CUresult cuGraphAddNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)


cdef CUresult cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeSetParams(hNode, nodeParams)


cdef CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecNodeSetParams(hGraphExec, hNode, nodeParams)


cdef CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphConditionalHandleCreate(pHandle_out, hGraph, ctx, defaultLaunchValue, flags)


cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)


cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)


cdef CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)


cdef CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)


cdef CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)


cdef CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxPotentialClusterSize(clusterSize, func, config)


cdef CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuOccupancyMaxActiveClusters(numClusters, func, config)


cdef CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetArray(hTexRef, hArray, Flags)


cdef CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)


cdef CUresult cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes)


cdef CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)


cdef CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)


cdef CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetAddressMode(hTexRef, dim, am)


cdef CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetFilterMode(hTexRef, fm)


cdef CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetMipmapFilterMode(hTexRef, fm)


cdef CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetMipmapLevelBias(hTexRef, bias)


cdef CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)


cdef CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)


cdef CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetBorderColor(hTexRef, pBorderColor)


cdef CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefSetFlags(hTexRef, Flags)


cdef CUresult cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetAddress_v2(pdptr, hTexRef)


cdef CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetArray(phArray, hTexRef)


cdef CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)


cdef CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetAddressMode(pam, hTexRef, dim)


cdef CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetFilterMode(pfm, hTexRef)


cdef CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)


cdef CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetMipmapFilterMode(pfm, hTexRef)


cdef CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetMipmapLevelBias(pbias, hTexRef)


cdef CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)


cdef CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)


cdef CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetBorderColor(pBorderColor, hTexRef)


cdef CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefGetFlags(pFlags, hTexRef)


cdef CUresult cuTexRefCreate(CUtexref* pTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefCreate(pTexRef)


cdef CUresult cuTexRefDestroy(CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexRefDestroy(hTexRef)


cdef CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSurfRefSetArray(hSurfRef, hArray, Flags)


cdef CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSurfRefGetArray(phArray, hSurfRef)


cdef CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)


cdef CUresult cuTexObjectDestroy(CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexObjectDestroy(texObject)


cdef CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexObjectGetResourceDesc(pResDesc, texObject)


cdef CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexObjectGetTextureDesc(pTexDesc, texObject)


cdef CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)


cdef CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSurfObjectCreate(pSurfObject, pResDesc)


cdef CUresult cuSurfObjectDestroy(CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSurfObjectDestroy(surfObject)


cdef CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuSurfObjectGetResourceDesc(pResDesc, surfObject)


cdef CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult cuTensorMapEncodeIm2colWide(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTensorMapEncodeIm2colWide(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill)


cdef CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuTensorMapReplaceAddress(tensorMap, globalAddress)


cdef CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)


cdef CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxEnablePeerAccess(peerContext, Flags)


cdef CUresult cuCtxDisablePeerAccess(CUcontext peerContext) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxDisablePeerAccess(peerContext)


cdef CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)


cdef CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsUnregisterResource(resource)


cdef CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)


cdef CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)


cdef CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)


cdef CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsResourceSetMapFlags_v2(resource, flags)


cdef CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsMapResources(count, resources, hStream)


cdef CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsUnmapResources(count, resources, hStream)


cdef CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus)


cdef CUresult cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpGetAttribute(attrib, value, size)


cdef CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpGetAttributeGlobal(attrib, value, size)


cdef CUresult cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpSetAttribute(attrib, value, size)


cdef CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpSetAttributeGlobal(attrib, value, size)


cdef CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGetExportTable(ppExportTable, pExportTableId)


cdef CUresult cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxCreate(phCtx, desc, dev, flags)


cdef CUresult cuGreenCtxDestroy(CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxDestroy(hCtx)


cdef CUresult cuCtxFromGreenCtx(CUcontext* pContext, CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxFromGreenCtx(pContext, hCtx)


cdef CUresult cuDeviceGetDevResource(CUdevice device, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetDevResource(device, resource, type)


cdef CUresult cuCtxGetDevResource(CUcontext hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetDevResource(hCtx, resource, type)


cdef CUresult cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxGetDevResource(hCtx, resource, type)


cdef CUresult cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, unsigned int minCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevSmResourceSplitByCount(result, nbGroups, input, remainder, flags, minCount)


cdef CUresult cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, unsigned int nbResources) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevResourceGenerateDesc(phDesc, resources, nbResources)


cdef CUresult cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxRecordEvent(hCtx, hEvent)


cdef CUresult cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxWaitEvent(hCtx, hEvent)


cdef CUresult cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx* phCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetGreenCtx(hStream, phCtx)


cdef CUresult cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx greenCtx, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxStreamCreate(phStream, greenCtx, flags, priority)


cdef CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void* userData, CUlogsCallbackHandle* callback_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLogsRegisterCallback(callbackFunc, userData, callback_out)


cdef CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLogsUnregisterCallback(callback)


cdef CUresult cuLogsCurrent(CUlogIterator* iterator_out, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLogsCurrent(iterator_out, flags)


cdef CUresult cuLogsDumpToFile(CUlogIterator* iterator, const char* pathToFile, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLogsDumpToFile(iterator, pathToFile, flags)


cdef CUresult cuLogsDumpToMemory(CUlogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLogsDumpToMemory(iterator, buffer, size, flags)


cdef CUresult cuCheckpointProcessGetRestoreThreadId(int pid, int* tid) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessGetRestoreThreadId(pid, tid)


cdef CUresult cuCheckpointProcessGetState(int pid, CUprocessState* state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessGetState(pid, state)


cdef CUresult cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessLock(pid, args)


cdef CUresult cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessCheckpoint(pid, args)


cdef CUresult cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessRestore(pid, args)


cdef CUresult cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCheckpointProcessUnlock(pid, args)


cdef CUresult cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsEGLRegisterImage(pCudaResource, image, flags)


cdef CUresult cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamConsumerConnect(conn, stream)


cdef CUresult cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamConsumerConnectWithFlags(conn, stream, flags)


cdef CUresult cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamConsumerDisconnect(conn)


cdef CUresult cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, timeout)


cdef CUresult cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream)


cdef CUresult cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamProducerConnect(conn, stream, width, height)


cdef CUresult cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamProducerDisconnect(conn)


cdef CUresult cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamProducerPresentFrame(conn, eglframe, pStream)


cdef CUresult cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEGLStreamProducerReturnFrame(conn, eglframe, pStream)


cdef CUresult cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsResourceGetMappedEglFrame(eglFrame, resource, index, mipLevel)


cdef CUresult cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuEventCreateFromEGLSync(phEvent, eglSync, flags)


cdef CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags)


cdef CUresult cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsGLRegisterImage(pCudaResource, image, target, Flags)


cdef CUresult cuGLGetDevices(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLGetDevices_v2(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)


cdef CUresult cuGLCtxCreate(CUcontext* pCtx, unsigned int Flags, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLCtxCreate_v2(pCtx, Flags, device)


cdef CUresult cuGLInit() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLInit()


cdef CUresult cuGLRegisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLRegisterBufferObject(buffer)


cdef CUresult cuGLMapBufferObject(CUdeviceptr* dptr, size_t* size, GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLMapBufferObject_v2(dptr, size, buffer)


cdef CUresult cuGLUnmapBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLUnmapBufferObject(buffer)


cdef CUresult cuGLUnregisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLUnregisterBufferObject(buffer)


cdef CUresult cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLSetBufferObjectMapFlags(buffer, Flags)


cdef CUresult cuGLMapBufferObjectAsync(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLMapBufferObjectAsync_v2(dptr, size, buffer, hStream)


cdef CUresult cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGLUnmapBufferObjectAsync(buffer, hStream)


cdef CUresult cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuProfilerInitialize(configFile, outputFile, outputMode)


cdef CUresult cuProfilerStart() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuProfilerStart()


cdef CUresult cuProfilerStop() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuProfilerStop()


cdef CUresult cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuVDPAUGetDevice(pDevice, vdpDevice, vdpGetProcAddress)


cdef CUresult cuVDPAUCtxCreate(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuVDPAUCtxCreate_v2(pCtx, flags, device, vdpDevice, vdpGetProcAddress)


cdef CUresult cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsVDPAURegisterVideoSurface(pCudaResource, vdpSurface, flags)


cdef CUresult cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphicsVDPAURegisterOutputSurface(pCudaResource, vdpSurface, flags)


cdef CUresult cuDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetHostAtomicCapabilities(capabilities, operations, count, dev)


cdef CUresult cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxGetDevice_v2(device, ctx)


cdef CUresult cuCtxSynchronize_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCtxSynchronize_v2(ctx)


cdef CUresult cuMemcpyBatchAsync(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyBatchAsync_v2(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream)


cdef CUresult cuMemcpy3DBatchAsync(size_t numOps, CUDA_MEMCPY3D_BATCH_OP* opList, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3DBatchAsync_v2(numOps, opList, flags, hStream)


cdef CUresult cuMemGetDefaultMemPool(CUmemoryPool* pool_out, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetDefaultMemPool(pool_out, location, type)


cdef CUresult cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemGetMemPool(pool, location, type)


cdef CUresult cuMemSetMemPool(CUmemLocation* location, CUmemAllocationType type, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemSetMemPool(location, type, pool)


cdef CUresult cuMemPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult cuMemDiscardBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemDiscardBatchAsync(dptrs, sizes, count, flags, hStream)


cdef CUresult cuMemDiscardAndPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult cuDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)


cdef CUresult cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long* greenCtxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGreenCtxGetId(greenCtx, greenCtxId)


cdef CUresult cuMulticastBindMem_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastBindMem_v2(mcHandle, dev, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult cuMulticastBindAddr_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMulticastBindAddr_v2(mcHandle, dev, mcOffset, memptr, size, flags)


cdef CUresult cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetContainingGraph(hNode, phGraph)


cdef CUresult cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int* nodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetLocalId(hNode, nodeId)


cdef CUresult cuGraphNodeGetToolsId(CUgraphNode hNode, unsigned long long* toolsNodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetToolsId(hNode, toolsNodeId)


cdef CUresult cuGraphGetId(CUgraph hGraph, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphGetId(hGraph, graphId)


cdef CUresult cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphExecGetId(hGraphExec, graphId)


cdef CUresult cuDevSmResourceSplit(CUdevResource* result, unsigned int nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, CU_DEV_SM_RESOURCE_GROUP_PARAMS* groupParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)


cdef CUresult cuStreamGetDevResource(CUstream hStream, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamGetDevResource(hStream, resource, type)


cdef CUresult cuKernelGetParamCount(CUkernel kernel, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuKernelGetParamCount(kernel, paramCount)


cdef CUresult cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, CUmemcpyAttributes* attr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpyWithAttributesAsync(dst, src, size, attr, hStream)


cdef CUresult cuMemcpy3DWithAttributesAsync(CUDA_MEMCPY3D_BATCH_OP* op, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuMemcpy3DWithAttributesAsync(op, flags, hStream)


cdef CUresult cuStreamBeginCaptureToCig(CUstream hStream, CUstreamCigCaptureParams* streamCigCaptureParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamBeginCaptureToCig(hStream, streamCigCaptureParams)


cdef CUresult cuStreamEndCaptureToCig(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuStreamEndCaptureToCig(hStream)


cdef CUresult cuFuncGetParamCount(CUfunction func, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuFuncGetParamCount(func, paramCount)


cdef CUresult cuLaunchHostFunc_v2(CUstream hStream, CUhostFn fn, void* userData, unsigned int syncMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuLaunchHostFunc_v2(hStream, fn, userData, syncMode)


cdef CUresult cuGraphNodeGetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuGraphNodeGetParams(hNode, nodeParams)


cdef CUresult cuCoredumpRegisterStartCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpRegisterStartCallback(callback, userData, callbackOut)


cdef CUresult cuCoredumpRegisterCompleteCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpRegisterCompleteCallback(callback, userData, callbackOut)


cdef CUresult cuCoredumpDeregisterStartCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpDeregisterStartCallback(callback)


cdef CUresult cuCoredumpDeregisterCompleteCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    return _driver._cuCoredumpDeregisterCompleteCallback(callback)
