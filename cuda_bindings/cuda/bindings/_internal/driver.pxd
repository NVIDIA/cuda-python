# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from ..cydriver cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef CUresult _cuGetErrorString(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGetErrorName(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuInit(unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDriverGetVersion(int* driverVersion) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGet(CUdevice* device, int ordinal) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetCount(int* count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetName(char* name, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevicePrimaryCtxRelease_v2(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevicePrimaryCtxReset_v2(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxDestroy_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxPushCurrent_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxPopCurrent_v2(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSetCurrent(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetCurrent(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetDevice(CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetFlags(unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSetFlags(unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetId(CUcontext ctx, unsigned long long* ctxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSynchronize() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSetLimit(CUlimit limit, size_t value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetLimit(size_t* pvalue, CUlimit limit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetCacheConfig(CUfunc_cache* pconfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSetCacheConfig(CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxResetPersistingL2Cache() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxAttach(CUcontext* pctx, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxDetach(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSetSharedMemConfig(CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleLoad(CUmodule* module, const char* fname) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleLoadData(CUmodule* module, const void* image) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleUnload(CUmodule hmod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLinkDestroy(CUlinkState state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryUnload(CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetModule(CUmodule* pMod, CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetName(const char** name, CUkernel hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetInfo_v2(size_t* free, size_t* total) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemFree_v2(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAllocHost_v2(void** pp, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemFreeHost(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemHostGetFlags(unsigned int* pFlags, void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void* userData, CUasyncCallbackHandle* callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuIpcCloseMemHandle(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemHostUnregister(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArrayDestroy(CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemBatchDecompressAsync(CUmemDecompressParams* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAddressFree(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemRelease(CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemUnmap(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolDestroy(CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamCreate(CUstream* phStream, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetPriority(CUstream hStream, int* priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetDevice(CUstream hStream, CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetFlags(CUstream hStream, unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetId(CUstream hStream, unsigned long long* streamId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetCtx(CUstream hStream, CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetCtx_v2(CUstream hStream, CUcontext* pCtx, CUgreenCtx* pGreenCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamQuery(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamSynchronize(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamDestroy_v2(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamCopyAttributes(CUstream dst, CUstream src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventCreate(CUevent* phEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventRecord(CUevent hEvent, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventQuery(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventSynchronize(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventDestroy_v2(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDestroyExternalMemory(CUexternalMemory extMem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDestroyExternalSemaphore(CUexternalSemaphore extSem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncGetName(const char** name, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncIsLoaded(CUfunctionLoadingState* state, CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncLoad(CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuParamSeti(CUfunction hfunc, int offset, unsigned int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuParamSetf(CUfunction hfunc, int offset, float value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunch(CUfunction f) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphCreate(CUgraph* phGraph, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGraphMemTrim(CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData, size_t* numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode* dependentNodes, CUgraphEdgeData* edgeData, size_t* numDependentNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphDestroyNode(CUgraphNode hNode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecDestroy(CUgraphExec hGraphExec) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphDestroy(CUgraph hGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuUserObjectRetain(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuUserObjectRelease(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphAddNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefCreate(CUtexref* pTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexRefDestroy(CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexObjectDestroy(CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSurfObjectDestroy(CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTensorMapEncodeIm2colWide(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxDisablePeerAccess(CUcontext peerContext) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxDestroy(CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxFromGreenCtx(CUcontext* pContext, CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetDevResource(CUdevice device, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetDevResource(CUcontext hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, unsigned int minCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, unsigned int nbResources) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx* phCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx greenCtx, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLogsRegisterCallback(CUlogsCallback callbackFunc, void* userData, CUlogsCallbackHandle* callback_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLogsUnregisterCallback(CUlogsCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLogsCurrent(CUlogIterator* iterator_out, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLogsDumpToFile(CUlogIterator* iterator, const char* pathToFile, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLogsDumpToMemory(CUlogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessGetRestoreThreadId(int pid, int* tid) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessGetState(int pid, CUprocessState* state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLCtxCreate_v2(CUcontext* pCtx, unsigned int Flags, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLInit() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLRegisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLMapBufferObject_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLUnmapBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLUnregisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLMapBufferObjectAsync_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuProfilerStart() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuProfilerStop() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuVDPAUCtxCreate_v2(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCtxSynchronize_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyBatchAsync_v2(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3DBatchAsync_v2(size_t numOps, CUDA_MEMCPY3D_BATCH_OP* opList, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetDefaultMemPool(CUmemoryPool* pool_out, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemSetMemPool(CUmemLocation* location, CUmemAllocationType type, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastBindMem_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMulticastBindAddr_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemDiscardBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemDiscardAndPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int* nodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetToolsId(CUgraphNode hNode, unsigned long long* toolsNodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphGetId(CUgraph hGraph, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuDevSmResourceSplit(CUdevResource* result, unsigned int nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, CU_DEV_SM_RESOURCE_GROUP_PARAMS* groupParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long* greenCtxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamGetDevResource(CUstream hStream, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuKernelGetParamCount(CUkernel kernel, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, CUmemcpyAttributes* attr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuMemcpy3DWithAttributesAsync(CUDA_MEMCPY3D_BATCH_OP* op, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamBeginCaptureToCig(CUstream hStream, CUstreamCigCaptureParams* streamCigCaptureParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuStreamEndCaptureToCig(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuFuncGetParamCount(CUfunction func, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuLaunchHostFunc_v2(CUstream hStream, CUhostFn fn, void* userData, unsigned int syncMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuGraphNodeGetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpRegisterStartCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpRegisterCompleteCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpDeregisterStartCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
cdef CUresult _cuCoredumpDeregisterCompleteCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil
