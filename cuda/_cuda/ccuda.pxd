# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda.ccuda cimport *

cdef CUresult _cuGetErrorString(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGetErrorName(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuInit(unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDriverGetVersion(int* driverVersion) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGet(CUdevice* device, int ordinal) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetCount(int* count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetName(char* name, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceTotalMem_v2(size_t* numbytes, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format pformat, unsigned numChannels, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDevicePrimaryCtxRelease_v2(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDevicePrimaryCtxReset_v2(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType typename, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxDestroy_v2(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxPushCurrent_v2(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxPopCurrent_v2(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxSetCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetDevice(CUdevice* device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetFlags(unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxSynchronize() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxSetLimit(CUlimit limit, size_t value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetLimit(size_t* pvalue, CUlimit limit) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetCacheConfig(CUfunc_cache* pconfig) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxSetCacheConfig(CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxSetSharedMemConfig(CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxResetPersistingL2Cache() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType typename) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxAttach(CUcontext* pctx, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxDetach(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleLoad(CUmodule* module, const char* fname) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleLoadData(CUmodule* module, const void* image) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleUnload(CUmodule hmod) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* numbytes, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLinkAddData_v2(CUlinkState state, CUjitInputType typename, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLinkAddFile_v2(CUlinkState state, CUjitInputType typename, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLinkDestroy(CUlinkState state) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetInfo_v2(size_t* free, size_t* total) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemFree_v2(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAllocHost_v2(void** pp, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemFreeHost(void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemHostGetFlags(unsigned int* pFlags, void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetPCIBusId(char* pciBusId, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuIpcCloseMemHandle(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemHostUnregister(void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArrayDestroy(CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAddressFree(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemRelease(CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemUnmap(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolDestroy(CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamCreate(CUstream* phStream, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetPriority(CUstream hStream, int* priority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetFlags(CUstream hStream, unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetCtx(CUstream hStream, CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamQuery(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamSynchronize(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamDestroy_v2(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamCopyAttributes(CUstream dst, CUstream src) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventCreate(CUevent* phEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventRecord(CUevent hEvent, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventQuery(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventSynchronize(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventDestroy_v2(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDestroyExternalMemory(CUexternalMemory extMem) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDestroyExternalSemaphore(CUexternalSemaphore extSem) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuFuncSetSharedSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuParamSeti(CUfunction hfunc, int offset, unsigned int value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuParamSetf(CUfunction hfunc, int offset, float value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunch(CUfunction f) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphCreate(CUgraph* phGraph, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGraphMemTrim(CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* typename) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, size_t* numEdges) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphDestroyNode(CUgraphNode hNode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphInstantiate_v2(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecDestroy(CUgraphExec hGraphExec) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphDestroy(CUgraph hGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuUserObjectRetain(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuUserObjectRelease(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefCreate(CUtexref* pTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexRefDestroy(CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexObjectDestroy(CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSurfObjectDestroy(CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuCtxDisablePeerAccess(CUcontext peerContext) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuProfilerStart() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuProfilerStop() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuVDPAUCtxCreate_v2(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult _cuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) nogil except ?CUDA_ERROR_NOT_FOUND
