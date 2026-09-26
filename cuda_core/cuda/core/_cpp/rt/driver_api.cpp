// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "driver_api.hpp"
#include <cstddef>
#include <cuda.h>

namespace cuda_core::rt {

// ============================================================================
// CUDA driver function pointers
//
// These are populated by _rt.pyx at module import time using
// function pointers extracted from cuda.bindings.cydriver.__pyx_capi__.
// ============================================================================

decltype(&cuGetErrorName) p_cuGetErrorName = nullptr;
decltype(&cuGetErrorString) p_cuGetErrorString = nullptr;

decltype(&cuDevicePrimaryCtxRetain) p_cuDevicePrimaryCtxRetain = nullptr;
decltype(&cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease = nullptr;
decltype(&cuCtxGetCurrent) p_cuCtxGetCurrent = nullptr;
decltype(&cuCtxSetCurrent) p_cuCtxSetCurrent = nullptr;
decltype(&cuCtxSynchronize) p_cuCtxSynchronize = nullptr;
decltype(&cuCtxGetStreamPriorityRange) p_cuCtxGetStreamPriorityRange = nullptr;
decltype(&cuCtxGetDevice) p_cuCtxGetDevice = nullptr;
decltype(&cuGraphNodeSetParams) p_cuGraphNodeSetParams = nullptr;
decltype(&cuGreenCtxCreate) p_cuGreenCtxCreate = nullptr;
decltype(&cuGreenCtxDestroy) p_cuGreenCtxDestroy = nullptr;
decltype(&cuCtxFromGreenCtx) p_cuCtxFromGreenCtx = nullptr;
decltype(&cuDevResourceGenerateDesc) p_cuDevResourceGenerateDesc = nullptr;

decltype(&cuGreenCtxStreamCreate) p_cuGreenCtxStreamCreate = nullptr;

decltype(&cuStreamCreateWithPriority) p_cuStreamCreateWithPriority = nullptr;
decltype(&cuStreamDestroy) p_cuStreamDestroy = nullptr;
decltype(&cuStreamGetCtx) p_cuStreamGetCtx = nullptr;

decltype(&cuEventCreate) p_cuEventCreate = nullptr;
decltype(&cuEventDestroy) p_cuEventDestroy = nullptr;
decltype(&cuIpcOpenEventHandle) p_cuIpcOpenEventHandle = nullptr;

decltype(&cuDeviceGetCount) p_cuDeviceGetCount = nullptr;

decltype(&cuMemPoolSetAccess) p_cuMemPoolSetAccess = nullptr;
decltype(&cuMemPoolDestroy) p_cuMemPoolDestroy = nullptr;
decltype(&cuMemPoolCreate) p_cuMemPoolCreate = nullptr;
decltype(&cuDeviceGetMemPool) p_cuDeviceGetMemPool = nullptr;
decltype(&cuMemPoolImportFromShareableHandle) p_cuMemPoolImportFromShareableHandle = nullptr;

decltype(&cuMemAllocFromPoolAsync) p_cuMemAllocFromPoolAsync = nullptr;
decltype(&cuMemAllocAsync) p_cuMemAllocAsync = nullptr;
decltype(&cuMemAlloc) p_cuMemAlloc = nullptr;
decltype(&cuMemAllocHost) p_cuMemAllocHost = nullptr;

decltype(&cuMemFreeAsync) p_cuMemFreeAsync = nullptr;
decltype(&cuMemFree) p_cuMemFree = nullptr;
decltype(&cuMemFreeHost) p_cuMemFreeHost = nullptr;

decltype(&cuMemPoolImportPointer) p_cuMemPoolImportPointer = nullptr;

decltype(&cuLibraryLoadFromFile) p_cuLibraryLoadFromFile = nullptr;
decltype(&cuLibraryLoadData) p_cuLibraryLoadData = nullptr;
decltype(&cuLibraryUnload) p_cuLibraryUnload = nullptr;
decltype(&cuLibraryGetKernel) p_cuLibraryGetKernel = nullptr;

// Graph
decltype(&cuGraphDestroy) p_cuGraphDestroy = nullptr;
decltype(&cuGraphInstantiateWithParams) p_cuGraphInstantiateWithParams = nullptr;
decltype(&cuGraphExecUpdate) p_cuGraphExecUpdate = nullptr;
decltype(&cuGraphExecDestroy) p_cuGraphExecDestroy = nullptr;
decltype(&cuUserObjectCreate) p_cuUserObjectCreate = nullptr;
decltype(&cuUserObjectRelease) p_cuUserObjectRelease = nullptr;
decltype(&cuGraphRetainUserObject) p_cuGraphRetainUserObject = nullptr;
decltype(&cuGraphReleaseUserObject) p_cuGraphReleaseUserObject = nullptr;
decltype(&cuGraphNodeFindInClone) p_cuGraphNodeFindInClone = nullptr;
decltype(&cuGraphChildGraphNodeGetGraph) p_cuGraphChildGraphNodeGetGraph = nullptr;

// Linker
decltype(&cuLinkDestroy) p_cuLinkDestroy = nullptr;

// GL interop pointers
decltype(&cuGraphicsUnmapResources) p_cuGraphicsUnmapResources = nullptr;
decltype(&cuGraphicsUnregisterResource) p_cuGraphicsUnregisterResource = nullptr;

decltype(&cuArray3DCreate) p_cuArray3DCreate = nullptr;
decltype(&cuArrayDestroy) p_cuArrayDestroy = nullptr;
decltype(&cuMipmappedArrayCreate) p_cuMipmappedArrayCreate = nullptr;
decltype(&cuMipmappedArrayDestroy) p_cuMipmappedArrayDestroy = nullptr;
decltype(&cuMipmappedArrayGetLevel) p_cuMipmappedArrayGetLevel = nullptr;
decltype(&cuTexObjectCreate) p_cuTexObjectCreate = nullptr;
decltype(&cuTexObjectDestroy) p_cuTexObjectDestroy = nullptr;
decltype(&cuSurfObjectCreate) p_cuSurfObjectCreate = nullptr;
decltype(&cuSurfObjectDestroy) p_cuSurfObjectDestroy = nullptr;

// SM resource split (13.1+ — may be null on older drivers/bindings)
#if CUDA_VERSION >= 13010
decltype(&cuDevSmResourceSplit) p_cuDevSmResourceSplit = nullptr;
#else
void* p_cuDevSmResourceSplit = nullptr;
#endif

// cuMemcpyWithAttributesAsync (13.2+ — may be null on older drivers/bindings)
#if CUDA_VERSION >= 13020
decltype(&cuMemcpyWithAttributesAsync) p_cuMemcpyWithAttributesAsync = nullptr;
#else
void* p_cuMemcpyWithAttributesAsync = nullptr;
#endif

// NVRTC function pointers
decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram = nullptr;

// NVVM function pointers (may be null if NVVM is not available)
NvvmDestroyProgramFn p_nvvmDestroyProgram = nullptr;

// nvJitLink function pointers (may be null if nvJitLink is not available)
NvJitLinkDestroyFn p_nvJitLinkDestroy = nullptr;

// ============================================================================
// SM resource split wrapper
// ============================================================================

CUresult sm_resource_split(CUdevResource* result, unsigned int nbGroups,
                           const CUdevResource* input, CUdevResource* remainder,
                           unsigned int flags, void* groupParams) {
#if CUDA_VERSION >= 13010
    if (!p_cuDevSmResourceSplit) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return p_cuDevSmResourceSplit(
        result, nbGroups, input, remainder, flags,
        static_cast<CU_DEV_SM_RESOURCE_GROUP_PARAMS*>(groupParams));
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

bool has_sm_resource_split() noexcept {
    return p_cuDevSmResourceSplit != nullptr;
}

// ============================================================================
// cuMemcpyWithAttributesAsync wrapper
// ============================================================================

CUresult memcpy_with_attributes_async(CUdeviceptr dst, CUdeviceptr src, size_t size,
                                       void* attr, CUstream hStream) {
#if CUDA_VERSION >= 13020
    if (!p_cuMemcpyWithAttributesAsync) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return p_cuMemcpyWithAttributesAsync(
        dst, src, size, static_cast<CUmemcpyAttributes*>(attr), hStream);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

bool has_memcpy_with_attributes_async() noexcept {
    return p_cuMemcpyWithAttributesAsync != nullptr;
}

}  // namespace cuda_core::rt
