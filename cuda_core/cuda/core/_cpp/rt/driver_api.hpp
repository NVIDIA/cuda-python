// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>

namespace cuda_core::rt {

// ============================================================================
// CUDA driver function pointers
//
// These are populated by _rt.pyx at module import time using
// function pointers extracted from cuda.bindings.cydriver.__pyx_capi__.
// ============================================================================

extern decltype(&cuGetErrorName) p_cuGetErrorName;
extern decltype(&cuGetErrorString) p_cuGetErrorString;

extern decltype(&cuDevicePrimaryCtxRetain) p_cuDevicePrimaryCtxRetain;
extern decltype(&cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease;
extern decltype(&cuCtxGetCurrent) p_cuCtxGetCurrent;
extern decltype(&cuCtxSetCurrent) p_cuCtxSetCurrent;
extern decltype(&cuCtxSynchronize) p_cuCtxSynchronize;
extern decltype(&cuCtxGetStreamPriorityRange) p_cuCtxGetStreamPriorityRange;
extern decltype(&cuCtxGetDevice) p_cuCtxGetDevice;
extern decltype(&cuGraphNodeSetParams) p_cuGraphNodeSetParams;
extern decltype(&cuGreenCtxCreate) p_cuGreenCtxCreate;
extern decltype(&cuGreenCtxDestroy) p_cuGreenCtxDestroy;
extern decltype(&cuCtxFromGreenCtx) p_cuCtxFromGreenCtx;
extern decltype(&cuDevResourceGenerateDesc) p_cuDevResourceGenerateDesc;

extern decltype(&cuGreenCtxStreamCreate) p_cuGreenCtxStreamCreate;

extern decltype(&cuStreamCreateWithPriority) p_cuStreamCreateWithPriority;
extern decltype(&cuStreamDestroy) p_cuStreamDestroy;
extern decltype(&cuStreamGetCtx) p_cuStreamGetCtx;

extern decltype(&cuEventCreate) p_cuEventCreate;
extern decltype(&cuEventDestroy) p_cuEventDestroy;
extern decltype(&cuIpcOpenEventHandle) p_cuIpcOpenEventHandle;

extern decltype(&cuDeviceGetCount) p_cuDeviceGetCount;

extern decltype(&cuMemPoolSetAccess) p_cuMemPoolSetAccess;
extern decltype(&cuMemPoolDestroy) p_cuMemPoolDestroy;
extern decltype(&cuMemPoolCreate) p_cuMemPoolCreate;
extern decltype(&cuDeviceGetMemPool) p_cuDeviceGetMemPool;
extern decltype(&cuMemPoolImportFromShareableHandle) p_cuMemPoolImportFromShareableHandle;

extern decltype(&cuMemAllocFromPoolAsync) p_cuMemAllocFromPoolAsync;
extern decltype(&cuMemAllocAsync) p_cuMemAllocAsync;
extern decltype(&cuMemAlloc) p_cuMemAlloc;
extern decltype(&cuMemAllocHost) p_cuMemAllocHost;

extern decltype(&cuMemFreeAsync) p_cuMemFreeAsync;
extern decltype(&cuMemFree) p_cuMemFree;
extern decltype(&cuMemFreeHost) p_cuMemFreeHost;

extern decltype(&cuMemPoolImportPointer) p_cuMemPoolImportPointer;

// Library
extern decltype(&cuLibraryLoadFromFile) p_cuLibraryLoadFromFile;
extern decltype(&cuLibraryLoadData) p_cuLibraryLoadData;
extern decltype(&cuLibraryUnload) p_cuLibraryUnload;
extern decltype(&cuLibraryGetKernel) p_cuLibraryGetKernel;

// Graph
extern decltype(&cuGraphDestroy) p_cuGraphDestroy;
extern decltype(&cuGraphInstantiateWithParams) p_cuGraphInstantiateWithParams;
extern decltype(&cuGraphExecUpdate) p_cuGraphExecUpdate;
extern decltype(&cuGraphExecDestroy) p_cuGraphExecDestroy;
extern decltype(&cuUserObjectCreate) p_cuUserObjectCreate;
extern decltype(&cuUserObjectRelease) p_cuUserObjectRelease;
extern decltype(&cuGraphRetainUserObject) p_cuGraphRetainUserObject;
extern decltype(&cuGraphReleaseUserObject) p_cuGraphReleaseUserObject;
extern decltype(&cuGraphNodeFindInClone) p_cuGraphNodeFindInClone;
extern decltype(&cuGraphChildGraphNodeGetGraph) p_cuGraphChildGraphNodeGetGraph;

// Linker
extern decltype(&cuLinkDestroy) p_cuLinkDestroy;

// Graphics interop
extern decltype(&cuGraphicsUnmapResources) p_cuGraphicsUnmapResources;
extern decltype(&cuGraphicsUnregisterResource) p_cuGraphicsUnregisterResource;

// Texture / surface / array (PR #467)
extern decltype(&cuArray3DCreate) p_cuArray3DCreate;
extern decltype(&cuArrayDestroy) p_cuArrayDestroy;
extern decltype(&cuMipmappedArrayCreate) p_cuMipmappedArrayCreate;
extern decltype(&cuMipmappedArrayDestroy) p_cuMipmappedArrayDestroy;
extern decltype(&cuMipmappedArrayGetLevel) p_cuMipmappedArrayGetLevel;
extern decltype(&cuTexObjectCreate) p_cuTexObjectCreate;
extern decltype(&cuTexObjectDestroy) p_cuTexObjectDestroy;
extern decltype(&cuSurfObjectCreate) p_cuSurfObjectCreate;
extern decltype(&cuSurfObjectDestroy) p_cuSurfObjectDestroy;

// ============================================================================
// NVRTC function pointers
//
// These are populated by _rt.pyx at module import time using
// function pointers extracted from cuda.bindings.cynvrtc.__pyx_capi__.
// ============================================================================

extern decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram;

// ============================================================================
// NVVM function pointers
//
// These are populated by _rt.pyx at module import time using
// function pointers extracted from cuda.bindings.cynvvm.__pyx_capi__.
// Note: May be null if NVVM is not available at runtime.
// ============================================================================

// Function pointer type for nvvmDestroyProgram (avoids nvvm.h dependency)
// Signature: nvvmResult nvvmDestroyProgram(nvvmProgram *prog)
using NvvmDestroyProgramFn = int (*)(nvvmProgram*);
extern NvvmDestroyProgramFn p_nvvmDestroyProgram;

// ============================================================================
// nvJitLink function pointers
//
// These are populated by _rt.pyx at module import time using
// function pointers extracted from cuda.bindings.cynvjitlink.__pyx_capi__.
// Note: May be null if nvJitLink is not available at runtime.
// ============================================================================

// Function pointer type for nvJitLinkDestroy (avoids nvJitLink.h dependency)
// Signature: nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle)
using NvJitLinkDestroyFn = int (*)(nvJitLink_t*);
extern NvJitLinkDestroyFn p_nvJitLinkDestroy;

}  // namespace cuda_core::rt
