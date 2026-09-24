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
// Driver and compiler-library function pointers
//
// The C++ under _cpp/rt/ calls the CUDA driver through the p_cuXxx pointers
// below. They hold the driver's own entry points, taken from the table that
// cuda-bindings builds when it loads the driver. cuda-bindings calls
// cuGetProcAddress for each symbol and chooses the ABI variant and the
// per-thread-default-stream variant. It exposes the table as
// cuda.bindings._internal.driver._inspect_function_pointers().
// cuda.core never loads the driver or resolves a symbol itself. It never calls
// cuda-bindings' Cython wrappers from C++: those raise a Python exception when
// the driver lacks a function, and C++ cannot see that exception. See
// https://github.com/NVIDIA/cuda-python/issues/2783.
//
// The first DRIVER_CALL that finds its pointer null fills the table, so that
// `import cuda.core` never touches the driver. The fill acquires the GIL and
// runs Python (see py_driver_fns.cpp), so a DRIVER_CALL must not run while the
// thread holds a C++ lock. Call ensure_fn_table() before you take the lock.
// Inside the lock, use the raw pointer (see deviceptr_import_ipc).
//
// After the fill a pointer is either the driver's entry point or null because
// the installed driver does not provide that function. Functions introduced
// at or before the first release of the build's CUDA major series are present
// in every driver cuda.core supports. The fill checks them and rejects an
// older driver. A newer function can be null. The Cython layer gates its use
// on the driver version (cy_driver_version()), so the C++ never checks a
// pointer for null before a call. A DRIVER_CALL that still finds null after
// the fill is therefore a gate bug or a failed fill. It reports an internal
// error and returns CUDA_ERROR_NOT_INITIALIZED from a trampoline of the right
// signature, and it never dereferences null. It never throws, so it is safe in
// noexcept deleters and cleanup paths.
// ============================================================================

// Each X(name, introduced) names a driver function cuda.core calls and the CUDA
// version cuda-bindings requests it at: the cudaVersion it passes to
// cuGetProcAddress, which is the ABI's introduction. tests/test_rt_layout.py
// checks the list against cuda-bindings' loader. `name` is the public name.
// cuda.h may map it to a versioned symbol, for example
// cuStreamDestroy -> cuStreamDestroy_v2, and the table key follows that
// mapping. The header cuda.core compiles against must therefore be the one
// cuda-bindings was generated from. build_hooks.py enforces this.
#define CUDA_CORE_DRIVER_FUNCTIONS(X)            \
    /* Error formatting */                       \
    X(cuGetErrorName, 6000)                      \
    X(cuGetErrorString, 6000)                    \
    /* Context */                                \
    X(cuDevicePrimaryCtxRetain, 7000)            \
    X(cuDevicePrimaryCtxRelease, 11000)          \
    X(cuCtxGetCurrent, 4000)                     \
    X(cuCtxSetCurrent, 4000)                     \
    X(cuCtxSynchronize, 2000)                    \
    X(cuCtxGetStreamPriorityRange, 5050)         \
    X(cuCtxGetDevice, 2000)                      \
    X(cuGraphNodeSetParams, 12020)               \
    X(cuGreenCtxCreate, 12040)                   \
    X(cuGreenCtxDestroy, 12040)                  \
    X(cuCtxFromGreenCtx, 12040)                  \
    X(cuDevResourceGenerateDesc, 12040)          \
    X(cuGreenCtxStreamCreate, 12050)             \
    /* Stream */                                 \
    X(cuStreamCreateWithPriority, 5050)          \
    X(cuStreamDestroy, 4000)                     \
    X(cuStreamGetCtx, 9020)                      \
    /* Event */                                  \
    X(cuEventCreate, 2000)                       \
    X(cuEventDestroy, 4000)                      \
    X(cuIpcOpenEventHandle, 4010)                \
    /* Device */                                 \
    X(cuDeviceGetCount, 2000)                    \
    /* Memory pool */                            \
    X(cuMemPoolSetAccess, 11020)                 \
    X(cuMemPoolDestroy, 11020)                   \
    X(cuMemPoolCreate, 11020)                    \
    X(cuDeviceGetMemPool, 11020)                 \
    X(cuMemPoolImportFromShareableHandle, 11020) \
    /* Memory allocation */                      \
    X(cuMemAllocFromPoolAsync, 11020)            \
    X(cuMemAllocAsync, 11020)                    \
    X(cuMemAlloc, 3020)                          \
    X(cuMemAllocHost, 3020)                      \
    X(cuMemFreeAsync, 11020)                     \
    X(cuMemFree, 3020)                           \
    X(cuMemFreeHost, 2000)                       \
    /* IPC */                                    \
    X(cuMemPoolImportPointer, 11020)             \
    /* Library */                                \
    X(cuLibraryLoadFromFile, 12000)              \
    X(cuLibraryLoadData, 12000)                  \
    X(cuLibraryUnload, 12000)                    \
    X(cuLibraryGetKernel, 12000)                 \
    /* Graph */                                  \
    X(cuGraphDestroy, 10000)                     \
    X(cuGraphInstantiateWithParams, 12000)       \
    X(cuGraphExecUpdate, 12000)                  \
    X(cuGraphExecDestroy, 10000)                 \
    X(cuUserObjectCreate, 11030)                 \
    X(cuUserObjectRelease, 11030)                \
    X(cuGraphRetainUserObject, 11030)            \
    X(cuGraphReleaseUserObject, 11030)           \
    X(cuGraphNodeFindInClone, 10000)             \
    X(cuGraphChildGraphNodeGetGraph, 10000)      \
    /* Linker */                                 \
    X(cuLinkDestroy, 5050)                       \
    /* Graphics interop. cuda-bindings requests 7000 (PTDS) or 3000 (legacy) */ \
    X(cuGraphicsUnmapResources, 7000)            \
    X(cuGraphicsUnregisterResource, 3000)        \
    /* Texture / surface / array (PR #467) */    \
    X(cuArray3DCreate, 3020)                     \
    X(cuArrayDestroy, 2000)                      \
    X(cuMipmappedArrayCreate, 5000)              \
    X(cuMipmappedArrayDestroy, 5000)             \
    X(cuMipmappedArrayGetLevel, 5000)            \
    X(cuTexObjectCreate, 5000)                   \
    X(cuTexObjectDestroy, 5000)                  \
    X(cuSurfObjectCreate, 5000)                  \
    X(cuSurfObjectDestroy, 5000)

#define CUDA_CORE_DECLARE_DRIVER_FN(name, introduced) extern decltype(&name) p_##name;
CUDA_CORE_DRIVER_FUNCTIONS(CUDA_CORE_DECLARE_DRIVER_FN)
#undef CUDA_CORE_DECLARE_DRIVER_FN

// Compiler-library entry points, one table per library, so that a fill of one
// table does not load the other libraries. NVVM and nvJitLink are optional at
// run time. Where this file does not include the library header, the
// declaration spells out the type. The types match
// `nvvmResult nvvmDestroyProgram(nvvmProgram*)` and
// `nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle*)` as int-sized enums.
extern decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram;
using NvvmDestroyProgramFn = int (*)(nvvmProgram*);
extern NvvmDestroyProgramFn p_nvvmDestroyProgram;
using NvJitLinkDestroyFn = int (*)(nvJitLink_t*);
extern NvJitLinkDestroyFn p_nvJitLinkDestroy;

// ============================================================================
// Function tables
// ============================================================================

enum class FnTable { driver, nvrtc, nvvm, nvjitlink };

struct FnEntry {
    const char* key;   // cuda-bindings' name for the slot, e.g. "__cuStreamDestroy_v2"
    const char* name;  // public name, for messages, e.g. "cuStreamDestroy"
    void** slot;       // the p_ pointer, as storage
    int introduced;    // CUDA version cuda-bindings requests the symbol at (0 = n/a)
};

// The entries of a table. Implemented in driver_api.cpp.
const FnEntry* fn_table_entries(FnTable table, std::size_t* count) noexcept;

// Whether a table is filled. An acquire load: a true result orders the slots.
bool fn_table_ready(FnTable table) noexcept;

// Fill a table from cuda-bindings if it is not ready. The fill acquires the
// GIL, imports cuda.bindings._internal.<lib>, calls
// _inspect_function_pointers(), and stores every entry's pointer. Never call
// it with a C++ lock held. It returns false, records the reason for
// fn_table_error(), and reports the reason through report_message() when:
//   - cuda-bindings cannot load the library
//   - a key is missing, because the installed cuda-bindings does not match
//     the header this build compiled against
//   - a baseline driver function is null, because the driver is older than
//     the CUDA major series supports
// A failed fill is latched: later calls return false at once and do not
// retry. The fill preserves a pending Python exception and never leaves one
// set. Implemented in py_driver_fns.cpp.
bool ensure_fn_table(FnTable table) noexcept;

// Copy the reason the fill of `table` failed into `buffer`. Return false if
// it did not fail. Implemented in py_driver_fns.cpp.
bool fn_table_error(FnTable table, char* buffer, std::size_t size) noexcept;

// Record a call to `name` while it is unavailable. After a failed fill, this
// function attaches the fill's reason to the error the caller raises. The fill
// reported that reason when it failed. A null entry in a filled table is a gate
// bug, which this function reports once per table. Implemented in
// py_driver_fns.cpp.
void report_unavailable_fn(FnTable table, const char* name) noexcept;

namespace detail {

// The status a trampoline returns in place of an unavailable function.
template <class R>
struct UnavailableStatus;
template <>
struct UnavailableStatus<CUresult> {
    static constexpr CUresult value = CUDA_ERROR_NOT_INITIALIZED;
};
template <>
struct UnavailableStatus<nvrtcResult> {
    static constexpr nvrtcResult value = NVRTC_ERROR_INTERNAL_ERROR;
};
template <>
struct UnavailableStatus<int> {
    static constexpr int value = -1;
};

// A function of the same signature as an unavailable entry point, so that a
// call site never dereferences null. Its status flows to the caller's normal
// error handling. The fill or report_unavailable_fn() already reported the
// cause.
template <class F>
struct Unavailable;
template <class R, class... A>
struct Unavailable<R (*)(A...)> {
    static R call(A...) noexcept { return UnavailableStatus<R>::value; }
};

// Fills the table on first use and returns the resolved pointer, or the
// trampoline when the function is unavailable after the fill. Never throws.
template <class F>
inline F fn_or_unavailable(F& slot, FnTable table, const char* name) noexcept {
    if (!fn_table_ready(table)) {
        ensure_fn_table(table);
    }
    if (F fn = slot) {
        return fn;
    }
    report_unavailable_fn(table, name);
    return &Unavailable<F>::call;
}

}  // namespace detail
}  // namespace cuda_core::rt

// A driver function's table entry, resolved on first use. DRIVER_CALL(name, args...)
// calls it. Use these macros for every driver call in the C++ layer, except under
// a C++ lock (see the header comment). Such a call carries a `// raw:` marker.
// Each macro pastes its own parameter: if it passed `name` through another
// macro, cuda.h's versioning macros would rewrite it first, for example
// cuMemFree -> cuMemFree_v2.
#define DRIVER_FN(name) \
    (::cuda_core::rt::detail::fn_or_unavailable(::cuda_core::rt::p_##name, ::cuda_core::rt::FnTable::driver, #name))
#define DRIVER_CALL(name, ...) \
    (::cuda_core::rt::detail::fn_or_unavailable(::cuda_core::rt::p_##name, ::cuda_core::rt::FnTable::driver, #name)(__VA_ARGS__))
#define NVRTC_CALL(name, ...) \
    (::cuda_core::rt::detail::fn_or_unavailable(::cuda_core::rt::p_##name, ::cuda_core::rt::FnTable::nvrtc, #name)(__VA_ARGS__))
#define NVVM_CALL(name, ...) \
    (::cuda_core::rt::detail::fn_or_unavailable(::cuda_core::rt::p_##name, ::cuda_core::rt::FnTable::nvvm, #name)(__VA_ARGS__))
#define NVJITLINK_CALL(name, ...) \
    (::cuda_core::rt::detail::fn_or_unavailable(::cuda_core::rt::p_##name, ::cuda_core::rt::FnTable::nvjitlink, #name)(__VA_ARGS__))
