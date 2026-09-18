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
// cuda-bindings builds when it loads the driver (cuGetProcAddress for each
// symbol, choosing the ABI variant and the per-thread-default-stream variant)
// and exposes as cuda.bindings._internal.driver._inspect_function_pointers().
// cuda.core never loads the driver or resolves a symbol itself, and it never
// calls cuda-bindings' Cython wrappers from C++: those raise a Python
// exception when the driver lacks a function, which C++ cannot see
// (https://github.com/NVIDIA/cuda-python/issues/2783).
//
// The table is filled lazily, the first time a DRIVER_CALL finds its pointer
// null, so that `import cuda.core` never touches the driver. The fill acquires
// the GIL and runs Python (see py_driver_fns.cpp), so a DRIVER_CALL must not
// be made while a C++ lock is held; call ensure_fn_table() before taking the
// lock and use the raw pointer inside it (see deviceptr_import_ipc).
//
// After the fill a pointer is either the driver's entry point or null because
// the installed driver does not provide that function. Functions introduced
// at or before the first release of the CUDA major series being built are
// present in every driver cuda.core supports; the fill checks them and
// rejects an older driver. A newer function can legitimately be null, and the
// Cython layer gates its use on the driver version (cy_driver_version()), so
// the C++ never checks a pointer for null before a call. A DRIVER_CALL that
// still finds null after the fill is therefore a gate bug (or a failed fill);
// it reports an internal error and returns CUDA_ERROR_NOT_INITIALIZED from a
// trampoline of the right signature instead of dereferencing null. It never
// throws, so it is safe in noexcept deleters and cleanup paths.
// ============================================================================

// Each X(name, introduced) names a driver function cuda.core calls and the CUDA
// version cuda-bindings requests it at (cuGetProcAddress's cudaVersion; the
// ABI's introduction). tests/test_rt_layout.py checks the list against
// cuda-bindings' loader. `name` is the public name; cuda.h may map it to a
// versioned symbol (cuStreamDestroy -> cuStreamDestroy_v2), and the table key
// follows that mapping, so the header cuda.core compiles against must be the one
// cuda-bindings was generated from (enforced by build_hooks.py).
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
    /* Graphics interop */                       \
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

// Compiler-library entry points, one table per library so that filling one
// does not load the others (NVVM and nvJitLink are optional at run time).
// Types are spelled out where the library header is not included; they match
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

// Whether a table has been filled (acquire: a true result orders the slots).
bool fn_table_ready(FnTable table) noexcept;

// Fill a table from cuda-bindings if it is not ready. Acquires the GIL (never
// call with a C++ lock held), imports cuda.bindings._internal.<lib>, calls
// _inspect_function_pointers(), and stores every entry's pointer. Returns
// false, records the reason (fn_table_error) and reports it through
// report_message() when cuda-bindings cannot load the library, a key is
// missing (the installed cuda-bindings does not match the header this build
// compiled against), or a baseline driver function is null (the driver is
// older than the CUDA major series supports). Never leaves a Python error set.
// Implemented in py_driver_fns.cpp.
bool ensure_fn_table(FnTable table) noexcept;

// The reason the last fill of `table` failed, or nullptr. Implemented in py_driver_fns.cpp.
const char* fn_table_error(FnTable table) noexcept;

// Report, once per table, that `name` was called while unavailable: a gate
// bug, or a failed fill (whose reason is included). Implemented in py_driver_fns.cpp.
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
// error handling; the cause has already been reported.
template <class F>
struct Unavailable;
template <class R, class... A>
struct Unavailable<R (*)(A...)> {
    static R call(A...) noexcept { return UnavailableStatus<R>::value; }
};

// The resolved pointer, filling the table on first use; the trampoline when
// the function is unavailable after the fill. Never throws.
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
// calls it; use these for every driver call in the C++ layer except under a C++
// lock (see the header comment; such a call is marked `// raw:`). Each macro
// pastes its own parameter: passing `name` through another macro would let
// cuda.h's versioning macros rewrite it (cuMemFree -> cuMemFree_v2) first.
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
