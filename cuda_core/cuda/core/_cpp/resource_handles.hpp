// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cstdint>
#include <memory>

// Forward declaration for NVVM - avoids nvvm.h dependency
// Use void* to match cuda.bindings.cynvvm's typedef
using nvvmProgram = void*;

namespace cuda_core {

// ============================================================================
// Thread-local error handling
// ============================================================================

// Get and clear the last CUDA error (like cudaGetLastError)
CUresult get_last_error() noexcept;

// Get the last CUDA error without clearing it (like cudaPeekAtLastError)
CUresult peek_last_error() noexcept;

// Explicitly clear the last error
void clear_last_error() noexcept;

// ============================================================================
// CUDA driver function pointers
//
// These are populated by _resource_handles.pyx at module import time using
// function pointers extracted from cuda.bindings.cydriver.__pyx_capi__.
// ============================================================================

extern decltype(&cuDevicePrimaryCtxRetain) p_cuDevicePrimaryCtxRetain;
extern decltype(&cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease;
extern decltype(&cuCtxGetCurrent) p_cuCtxGetCurrent;

extern decltype(&cuStreamCreateWithPriority) p_cuStreamCreateWithPriority;
extern decltype(&cuStreamDestroy) p_cuStreamDestroy;

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

// Graphics interop
extern decltype(&cuGraphicsUnregisterResource) p_cuGraphicsUnregisterResource;

// ============================================================================
// NVRTC function pointers
//
// These are populated by _resource_handles.pyx at module import time using
// function pointers extracted from cuda.bindings.cynvrtc.__pyx_capi__.
// ============================================================================

extern decltype(&nvrtcDestroyProgram) p_nvrtcDestroyProgram;

// ============================================================================
// NVVM function pointers
//
// These are populated by _resource_handles.pyx at module import time using
// function pointers extracted from cuda.bindings.cynvvm.__pyx_capi__.
// Note: May be null if NVVM is not available at runtime.
// ============================================================================

// Function pointer type for nvvmDestroyProgram (avoids nvvm.h dependency)
// Signature: nvvmResult nvvmDestroyProgram(nvvmProgram *prog)
using NvvmDestroyProgramFn = int (*)(nvvmProgram*);
extern NvvmDestroyProgramFn p_nvvmDestroyProgram;

// ============================================================================
// Handle type aliases - expose only the raw CUDA resource
// ============================================================================

using ContextHandle = std::shared_ptr<const CUcontext>;
using StreamHandle = std::shared_ptr<const CUstream>;
using EventHandle = std::shared_ptr<const CUevent>;
using MemoryPoolHandle = std::shared_ptr<const CUmemoryPool>;
using LibraryHandle = std::shared_ptr<const CUlibrary>;
using KernelHandle = std::shared_ptr<const CUkernel>;
using GraphicsResourceHandle = std::shared_ptr<const CUgraphicsResource>;
using NvrtcProgramHandle = std::shared_ptr<const nvrtcProgram>;
using NvvmProgramHandle = std::shared_ptr<const nvvmProgram>;


// ============================================================================
// Context handle functions
// ============================================================================

// Function to create a non-owning context handle (references existing context).
ContextHandle create_context_handle_ref(CUcontext ctx);

// Get handle to the primary context for a device (with thread-local caching)
// Returns empty handle on error (caller must check)
ContextHandle get_primary_context(int device_id);

// Get handle to the current CUDA context
// Returns empty handle if no context is current (caller must check)
ContextHandle get_current_context();

// ============================================================================
// Stream handle functions
// ============================================================================

// Create an owning stream handle by calling cuStreamCreateWithPriority.
// The stream structurally depends on the provided context handle.
// When the last reference is released, cuStreamDestroy is called automatically.
// Returns empty handle on error (caller must check).
StreamHandle create_stream_handle(const ContextHandle& h_ctx, unsigned int flags, int priority);

// Create a non-owning stream handle (references existing stream).
// Use for borrowed streams (from foreign code) or built-in streams.
// The stream will NOT be destroyed when the handle is released.
// Caller is responsible for keeping the stream's context alive.
StreamHandle create_stream_handle_ref(CUstream stream);

// Create a non-owning stream handle that prevents a Python owner from being GC'd.
// The owner's refcount is incremented; decremented when handle is released.
// The owner is responsible for keeping the stream's context alive.
StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner);

// Get non-owning handle to the legacy default stream (CU_STREAM_LEGACY)
// Note: Legacy stream has no specific context dependency.
StreamHandle get_legacy_stream();

// Get non-owning handle to the per-thread default stream (CU_STREAM_PER_THREAD)
// Note: Per-thread stream has no specific context dependency.
StreamHandle get_per_thread_stream();

// ============================================================================
// Event handle functions
// ============================================================================

// Create an owning event handle by calling cuEventCreate.
// The event structurally depends on the provided context handle.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle(const ContextHandle& h_ctx, unsigned int flags);

// Create an owning event handle without context dependency.
// Use for temporary events that are created and destroyed in the same scope.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle_noctx(unsigned int flags);

// Create an owning event handle from an IPC handle.
// The originating process owns the event and its context.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle);

// ============================================================================
// Memory pool handle functions
// ============================================================================

// Create an owning memory pool handle by calling cuMemPoolCreate.
// Memory pools are device-scoped (not context-scoped).
// When the last reference is released, cuMemPoolDestroy is called automatically.
// Returns empty handle on error (caller must check).
MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props);

// Create a non-owning memory pool handle (references existing pool).
// Use for device default/current pools that are managed by the driver.
// The pool will NOT be destroyed when the handle is released.
MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool);

// Get non-owning handle to the current memory pool for a device.
// Returns empty handle on error (caller must check).
MemoryPoolHandle get_device_mempool(int device_id);

// Create an owning memory pool handle from an IPC import.
// The file descriptor is NOT owned by this handle (caller manages FD separately).
// When the last reference is released, cuMemPoolDestroy is called automatically.
// Returns empty handle on error (caller must check).
MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type);

// ============================================================================
// Device pointer handle functions
// ============================================================================

using DevicePtrHandle = std::shared_ptr<const CUdeviceptr>;

// Allocate device memory from a pool asynchronously via cuMemAllocFromPoolAsync.
// The pointer structurally depends on the provided pool handle (captured in deleter).
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_from_pool(
    size_t size,
    const MemoryPoolHandle& h_pool,
    const StreamHandle& h_stream);

// Allocate device memory asynchronously via cuMemAllocAsync.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_async(size_t size, const StreamHandle& h_stream);

// Allocate device memory synchronously via cuMemAlloc.
// When the last reference is released, cuMemFree is called.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc(size_t size);

// Allocate pinned host memory via cuMemAllocHost.
// When the last reference is released, cuMemFreeHost is called.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_host(size_t size);

// Create a non-owning device pointer handle (references existing pointer).
// Use for foreign pointers (e.g., from external libraries).
// The pointer will NOT be freed when the handle is released.
DevicePtrHandle deviceptr_create_ref(CUdeviceptr ptr);

// Create a non-owning device pointer handle that prevents a Python owner from being GC'd.
// The owner's refcount is incremented; decremented when handle is released.
// The pointer will NOT be freed when the handle is released.
// If owner is nullptr, equivalent to deviceptr_create_ref.
DevicePtrHandle deviceptr_create_with_owner(CUdeviceptr ptr, PyObject* owner);

// Callback type for MemoryResource deallocation.
// Called from the shared_ptr deleter when a handle created via
// deviceptr_create_with_mr is destroyed.  The implementation is responsible
// for converting raw C types to Python objects and calling
// mr.deallocate(ptr, size, stream).
using MRDeallocCallback = void (*)(PyObject* mr, CUdeviceptr ptr,
                                   size_t size, const StreamHandle& stream);

// Register the MR deallocation callback.
void register_mr_dealloc_callback(MRDeallocCallback cb);

// Create a device pointer handle whose destructor calls mr.deallocate()
// via the registered callback.  The mr's refcount is incremented and
// decremented when the handle is released.
// If mr is nullptr, equivalent to deviceptr_create_ref.
DevicePtrHandle deviceptr_create_with_mr(CUdeviceptr ptr, size_t size, PyObject* mr);

// Import a device pointer from IPC via cuMemPoolImportPointer.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Note: Does not yet implement reference counting for nvbug 5570902.
// On error, returns empty handle and sets thread-local error (use get_last_error()).
DevicePtrHandle deviceptr_import_ipc(
    const MemoryPoolHandle& h_pool,
    const void* export_data,
    const StreamHandle& h_stream);

// Access the deallocation stream for a device pointer handle (read-only).
// For non-owning handles, the stream is not used but can still be accessed.
StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept;

// Set the deallocation stream for a device pointer handle.
void set_deallocation_stream(const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept;

// ============================================================================
// Library handle functions
// ============================================================================

// Create an owning library handle by loading from a file path.
// When the last reference is released, cuLibraryUnload is called automatically.
// Returns empty handle on error (caller must check).
LibraryHandle create_library_handle_from_file(const char* path);

// Create an owning library handle by loading from memory data.
// The driver makes an internal copy of the data; caller can free it after return.
// When the last reference is released, cuLibraryUnload is called automatically.
// Returns empty handle on error (caller must check).
LibraryHandle create_library_handle_from_data(const void* data);

// Create a non-owning library handle (references existing library).
// Use for borrowed libraries (e.g., from foreign code).
// The library will NOT be unloaded when the handle is released.
LibraryHandle create_library_handle_ref(CUlibrary library);

// ============================================================================
// Kernel handle functions
// ============================================================================

// Get a kernel from a library by name.
// The kernel structurally depends on the provided library handle.
// Kernels have no explicit destroy - their lifetime is tied to the library.
// Returns empty handle on error (caller must check).
KernelHandle create_kernel_handle(const LibraryHandle& h_library, const char* name);

// Create a non-owning kernel handle with library dependency.
// Use for borrowed kernels. The library handle keeps the library alive.
KernelHandle create_kernel_handle_ref(CUkernel kernel, const LibraryHandle& h_library);

// ============================================================================
// Graphics resource handle functions
// ============================================================================

// Create an owning graphics resource handle.
// When the last reference is released, cuGraphicsUnregisterResource is called automatically.
// Use for CUgraphicsResource handles obtained from cuGraphicsGLRegisterBuffer etc.
GraphicsResourceHandle create_graphics_resource_handle(CUgraphicsResource resource);

// ============================================================================
// NVRTC Program handle functions
// ============================================================================

// Create an owning NVRTC program handle.
// When the last reference is released, nvrtcDestroyProgram is called.
// Use this to wrap a program created via nvrtcCreateProgram.
NvrtcProgramHandle create_nvrtc_program_handle(nvrtcProgram prog);

// Create a non-owning NVRTC program handle (references existing program).
// The program will NOT be destroyed when the handle is released.
NvrtcProgramHandle create_nvrtc_program_handle_ref(nvrtcProgram prog);

// ============================================================================
// NVVM Program handle functions
// ============================================================================

// Create an owning NVVM program handle.
// When the last reference is released, nvvmDestroyProgram is called.
// Use this to wrap a program created via nvvmCreateProgram.
// Note: If NVVM is not available (p_nvvmDestroyProgram is null), the deleter is a no-op.
NvvmProgramHandle create_nvvm_program_handle(nvvmProgram prog);

// Create a non-owning NVVM program handle (references existing program).
// The program will NOT be destroyed when the handle is released.
NvvmProgramHandle create_nvvm_program_handle_ref(nvvmProgram prog);

// ============================================================================
// Overloaded helper functions to extract raw resources from handles
// ============================================================================

// as_cu() - extract the raw CUDA handle
inline CUcontext as_cu(const ContextHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUstream as_cu(const StreamHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUevent as_cu(const EventHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUmemoryPool as_cu(const MemoryPoolHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUdeviceptr as_cu(const DevicePtrHandle& h) noexcept {
    return h ? *h : 0;
}

inline CUlibrary as_cu(const LibraryHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUkernel as_cu(const KernelHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUgraphicsResource as_cu(const GraphicsResourceHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline nvrtcProgram as_cu(const NvrtcProgramHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline nvvmProgram as_cu(const NvvmProgramHandle& h) noexcept {
    return h ? *h : nullptr;
}

// as_intptr() - extract handle as intptr_t for Python interop
// Using signed intptr_t per C standard convention and issue #1342
inline std::intptr_t as_intptr(const ContextHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const StreamHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const EventHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const MemoryPoolHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const DevicePtrHandle& h) noexcept {
    return static_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const LibraryHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const KernelHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const GraphicsResourceHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const NvrtcProgramHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

inline std::intptr_t as_intptr(const NvvmProgramHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(as_cu(h));
}

// as_py() - convert handle to Python wrapper object (returns new reference)
namespace detail {
// n.b. class lookup is not cached to avoid deadlock hazard, see DESIGN.md
inline PyObject* make_py(const char* module_name, const char* class_name, std::intptr_t value) noexcept {
    PyObject* mod = PyImport_ImportModule(module_name);
    if (!mod) return nullptr;
    PyObject* cls = PyObject_GetAttrString(mod, class_name);
    Py_DECREF(mod);
    if (!cls) return nullptr;
    PyObject* result = PyObject_CallFunction(cls, "L", value);
    Py_DECREF(cls);
    return result;
}
}  // namespace detail

inline PyObject* as_py(const ContextHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUcontext", as_intptr(h));
}

inline PyObject* as_py(const StreamHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUstream", as_intptr(h));
}

inline PyObject* as_py(const EventHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUevent", as_intptr(h));
}

inline PyObject* as_py(const MemoryPoolHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUmemoryPool", as_intptr(h));
}

inline PyObject* as_py(const DevicePtrHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUdeviceptr", as_intptr(h));
}

inline PyObject* as_py(const LibraryHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUlibrary", as_intptr(h));
}

inline PyObject* as_py(const KernelHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUkernel", as_intptr(h));
}

inline PyObject* as_py(const NvrtcProgramHandle& h) noexcept {
    return detail::make_py("cuda.bindings.nvrtc", "nvrtcProgram", as_intptr(h));
}

inline PyObject* as_py(const NvvmProgramHandle& h) noexcept {
    // NVVM bindings use raw integers, not wrapper classes
    return PyLong_FromSsize_t(as_intptr(h));
}

inline PyObject* as_py(const GraphicsResourceHandle& h) noexcept {
    return detail::make_py("cuda.bindings.driver", "CUgraphicsResource", as_intptr(h));
}

}  // namespace cuda_core
