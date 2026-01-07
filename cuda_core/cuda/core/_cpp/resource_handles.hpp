// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>
#include <cuda.h>
#include <cstdint>
#include <memory>

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
// Handle type aliases - expose only the raw CUDA resource
// ============================================================================

using ContextHandle = std::shared_ptr<const CUcontext>;
using StreamHandle = std::shared_ptr<const CUstream>;
using EventHandle = std::shared_ptr<const CUevent>;
using MemoryPoolHandle = std::shared_ptr<const CUmemoryPool>;

// ============================================================================
// Context handle functions
// ============================================================================

// Function to create a non-owning context handle (references existing context).
ContextHandle create_context_handle_ref(CUcontext ctx);

// Get handle to the primary context for a device (with thread-local caching)
// Returns empty handle on error (caller must check)
ContextHandle get_primary_context(int device_id) noexcept;

// Get handle to the current CUDA context
// Returns empty handle if no context is current (caller must check)
ContextHandle get_current_context() noexcept;

// ============================================================================
// Stream handle functions
// ============================================================================

// Create an owning stream handle by calling cuStreamCreateWithPriority.
// The stream structurally depends on the provided context handle.
// When the last reference is released, cuStreamDestroy is called automatically.
// Returns empty handle on error (caller must check).
StreamHandle create_stream_handle(ContextHandle h_ctx, unsigned int flags, int priority);

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
StreamHandle get_legacy_stream() noexcept;

// Get non-owning handle to the per-thread default stream (CU_STREAM_PER_THREAD)
// Note: Per-thread stream has no specific context dependency.
StreamHandle get_per_thread_stream() noexcept;

// ============================================================================
// Event handle functions
// ============================================================================

// Create an owning event handle by calling cuEventCreate.
// The event structurally depends on the provided context handle.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags);

// Create an owning event handle without context dependency.
// Use for temporary events that are created and destroyed in the same scope.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle(unsigned int flags);

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
MemoryPoolHandle get_device_mempool(int device_id) noexcept;

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
    MemoryPoolHandle h_pool,
    StreamHandle h_stream);

// Allocate device memory asynchronously via cuMemAllocAsync.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Returns empty handle on error (caller must check).
DevicePtrHandle deviceptr_alloc_async(size_t size, StreamHandle h_stream);

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

// Import a device pointer from IPC via cuMemPoolImportPointer.
// When the last reference is released, cuMemFreeAsync is called on the stored stream.
// Note: Does not yet implement reference counting for nvbug 5570902.
// On error, returns empty handle and sets thread-local error (use get_last_error()).
DevicePtrHandle deviceptr_import_ipc(
    MemoryPoolHandle h_pool,
    const void* export_data,
    StreamHandle h_stream);

// Access the deallocation stream for a device pointer handle (read-only).
// For non-owning handles, the stream is not used but can still be accessed.
StreamHandle deallocation_stream(const DevicePtrHandle& h);

// Set the deallocation stream for a device pointer handle.
void set_deallocation_stream(const DevicePtrHandle& h, StreamHandle h_stream);

// ============================================================================
// Overloaded helper functions to extract raw resources from handles
// ============================================================================

// cu() - extract the raw CUDA handle
inline CUcontext cu(const ContextHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUstream cu(const StreamHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUevent cu(const EventHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUmemoryPool cu(const MemoryPoolHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUdeviceptr cu(const DevicePtrHandle& h) noexcept {
    return h ? *h : 0;
}

// intptr() - extract handle as intptr_t for Python interop
// Using signed intptr_t per C standard convention and issue #1342
inline std::intptr_t intptr(const ContextHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(cu(h));
}

inline std::intptr_t intptr(const StreamHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(cu(h));
}

inline std::intptr_t intptr(const EventHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(cu(h));
}

inline std::intptr_t intptr(const MemoryPoolHandle& h) noexcept {
    return reinterpret_cast<std::intptr_t>(cu(h));
}

inline std::intptr_t intptr(const DevicePtrHandle& h) noexcept {
    return static_cast<std::intptr_t>(cu(h));
}

// py() - convert handle to Python driver wrapper object (returns new reference)
namespace detail {
// n.b. class lookup is not cached to avoid deadlock hazard, see DESIGN.md
inline PyObject* make_py(const char* class_name, std::intptr_t value) {
    PyObject* mod = PyImport_ImportModule("cuda.bindings.driver");
    if (!mod) return nullptr;
    PyObject* cls = PyObject_GetAttrString(mod, class_name);
    Py_DECREF(mod);
    if (!cls) return nullptr;
    PyObject* result = PyObject_CallFunction(cls, "L", value);
    Py_DECREF(cls);
    return result;
}
}  // namespace detail

inline PyObject* py(const ContextHandle& h) {
    return detail::make_py("CUcontext", intptr(h));
}

inline PyObject* py(const StreamHandle& h) {
    return detail::make_py("CUstream", intptr(h));
}

inline PyObject* py(const EventHandle& h) {
    return detail::make_py("CUevent", intptr(h));
}

inline PyObject* py(const MemoryPoolHandle& h) {
    return detail::make_py("CUmemoryPool", intptr(h));
}

inline PyObject* py(const DevicePtrHandle& h) {
    return detail::make_py("CUdeviceptr", intptr(h));
}

}  // namespace cuda_core
