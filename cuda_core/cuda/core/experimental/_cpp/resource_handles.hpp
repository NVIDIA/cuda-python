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
// Handle type aliases - expose only the raw CUDA resource
// ============================================================================

using ContextHandle = std::shared_ptr<const CUcontext>;
using StreamHandle = std::shared_ptr<const CUstream>;
using EventHandle = std::shared_ptr<const CUevent>;

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

// Create an owning event handle from an IPC handle.
// The originating process owns the event and its context.
// When the last reference is released, cuEventDestroy is called automatically.
// Returns empty handle on error (caller must check).
EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle);

// ============================================================================
// Overloaded helper functions to extract raw resources from handles
// ============================================================================

// native() - extract the raw CUDA handle
inline CUcontext native(const ContextHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUstream native(const StreamHandle& h) noexcept {
    return h ? *h : nullptr;
}

inline CUevent native(const EventHandle& h) noexcept {
    return h ? *h : nullptr;
}

// intptr() - extract handle as uintptr_t for Python interop
inline std::uintptr_t intptr(const ContextHandle& h) noexcept {
    return reinterpret_cast<std::uintptr_t>(h ? *h : nullptr);
}

inline std::uintptr_t intptr(const StreamHandle& h) noexcept {
    return reinterpret_cast<std::uintptr_t>(h ? *h : nullptr);
}

inline std::uintptr_t intptr(const EventHandle& h) noexcept {
    return reinterpret_cast<std::uintptr_t>(h ? *h : nullptr);
}

// py() - convert handle to Python driver wrapper object
// Returns new reference. Caller must hold GIL.
inline PyObject* py(const ContextHandle& h) {
    static PyObject* cls = nullptr;
    if (!cls) {
        PyObject* mod = PyImport_ImportModule("cuda.bindings.driver");
        if (!mod) return nullptr;
        cls = PyObject_GetAttrString(mod, "CUcontext");
        Py_DECREF(mod);
        if (!cls) return nullptr;
    }
    std::uintptr_t val = h ? reinterpret_cast<std::uintptr_t>(*h) : 0;
    return PyObject_CallFunction(cls, "K", val);
}

inline PyObject* py(const StreamHandle& h) {
    static PyObject* cls = nullptr;
    if (!cls) {
        PyObject* mod = PyImport_ImportModule("cuda.bindings.driver");
        if (!mod) return nullptr;
        cls = PyObject_GetAttrString(mod, "CUstream");
        Py_DECREF(mod);
        if (!cls) return nullptr;
    }
    std::uintptr_t val = h ? reinterpret_cast<std::uintptr_t>(*h) : 0;
    return PyObject_CallFunction(cls, "K", val);
}

inline PyObject* py(const EventHandle& h) {
    static PyObject* cls = nullptr;
    if (!cls) {
        PyObject* mod = PyImport_ImportModule("cuda.bindings.driver");
        if (!mod) return nullptr;
        cls = PyObject_GetAttrString(mod, "CUevent");
        Py_DECREF(mod);
        if (!cls) return nullptr;
    }
    std::uintptr_t val = h ? reinterpret_cast<std::uintptr_t>(*h) : 0;
    return PyObject_CallFunction(cls, "K", val);
}

}  // namespace cuda_core
