// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>
#include <cuda.h>
#include <cstdint>
#include <memory>

namespace cuda_core {

// Handle type aliases - expose only the raw CUDA resource
using ContextHandle = std::shared_ptr<const CUcontext>;

// Function to create a non-owning context handle (references existing context).
ContextHandle create_context_handle_ref(CUcontext ctx);

// ============================================================================
// Context acquisition functions (pure C++, nogil-safe)
// ============================================================================

// Get handle to the primary context for a device (with thread-local caching)
// Returns empty handle on error (caller must check)
ContextHandle get_primary_context(int dev_id) noexcept;

// Get handle to the current CUDA context
// Returns empty handle if no context is current (caller must check)
ContextHandle get_current_context() noexcept;

// ============================================================================
// Helper functions to extract raw resources from handles
// These are defined as inline C++ functions to support overloading when
// additional handle types (e.g., StreamHandle) are added.
// ============================================================================

// native() - extract the raw CUDA handle
inline CUcontext native(const ContextHandle& h) noexcept {
    return h ? *h : nullptr;
}

// intptr() - extract handle as uintptr_t for Python interop
inline std::uintptr_t intptr(const ContextHandle& h) noexcept {
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

}  // namespace cuda_core
