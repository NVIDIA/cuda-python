// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include "resource_handles.hpp"
#include <cuda.h>
#include <vector>

namespace cuda_core {

// Helper to release the GIL while calling into the CUDA driver.
// This guard is *conditional*: if the caller already dropped the GIL,
// we avoid calling PyEval_SaveThread (which requires holding the GIL).
// It also handles the case where Python is finalizing and GIL operations
// are no longer safe.
class GILReleaseGuard {
public:
    GILReleaseGuard() : tstate_(nullptr), released_(false) {
        // Don't try to manipulate GIL if Python is finalizing
        if (!Py_IsInitialized() || _Py_IsFinalizing()) {
            return;
        }
        // PyGILState_Check() returns 1 if the GIL is held by this thread.
        if (PyGILState_Check()) {
            tstate_ = PyEval_SaveThread();
            released_ = true;
        }
    }

    ~GILReleaseGuard() {
        if (released_) {
            PyEval_RestoreThread(tstate_);
        }
    }

private:
    PyThreadState* tstate_;
    bool released_;
};

// Internal box structure for Context (kept private to this TU)
struct ContextBox {
    CUcontext resource;
};

ContextHandle create_context_handle_ref(CUcontext ctx) {
    // Creates a non-owning handle that references an existing context
    // (e.g., primary context managed by CUDA driver)

    // Use default deleter - it will delete the box, but not touch the CUcontext
    // CUcontext lifetime is managed externally (e.g., by CUDA driver)
    auto box = new ContextBox{ctx};
    auto box_ptr = std::shared_ptr<const ContextBox>(box);

    // Use aliasing constructor to create handle that exposes only CUcontext
    // The handle's reference count is tied to box_ptr, but it points to &box_ptr->resource
    return ContextHandle(box_ptr, &box_ptr->resource);
}

// Thread-local storage for primary context cache
// Each thread maintains its own cache of primary contexts indexed by device ID
thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle get_primary_context(int dev_id) noexcept {
    // Check thread-local cache
    if (static_cast<size_t>(dev_id) < primary_context_cache.size()) {
        auto cached = primary_context_cache[dev_id];
        if (cached.get() != nullptr) {
            return cached;  // Cache hit
        }
    }

    // Cache miss - acquire primary context from driver
    CUcontext ctx;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuDevicePrimaryCtxRetain(&ctx, dev_id);
    }
    if (err != CUDA_SUCCESS) {
        // Return empty handle on error (caller must check)
        return ContextHandle();
    }

    // Create owning handle with custom deleter that releases the primary context
    auto box = new ContextBox{ctx};
    auto box_ptr = std::shared_ptr<const ContextBox>(box, [dev_id](const ContextBox* b) {
        GILReleaseGuard gil;
        cuDevicePrimaryCtxRelease(dev_id);
        delete b;
    });

    // Use aliasing constructor to expose only CUcontext
    auto h_context = ContextHandle(box_ptr, &box_ptr->resource);

    // Resize cache if needed
    if (static_cast<size_t>(dev_id) >= primary_context_cache.size()) {
        primary_context_cache.resize(dev_id + 1);
    }
    primary_context_cache[dev_id] = h_context;

    return h_context;
}

ContextHandle get_current_context() noexcept {
    CUcontext ctx = nullptr;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuCtxGetCurrent(&ctx);
    }
    if (err != CUDA_SUCCESS || ctx == nullptr) {
        // Return empty handle if no current context or error
        return ContextHandle();
    }
    return create_context_handle_ref(ctx);
}

// ============================================================================
// Stream Handles
// ============================================================================

// TODO: Implement StreamH create_stream_handle(...) when Stream gets handle support

// ============================================================================
// Event Handles
// ============================================================================

// TODO: Implement EventH create_event_handle(...) when Event gets handle support

// ============================================================================
// Device Pointer Handles
// ============================================================================

// TODO: Implement DevicePtrH create_deviceptr_handle(...) when DevicePtr gets handle support

// ============================================================================
// Memory Pool Handles
// ============================================================================

// TODO: Implement MemPoolH create_mempool_handle(...) when MemPool gets handle support

}  // namespace cuda_core
