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

    // Non-copyable, non-movable
    GILReleaseGuard(const GILReleaseGuard&) = delete;
    GILReleaseGuard& operator=(const GILReleaseGuard&) = delete;

private:
    PyThreadState* tstate_;
    bool released_;
};

// Helper to acquire the GIL when we might not hold it.
// Use in C++ destructors that need to manipulate Python objects.
// Symmetric counterpart to GILReleaseGuard.
class GILAcquireGuard {
public:
    GILAcquireGuard() : acquired_(false) {
        // Don't try to acquire GIL if Python is finalizing
        if (!Py_IsInitialized() || _Py_IsFinalizing()) {
            return;
        }
        gstate_ = PyGILState_Ensure();
        acquired_ = true;
    }

    ~GILAcquireGuard() {
        if (acquired_) {
            PyGILState_Release(gstate_);
        }
    }

    // Check if GIL was successfully acquired (for conditional operations)
    bool acquired() const { return acquired_; }

    // Non-copyable, non-movable
    GILAcquireGuard(const GILAcquireGuard&) = delete;
    GILAcquireGuard& operator=(const GILAcquireGuard&) = delete;

private:
    PyGILState_STATE gstate_;
    bool acquired_;
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
    auto box = std::shared_ptr<const ContextBox>(new ContextBox{ctx});

    // Use aliasing constructor to create handle that exposes only CUcontext
    // The handle's reference count is tied to box, but it points to &box->resource
    return ContextHandle(box, &box->resource);
}

// Thread-local storage for primary context cache
// Each thread maintains its own cache of primary contexts indexed by device ID
thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle get_primary_context(int device_id) noexcept {
    // Check thread-local cache
    if (static_cast<size_t>(device_id) < primary_context_cache.size()) {
        auto cached = primary_context_cache[device_id];
        if (cached.get() != nullptr) {
            return cached;  // Cache hit
        }
    }

    // Cache miss - acquire primary context from driver
    CUcontext ctx;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuDevicePrimaryCtxRetain(&ctx, device_id);
    }
    if (err != CUDA_SUCCESS) {
        // Return empty handle on error (caller must check)
        return ContextHandle();
    }

    // Create owning handle with custom deleter that releases the primary context
    auto box = std::shared_ptr<const ContextBox>(new ContextBox{ctx}, [device_id](const ContextBox* b) {
        GILReleaseGuard gil;
        cuDevicePrimaryCtxRelease(device_id);
        delete b;
    });

    // Use aliasing constructor to expose only CUcontext
    auto h_context = ContextHandle(box, &box->resource);

    // Resize cache if needed
    if (static_cast<size_t>(device_id) >= primary_context_cache.size()) {
        primary_context_cache.resize(device_id + 1);
    }
    primary_context_cache[device_id] = h_context;

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

// Internal box structure for Stream
struct StreamBox {
    CUstream resource;
};

StreamHandle create_stream_handle(ContextHandle h_ctx, unsigned int flags, int priority) {
    // Creates an owning stream handle - calls cuStreamCreateWithPriority internally.
    // The context handle is captured in the deleter to ensure context outlives the stream.
    // Returns empty handle on error (caller must check).
    CUstream stream;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuStreamCreateWithPriority(&stream, flags, priority);
    }
    if (err != CUDA_SUCCESS) {
        return StreamHandle();
    }

    // Capture h_ctx in lambda - shared_ptr control block keeps it alive
    auto box = std::shared_ptr<const StreamBox>(new StreamBox{stream}, [h_ctx](const StreamBox* b) {
        GILReleaseGuard gil;
        cuStreamDestroy(b->resource);
        delete b;
        // h_ctx destructor runs here when last stream reference is released
    });

    // Use aliasing constructor to expose only CUstream
    return StreamHandle(box, &box->resource);
}

StreamHandle create_stream_handle_ref(CUstream stream) {
    // Creates a non-owning handle - stream will NOT be destroyed.
    // Caller is responsible for keeping the stream's context alive.
    auto box = std::shared_ptr<const StreamBox>(new StreamBox{stream});

    // Use aliasing constructor to expose only CUstream
    return StreamHandle(box, &box->resource);
}

StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner) {
    // Creates a non-owning handle that prevents a Python owner from being GC'd.
    // The owner's refcount is incremented here and decremented when handle is released.
    // The owner is responsible for keeping the stream's context alive.
    Py_XINCREF(owner);

    auto box = std::shared_ptr<const StreamBox>(new StreamBox{stream}, [owner](const StreamBox* b) {
        // Safely decrement owner refcount (GILAcquireGuard handles finalization check)
        {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_XDECREF(owner);
            }
        }
        delete b;
    });

    return StreamHandle(box, &box->resource);
}

StreamHandle get_legacy_stream() noexcept {
    // Return non-owning handle to the legacy default stream.
    // Use function-local static for efficient repeated access.
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_LEGACY);
    return handle;
}

StreamHandle get_per_thread_stream() noexcept {
    // Return non-owning handle to the per-thread default stream.
    // Use function-local static for efficient repeated access.
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_PER_THREAD);
    return handle;
}

// ============================================================================
// Event Handles
// ============================================================================

// Internal box structure for Event
struct EventBox {
    CUevent resource;
};

EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags) {
    // Creates an owning event handle - calls cuEventCreate internally.
    // The context handle is captured in the deleter to ensure context outlives the event.
    // Returns empty handle on error (caller must check).
    CUevent event;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuEventCreate(&event, flags);
    }
    if (err != CUDA_SUCCESS) {
        return EventHandle();
    }

    // Capture h_ctx in lambda - shared_ptr control block keeps it alive
    auto box = std::shared_ptr<const EventBox>(new EventBox{event}, [h_ctx](const EventBox* b) {
        GILReleaseGuard gil;
        cuEventDestroy(b->resource);
        delete b;
        // h_ctx destructor runs here when last event reference is released
    });

    // Use aliasing constructor to expose only CUevent
    return EventHandle(box, &box->resource);
}

EventHandle create_event_handle(unsigned int flags) {
    // Creates an owning event handle without context dependency.
    // Use for temporary events that are created and destroyed in the same scope.
    // Returns empty handle on error (caller must check).
    return create_event_handle(ContextHandle{}, flags);
}

EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle) {
    // Creates an owning event handle from an IPC handle.
    // The originating process owns the event and its context.
    // Returns empty handle on error (caller must check).
    CUevent event;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuIpcOpenEventHandle(&event, ipc_handle);
    }
    if (err != CUDA_SUCCESS) {
        return EventHandle();
    }

    auto box = std::shared_ptr<const EventBox>(new EventBox{event}, [](const EventBox* b) {
        GILReleaseGuard gil;
        cuEventDestroy(b->resource);
        delete b;
    });

    // Use aliasing constructor to expose only CUevent
    return EventHandle(box, &box->resource);
}

// ============================================================================
// Memory Pool Handles
// ============================================================================

// Internal box structure for MemoryPool
struct MemoryPoolBox {
    CUmemoryPool resource;
};

// Helper to clear peer access before destroying a memory pool.
// Works around nvbug 5698116: recycled pool handles inherit peer access state.
static void clear_mempool_peer_access(CUmemoryPool pool) {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        return;
    }

    std::vector<CUmemAccessDesc> clear_access(device_count);
    for (int i = 0; i < device_count; ++i) {
        clear_access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        clear_access[i].location.id = i;
        clear_access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    }

    // Ignore errors - best effort cleanup
    cuMemPoolSetAccess(pool, clear_access.data(), device_count);
}

// Helper to wrap a raw pool in an owning handle.
// The deleter clears peer access (nvbug 5698116 workaround) and destroys the pool.
static MemoryPoolHandle wrap_mempool_owned(CUmemoryPool pool) {
    auto box = std::shared_ptr<const MemoryPoolBox>(new MemoryPoolBox{pool}, [](const MemoryPoolBox* b) {
        GILReleaseGuard gil;
        clear_mempool_peer_access(b->resource);
        cuMemPoolDestroy(b->resource);
        delete b;
    });
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props) {
    // Creates an owning memory pool handle - calls cuMemPoolCreate internally.
    // Memory pools are device-scoped (not context-scoped).
    // Returns empty handle on error (caller must check).
    CUmemoryPool pool;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuMemPoolCreate(&pool, &props);
    }
    return err == CUDA_SUCCESS ? wrap_mempool_owned(pool) : MemoryPoolHandle();
}

MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool) {
    // Creates a non-owning handle - pool will NOT be destroyed.
    // Use for device default/current pools managed by the driver.
    auto box = std::shared_ptr<const MemoryPoolBox>(new MemoryPoolBox{pool});

    // Use aliasing constructor to expose only CUmemoryPool
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle get_device_mempool(int device_id) noexcept {
    // Get the current memory pool for a device.
    // Returns a non-owning handle (pool managed by driver).
    CUmemoryPool pool;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuDeviceGetMemPool(&pool, device_id);
    }
    if (err != CUDA_SUCCESS) {
        return MemoryPoolHandle();
    }
    return create_mempool_handle_ref(pool);
}

MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type) {
    // Creates an owning memory pool handle from an IPC import.
    // The file descriptor is NOT owned by this handle.
    // Returns empty handle on error (caller must check).
    CUmemoryPool pool;
    CUresult err;
    {
        GILReleaseGuard gil;
        err = cuMemPoolImportFromShareableHandle(&pool, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), handle_type, 0);
    }
    return err == CUDA_SUCCESS ? wrap_mempool_owned(pool) : MemoryPoolHandle();
}

}  // namespace cuda_core
