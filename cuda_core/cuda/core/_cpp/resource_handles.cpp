// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include "resource_handles.hpp"
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda_core {

// ============================================================================
// CUDA driver lazy resolution via cuda-bindings (CPU-only import + MVC)
// ============================================================================

namespace {

std::once_flag driver_load_once;
std::atomic<bool> driver_loaded{false};

#if PY_VERSION_HEX < 0x030D0000
extern "C" int _Py_IsFinalizing(void);
#endif

inline bool py_is_finalizing() noexcept {
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing();
#else
    // Python < 3.13 does not expose Py_IsFinalizing() publicly. Use the private
    // API that exists in those versions.
    return _Py_IsFinalizing() != 0;
#endif
}

// ============================================================================
// GIL management helpers
// ============================================================================

// Helper to release the GIL while calling into the CUDA driver.
// This guard is *conditional*: if the caller already dropped the GIL,
// we avoid calling PyEval_SaveThread (which requires holding the GIL).
// It also handles the case where Python is finalizing and GIL operations
// are no longer safe.
class GILReleaseGuard {
public:
    GILReleaseGuard() : tstate_(nullptr), released_(false) {
        // Don't try to manipulate GIL if Python is finalizing
        if (!Py_IsInitialized() || py_is_finalizing()) {
            return;
        }
        // PyGILState_Check() returns 1 if the GIL is held by this thread.
        if (PyGILState_Check()) {
            tstate_ = PyEval_SaveThread();
            released_ = true;
        }
        // Note: If the GIL is not released (finalizing, or not held):
        // - Reduces parallelism (other Python threads remain blocked)
        // - No deadlock risk as long as the guarded code doesn't call back into Python
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
class GILAcquireGuard {
public:
    GILAcquireGuard() : acquired_(false) {
        // Don't try to acquire GIL if Python is finalizing
        if (!Py_IsInitialized() || py_is_finalizing()) {
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

    bool acquired() const { return acquired_; }

    // Non-copyable, non-movable
    GILAcquireGuard(const GILAcquireGuard&) = delete;
    GILAcquireGuard& operator=(const GILAcquireGuard&) = delete;

private:
    PyGILState_STATE gstate_;
    bool acquired_;
};


#define DECLARE_DRIVER_FN(name) using name##_t = decltype(&name); name##_t p_##name = nullptr

DECLARE_DRIVER_FN(cuDevicePrimaryCtxRetain);
DECLARE_DRIVER_FN(cuDevicePrimaryCtxRelease);
DECLARE_DRIVER_FN(cuCtxGetCurrent);

DECLARE_DRIVER_FN(cuStreamCreateWithPriority);
DECLARE_DRIVER_FN(cuStreamDestroy);

DECLARE_DRIVER_FN(cuEventCreate);
DECLARE_DRIVER_FN(cuEventDestroy);
DECLARE_DRIVER_FN(cuIpcOpenEventHandle);

DECLARE_DRIVER_FN(cuDeviceGetCount);

DECLARE_DRIVER_FN(cuMemPoolSetAccess);
DECLARE_DRIVER_FN(cuMemPoolDestroy);
DECLARE_DRIVER_FN(cuMemPoolCreate);
DECLARE_DRIVER_FN(cuDeviceGetMemPool);
DECLARE_DRIVER_FN(cuMemPoolImportFromShareableHandle);

DECLARE_DRIVER_FN(cuMemAllocFromPoolAsync);
DECLARE_DRIVER_FN(cuMemAllocAsync);
DECLARE_DRIVER_FN(cuMemAlloc);
DECLARE_DRIVER_FN(cuMemAllocHost);

DECLARE_DRIVER_FN(cuMemFreeAsync);
DECLARE_DRIVER_FN(cuMemFree);
DECLARE_DRIVER_FN(cuMemFreeHost);

DECLARE_DRIVER_FN(cuMemPoolImportPointer);

#undef DECLARE_DRIVER_FN

bool load_driver_api() noexcept {
    struct CudaDriverApiV1 {
        std::uint32_t abi_version;
        std::uint32_t struct_size;

        std::uintptr_t cuDevicePrimaryCtxRetain;
        std::uintptr_t cuDevicePrimaryCtxRelease;
        std::uintptr_t cuCtxGetCurrent;

        std::uintptr_t cuStreamCreateWithPriority;
        std::uintptr_t cuStreamDestroy;

        std::uintptr_t cuEventCreate;
        std::uintptr_t cuEventDestroy;
        std::uintptr_t cuIpcOpenEventHandle;

        std::uintptr_t cuDeviceGetCount;

        std::uintptr_t cuMemPoolSetAccess;
        std::uintptr_t cuMemPoolDestroy;
        std::uintptr_t cuMemPoolCreate;
        std::uintptr_t cuDeviceGetMemPool;
        std::uintptr_t cuMemPoolImportFromShareableHandle;

        std::uintptr_t cuMemAllocFromPoolAsync;
        std::uintptr_t cuMemAllocAsync;
        std::uintptr_t cuMemAlloc;
        std::uintptr_t cuMemAllocHost;

        std::uintptr_t cuMemFreeAsync;
        std::uintptr_t cuMemFree;
        std::uintptr_t cuMemFreeHost;

        std::uintptr_t cuMemPoolImportPointer;
    };

    static constexpr const char* capsule_name =
        "cuda.core._resource_handles._CUDA_DRIVER_API_V1";

    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return false;
    }

    // `_resource_handles` is already loaded (it exports the handle API capsule),
    // so avoid import machinery and just grab the module object.
    PyObject* mod = PyImport_AddModule("cuda.core._resource_handles");  // borrowed
    if (!mod) {
        PyErr_Clear();
        return false;
    }

    PyObject* fn = PyObject_GetAttrString(mod, "_get_cuda_driver_api_v1_capsule");  // new ref
    if (!fn) {
        PyErr_Clear();
        return false;
    }

    PyObject* cap = PyObject_CallFunctionObjArgs(fn, nullptr);
    Py_DECREF(fn);
    if (!cap) {
        PyErr_Clear();
        return false;
    }

    const auto* api = static_cast<const CudaDriverApiV1*>(PyCapsule_GetPointer(cap, capsule_name));
    Py_DECREF(cap);

    if (!api) {
        PyErr_Clear();
        return false;
    }
    if (api->abi_version != 1 || api->struct_size < sizeof(CudaDriverApiV1)) {
        return false;
    }

#define LOAD_ADDR(name)                                             \
    do {                                                            \
        if (api->name == 0) {                                       \
            return false;                                           \
        }                                                           \
        p_##name = reinterpret_cast<decltype(p_##name)>(api->name); \
    } while (0)

    LOAD_ADDR(cuDevicePrimaryCtxRetain);
    LOAD_ADDR(cuDevicePrimaryCtxRelease);
    LOAD_ADDR(cuCtxGetCurrent);

    LOAD_ADDR(cuStreamCreateWithPriority);
    LOAD_ADDR(cuStreamDestroy);

    LOAD_ADDR(cuEventCreate);
    LOAD_ADDR(cuEventDestroy);
    LOAD_ADDR(cuIpcOpenEventHandle);

    LOAD_ADDR(cuDeviceGetCount);

    LOAD_ADDR(cuMemPoolSetAccess);
    LOAD_ADDR(cuMemPoolDestroy);
    LOAD_ADDR(cuMemPoolCreate);
    LOAD_ADDR(cuDeviceGetMemPool);
    LOAD_ADDR(cuMemPoolImportFromShareableHandle);

    LOAD_ADDR(cuMemAllocFromPoolAsync);
    LOAD_ADDR(cuMemAllocAsync);
    LOAD_ADDR(cuMemAlloc);
    LOAD_ADDR(cuMemAllocHost);

    LOAD_ADDR(cuMemFreeAsync);
    LOAD_ADDR(cuMemFree);
    LOAD_ADDR(cuMemFreeHost);

    LOAD_ADDR(cuMemPoolImportPointer);

#undef LOAD_ADDR

    return true;
}

bool ensure_driver_loaded() noexcept {
    // Fast path: already loaded (no locking needed)
    if (driver_loaded.load(std::memory_order_acquire)) {
        return true;
    }

    // Slow path: release GIL before acquiring call_once guard.
    // This ensures lock order is always: guard mutex -> GIL, preventing deadlock.
    // See DESIGN.md "Static Initialization and Deadlock Hazards".
    GILReleaseGuard release_gil;
    std::call_once(driver_load_once, []() {
        // Inside call_once, safe to acquire GIL (correct lock order).
        // load_driver_api() acquires GIL internally via GILAcquireGuard.
        driver_loaded.store(load_driver_api(), std::memory_order_release);
    });
    return driver_loaded.load(std::memory_order_acquire);
}

}  // namespace

// ============================================================================
// Thread-local error handling
// ============================================================================

// Thread-local status of the most recent CUDA API call in this module.
static thread_local CUresult err = CUDA_SUCCESS;

CUresult get_last_error() noexcept {
    CUresult e = err;
    err = CUDA_SUCCESS;
    return e;
}

CUresult peek_last_error() noexcept {
    return err;
}

void clear_last_error() noexcept {
    err = CUDA_SUCCESS;
}

// ============================================================================
// Context Handles
// ============================================================================

namespace {
struct ContextBox {
    CUcontext resource;
};
}  // namespace

ContextHandle create_context_handle_ref(CUcontext ctx) noexcept {
    auto box = std::make_shared<const ContextBox>(ContextBox{ctx});
    return ContextHandle(box, &box->resource);
}

// Thread-local cache of primary contexts indexed by device ID
static thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle get_primary_context(int device_id) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    // Check thread-local cache
    if (static_cast<size_t>(device_id) < primary_context_cache.size()) {
        if (auto cached = primary_context_cache[device_id]) {
            return cached;
        }
    }

    // Cache miss - acquire primary context from driver
    GILReleaseGuard gil;
    CUcontext ctx;
    if (CUDA_SUCCESS != (err = p_cuDevicePrimaryCtxRetain(&ctx, device_id))) {
        return {};
    }

    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx},
        [device_id](const ContextBox* b) {
            GILReleaseGuard gil;
            p_cuDevicePrimaryCtxRelease(device_id);
            delete b;
        }
    );
    auto h = ContextHandle(box, &box->resource);

    // Update cache
    if (static_cast<size_t>(device_id) >= primary_context_cache.size()) {
        primary_context_cache.resize(device_id + 1);
    }
    primary_context_cache[device_id] = h;
    return h;
}

ContextHandle get_current_context() noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuCtxGetCurrent(&ctx))) {
        return {};
    }
    if (!ctx) {
        return {};  // No current context (not an error)
    }
    return create_context_handle_ref(ctx);
}

// ============================================================================
// Stream Handles
// ============================================================================

namespace {
struct StreamBox {
    CUstream resource;
};
}  // namespace

StreamHandle create_stream_handle(ContextHandle h_ctx, unsigned int flags, int priority) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUstream stream;
    if (CUDA_SUCCESS != (err = p_cuStreamCreateWithPriority(&stream, flags, priority))) {
        return {};
    }

    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream},
        [h_ctx](const StreamBox* b) {
            GILReleaseGuard gil;
            p_cuStreamDestroy(b->resource);
            delete b;
        }
    );
    return StreamHandle(box, &box->resource);
}

StreamHandle create_stream_handle_ref(CUstream stream) noexcept {
    auto box = std::make_shared<const StreamBox>(StreamBox{stream});
    return StreamHandle(box, &box->resource);
}

StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner) noexcept {
    if (!owner) {
        return create_stream_handle_ref(stream);
    }
    // GIL required when owner is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        // Python finalizing - fall back to ref version (no owner tracking)
        return create_stream_handle_ref(stream);
    }
    Py_INCREF(owner);
    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream},
        [owner](const StreamBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_DECREF(owner);
            }
            delete b;
        }
    );
    return StreamHandle(box, &box->resource);
}

StreamHandle get_legacy_stream() noexcept {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_LEGACY);
    return handle;
}

StreamHandle get_per_thread_stream() noexcept {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_PER_THREAD);
    return handle;
}

// ============================================================================
// Event Handles
// ============================================================================

namespace {
struct EventBox {
    CUevent resource;
};
}  // namespace

EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUevent event;
    if (CUDA_SUCCESS != (err = p_cuEventCreate(&event, flags))) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event},
        [h_ctx](const EventBox* b) {
            GILReleaseGuard gil;
            p_cuEventDestroy(b->resource);
            delete b;
        }
    );
    return EventHandle(box, &box->resource);
}

EventHandle create_event_handle_noctx(unsigned int flags) noexcept {
    return create_event_handle(ContextHandle{}, flags);
}

EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUevent event;
    if (CUDA_SUCCESS != (err = p_cuIpcOpenEventHandle(&event, ipc_handle))) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event},
        [](const EventBox* b) {
            GILReleaseGuard gil;
            p_cuEventDestroy(b->resource);
            delete b;
        }
    );
    return EventHandle(box, &box->resource);
}

// ============================================================================
// Memory Pool Handles
// ============================================================================

namespace {
struct MemoryPoolBox {
    CUmemoryPool resource;
};
}  // namespace

// Helper to clear peer access before destroying a memory pool.
// Works around nvbug 5698116: recycled pool handles inherit peer access state.
static void clear_mempool_peer_access(CUmemoryPool pool) {
    int device_count = 0;
    if (p_cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        return;
    }

    std::vector<CUmemAccessDesc> clear_access(device_count);
    for (int i = 0; i < device_count; ++i) {
        clear_access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        clear_access[i].location.id = i;
        clear_access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    }
    p_cuMemPoolSetAccess(pool, clear_access.data(), device_count);  // Best effort
}

static MemoryPoolHandle wrap_mempool_owned(CUmemoryPool pool) {
    auto box = std::shared_ptr<const MemoryPoolBox>(
        new MemoryPoolBox{pool},
        [](const MemoryPoolBox* b) {
            GILReleaseGuard gil;
            clear_mempool_peer_access(b->resource);
            p_cuMemPoolDestroy(b->resource);
            delete b;
        }
    );
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuMemPoolCreate(&pool, &props))) {
        return {};
    }
    return wrap_mempool_owned(pool);
}

MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool) noexcept {
    auto box = std::make_shared<const MemoryPoolBox>(MemoryPoolBox{pool});
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle get_device_mempool(int device_id) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuDeviceGetMemPool(&pool, device_id))) {
        return {};
    }
    return create_mempool_handle_ref(pool);
}

MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUmemoryPool pool;
    auto handle_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(fd));
    if (CUDA_SUCCESS != (err = p_cuMemPoolImportFromShareableHandle(&pool, handle_ptr, handle_type, 0))) {
        return {};
    }
    return wrap_mempool_owned(pool);
}

// ============================================================================
// Device Pointer Handles
// ============================================================================

namespace {
struct DevicePtrBox {
    CUdeviceptr resource;
    // Mutable to allow set_deallocation_stream() to update the stream
    // through a const DevicePtrHandle. The stream can be changed after
    // allocation (e.g., to synchronize deallocation with a different stream).
    mutable StreamHandle h_stream;
};
}  // namespace

// Recovers the owning DevicePtrBox from the aliased CUdeviceptr pointer.
// This works because DevicePtrHandle is a shared_ptr alias pointing to
// &box->resource, so we can compute the containing struct using offsetof.
// The const_cast is safe because we only use this to access the mutable
// h_stream member or in the deleter (where the box is being destroyed).
static DevicePtrBox* get_box(const DevicePtrHandle& h) {
    const CUdeviceptr* p = h.get();
    return reinterpret_cast<DevicePtrBox*>(
        reinterpret_cast<char*>(const_cast<CUdeviceptr*>(p)) - offsetof(DevicePtrBox, resource)
    );
}

StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept {
    return get_box(h)->h_stream;
}

void set_deallocation_stream(const DevicePtrHandle& h, StreamHandle h_stream) noexcept {
    get_box(h)->h_stream = std::move(h_stream);
}

DevicePtrHandle deviceptr_alloc_from_pool(size_t size, MemoryPoolHandle h_pool, StreamHandle h_stream) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocFromPoolAsync(&ptr, size, *h_pool, as_cu(h_stream)))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, h_stream},
        [h_pool](DevicePtrBox* b) {
            GILReleaseGuard gil;
            p_cuMemFreeAsync(b->resource, as_cu(b->h_stream));
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_alloc_async(size_t size, StreamHandle h_stream) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocAsync(&ptr, size, as_cu(h_stream)))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, h_stream},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            p_cuMemFreeAsync(b->resource, as_cu(b->h_stream));
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_alloc(size_t size) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAlloc(&ptr, size))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, StreamHandle{}},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            p_cuMemFree(b->resource);
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_alloc_host(size_t size) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }
    GILReleaseGuard gil;
    void* ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocHost(&ptr, size))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{reinterpret_cast<CUdeviceptr>(ptr), StreamHandle{}},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            p_cuMemFreeHost(reinterpret_cast<void*>(b->resource));
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_ref(CUdeviceptr ptr) noexcept {
    auto box = std::make_shared<DevicePtrBox>(DevicePtrBox{ptr, StreamHandle{}});
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_with_owner(CUdeviceptr ptr, PyObject* owner) noexcept {
    if (!owner) {
        return deviceptr_create_ref(ptr);
    }
    // GIL required when owner is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        // Python finalizing - fall back to ref version (no owner tracking)
        return deviceptr_create_ref(ptr);
    }
    Py_INCREF(owner);
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, StreamHandle{}},
        [owner](DevicePtrBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_DECREF(owner);
            }
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// IPC Pointer Cache
// ============================================================================
// This cache handles duplicate IPC imports, which behave differently depending
// on the memory type:
//
// 1. Memory pool allocations (DeviceMemoryResource):
//    Multiple imports of the same allocation succeed and return duplicate
//    pointers. However, the driver has a reference counting bug (nvbug 5570902)
//    where the first cuMemFreeAsync incorrectly unmaps the memory even when
//    imported multiple times. A driver fix is expected.
//
// 2. Pinned memory allocations (PinnedMemoryResource):
//    Duplicate imports result in CUDA_ERROR_ALREADY_MAPPED.
//
// The cache solves both issues by checking the cache before calling
// cuMemPoolImportPointer and returning the existing handle for duplicate
// imports. This provides a consistent user experience where the same IPC
// descriptor can be imported multiple times regardless of memory type.
//
// The cache key is the export_data bytes (CUmemPoolPtrExportData), not the
// returned pointer, because we must check before calling the driver API.


// TODO: When driver fix for nvbug 5570902 is available, consider whether
// the cache is still needed for memory pool allocations (it will still be
// needed for pinned memory).
static bool use_ipc_ptr_cache() {
    return true;
}

namespace {
// Wrapper for CUmemPoolPtrExportData to use as map key
struct ExportDataKey {
    CUmemPoolPtrExportData data;

    bool operator==(const ExportDataKey& other) const {
        return std::memcmp(&data, &other.data, sizeof(data)) == 0;
    }
};

struct ExportDataKeyHash {
    std::size_t operator()(const ExportDataKey& key) const {
        // Simple hash of the bytes
        std::size_t h = 0;
        const auto* bytes = reinterpret_cast<const unsigned char*>(&key.data);
        for (std::size_t i = 0; i < sizeof(key.data); ++i) {
            h = h * 31 + bytes[i];
        }
        return h;
    }
};

}

static std::mutex ipc_ptr_cache_mutex;
static std::unordered_map<ExportDataKey, std::weak_ptr<DevicePtrBox>, ExportDataKeyHash> ipc_ptr_cache;

DevicePtrHandle deviceptr_import_ipc(MemoryPoolHandle h_pool, const void* export_data, StreamHandle h_stream) noexcept {
    if (!ensure_driver_loaded()) {
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }

    auto data = const_cast<CUmemPoolPtrExportData*>(
        reinterpret_cast<const CUmemPoolPtrExportData*>(export_data));

    if (use_ipc_ptr_cache()) {
        // Check cache before calling cuMemPoolImportPointer
        ExportDataKey key;
        std::memcpy(&key.data, data, sizeof(key.data));

        std::lock_guard<std::mutex> lock(ipc_ptr_cache_mutex);

        auto it = ipc_ptr_cache.find(key);
        if (it != ipc_ptr_cache.end()) {
            if (auto box = it->second.lock()) {
                // Cache hit - return existing handle
                return DevicePtrHandle(box, &box->resource);
            }
            ipc_ptr_cache.erase(it);  // Expired entry
        }

        // Cache miss - import the pointer
        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        // Create new handle with cache-clearing deleter
        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, h_stream},
            [h_pool, key](DevicePtrBox* b) {
                GILReleaseGuard gil;
                {
                    std::lock_guard<std::mutex> lock(ipc_ptr_cache_mutex);
                    // Only erase if expired - avoids race where another thread
                    // replaced the entry with a new import before we acquired the lock.
                    auto it = ipc_ptr_cache.find(key);
                    if (it != ipc_ptr_cache.end() && it->second.expired()) {
                        ipc_ptr_cache.erase(it);
                    }
                }
                p_cuMemFreeAsync(b->resource, as_cu(b->h_stream));
                delete b;
            }
        );
        ipc_ptr_cache[key] = box;
        return DevicePtrHandle(box, &box->resource);

    } else {
        // No caching - simple handle creation
        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, h_stream},
            [h_pool](DevicePtrBox* b) {
                GILReleaseGuard gil;
                p_cuMemFreeAsync(b->resource, as_cu(b->h_stream));
                delete b;
            }
        );
        return DevicePtrHandle(box, &box->resource);
    }
}

}  // namespace cuda_core
