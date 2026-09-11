// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "context_scope.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace cuda_core::rt {

using namespace detail;

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
// Must be noexcept since it's called from a shared_ptr deleter.
static void clear_mempool_peer_access(CUmemoryPool pool) noexcept {
    try {
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
    } catch (...) {
        // Swallow exceptions - this is best-effort cleanup in destructor context
    }
}

static MemoryPoolHandle wrap_mempool_owned(CUmemoryPool pool) {
    auto box = std::shared_ptr<const MemoryPoolBox>(
        new MemoryPoolBox{pool},
        [](const MemoryPoolBox* b) {
            GILReleaseGuard gil;
            clear_mempool_peer_access(b->resource);
            pw_cuMemPoolDestroy(b->resource);
            delete b;
        }
    );
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle create_mempool_handle(const CUmemPoolProps& props) {
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuMemPoolCreate(&pool, &props))) {
        return {};
    }
    return wrap_mempool_owned(pool);
}

MemoryPoolHandle create_mempool_handle_ref(CUmemoryPool pool) {
    auto box = std::make_shared<const MemoryPoolBox>(MemoryPoolBox{pool});
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle get_device_mempool(int device_id) {
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = p_cuDeviceGetMemPool(&pool, device_id))) {
        return {};
    }
    return create_mempool_handle_ref(pool);
}

MemoryPoolHandle create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type) {
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
    // Mutable so set_deallocation_stream() can update free ordering through a
    // const DevicePtrHandle. Built with make_deallocation_stream so default-
    // stream tokens carry a bound context.
    mutable DeallocationStream deallocation;
};
}  // namespace

// Recovers the owning DevicePtrBox from the aliased CUdeviceptr pointer.
// This works because DevicePtrHandle is a shared_ptr alias pointing to
// &box->resource, so we can compute the containing struct using offsetof.
// The const_cast is safe because we only use this to access the mutable
// deallocation member or in the deleter (where the box is being destroyed).
static DevicePtrBox* get_box(const DevicePtrHandle& h) {
    const CUdeviceptr* p = h.get();
    return reinterpret_cast<DevicePtrBox*>(
        reinterpret_cast<char*>(const_cast<CUdeviceptr*>(p)) - offsetof(DevicePtrBox, resource)
    );
}

// Return the stream that orders a device pointer's deallocation.
StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept {
    return get_box(h)->deallocation.h_stream;
}

// Replace the stream that orders a device pointer's deallocation.
CUresult set_deallocation_stream(const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept {
    if (!h) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        return err != CUDA_SUCCESS ? err : CUDA_ERROR_INVALID_CONTEXT;
    }
    get_box(h)->deallocation = std::move(ds);
    return CUDA_SUCCESS;
}

DevicePtrHandle deviceptr_alloc_from_pool(size_t size, const MemoryPoolHandle& h_pool, const StreamHandle& h_stream) {
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocFromPoolAsync(&ptr, size, *h_pool, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_pool](DevicePtrBox* b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync",
                [&]() noexcept {
                    return p_cuMemFreeAsync(
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_alloc_async(size_t size, const StreamHandle& h_stream) {
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocAsync(&ptr, size, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync",
                [&]() noexcept {
                    return p_cuMemFreeAsync(
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// Allocate device memory synchronously with the provided context current.
CUresult deviceptr_alloc_raw(CUdeviceptr* ptr, size_t size,
                             const ContextHandle& h_context) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return p_cuMemAlloc(ptr, size); },
        [&]() noexcept { pw_cuMemFree(*ptr); },
        /*undo_requires_target_context=*/false);
}

DevicePtrHandle deviceptr_alloc_host(size_t size) {
    GILReleaseGuard gil;
    void* ptr;
    if (CUDA_SUCCESS != (err = p_cuMemAllocHost(&ptr, size))) {
        return {};
    }

    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{reinterpret_cast<CUdeviceptr>(ptr), DeallocationStream{}},
        [](DevicePtrBox* b) {
            GILReleaseGuard gil;
            pw_cuMemFreeHost(reinterpret_cast<void*>(b->resource));
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_ref(CUdeviceptr ptr) {
    auto box = std::make_shared<DevicePtrBox>(DevicePtrBox{ptr, DeallocationStream{}});
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle deviceptr_create_with_owner(CUdeviceptr ptr, PyObject* owner) {
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
        new DevicePtrBox{ptr, DeallocationStream{}},
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

DevicePtrHandle deviceptr_create_mapped_graphics(
    CUdeviceptr ptr,
    const GraphicsResourceHandle& h_resource,
    const StreamHandle& h_stream
) {
    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        return {};
    }
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_resource](DevicePtrBox* b) {
            GILReleaseGuard gil;
            CUgraphicsResource resource = as_cu(h_resource);
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuGraphicsUnmapResources",
                [&]() noexcept {
                    return p_cuGraphicsUnmapResources(
                        1, &resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// MemoryResource-owned Device Pointer Handles
// ============================================================================

static MRDeallocCallback mr_dealloc_cb = nullptr;

void register_mr_dealloc_callback(MRDeallocCallback cb) {
    mr_dealloc_cb = cb;
}

DevicePtrHandle deviceptr_create_with_mr(CUdeviceptr ptr, size_t size, PyObject* mr) {
    if (!mr) {
        return deviceptr_create_ref(ptr);
    }
    // GIL required when mr is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return deviceptr_create_ref(ptr);
    }
    Py_INCREF(mr);
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, DeallocationStream{}},
        [mr, size](DevicePtrBox* b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                if (mr_dealloc_cb) {
                    const DeallocationStream& stream = b->deallocation;
                    cleanup_in_context(
                        deallocation_context(stream), "MemoryResource.deallocate",
                        [&]() noexcept {
                            mr_dealloc_cb(mr, b->resource, size, stream.h_stream);
                            return CUDA_SUCCESS;
                        });
                }
                Py_DECREF(mr);
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

static HandleRegistry<ExportDataKey, DevicePtrHandle, ExportDataKeyHash> ipc_ptr_cache;
static std::mutex ipc_import_mutex;

DevicePtrHandle deviceptr_import_ipc(const MemoryPoolHandle& h_pool, const void* export_data, const StreamHandle& h_stream) {
    auto data = const_cast<CUmemPoolPtrExportData*>(
        reinterpret_cast<const CUmemPoolPtrExportData*>(export_data));

    if (use_ipc_ptr_cache()) {
        ExportDataKey key;
        std::memcpy(&key.data, data, sizeof(key.data));

        std::lock_guard<std::mutex> lock(ipc_import_mutex);

        if (auto h = ipc_ptr_cache.lookup(key)) {
            return h;
        }

        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        DeallocationStream ds;
        if (!make_deallocation_stream(h_stream, ds)) {
            pw_cuMemFreeAsync(ptr, as_cu(h_stream));
            return {};
        }

        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, std::move(ds)},
            [h_pool, key](DevicePtrBox* b) {
                ipc_ptr_cache.unregister_handle(key);
                GILReleaseGuard gil;
                const DeallocationStream& stream = b->deallocation;
                cleanup_in_context(
                    deallocation_context(stream), "cuMemFreeAsync",
                    [&]() noexcept {
                        return p_cuMemFreeAsync(
                            b->resource, as_cu(stream.h_stream));
                    });
                delete b;
            }
        );
        DevicePtrHandle h(box, &box->resource);
        ipc_ptr_cache.register_handle(key, h);
        return h;

    } else {
        GILReleaseGuard gil;
        CUdeviceptr ptr;
        if (CUDA_SUCCESS != (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {
            return {};
        }

        DeallocationStream ds;
        if (!make_deallocation_stream(h_stream, ds)) {
            pw_cuMemFreeAsync(ptr, as_cu(h_stream));
            return {};
        }

        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{ptr, std::move(ds)},
            [h_pool](DevicePtrBox* b) {
                GILReleaseGuard gil;
                const DeallocationStream& stream = b->deallocation;
                cleanup_in_context(
                    deallocation_context(stream), "cuMemFreeAsync",
                    [&]() noexcept {
                        return p_cuMemFreeAsync(
                            b->resource, as_cu(stream.h_stream));
                    });
                delete b;
            }
        );
        return DevicePtrHandle(box, &box->resource);
    }
}

// ============================================================================
// File Descriptor Handles
// ============================================================================

FileDescriptorHandle create_fd_handle(int fd) {
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle is not supported on Windows");
#else
    return FileDescriptorHandle(
        new int(fd),
        [](const int* p) {
            if (::close(*p) != 0) {
                report_message("close() failed for an IPC file descriptor; the descriptor may have leaked");
            }
            delete p;
        }
    );
#endif
}

FileDescriptorHandle create_fd_handle_ref(int fd) {
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle_ref is not supported on Windows");
#else
    return std::make_shared<const int>(fd);
#endif
}

}  // namespace cuda_core::rt
