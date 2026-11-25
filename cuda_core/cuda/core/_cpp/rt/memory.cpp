// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "context_scope.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
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

// Process-wide count of live *owned* memory pools -- those this layer created
// with cuMemPoolCreate or imported (and will cuMemPoolDestroy). Borrowed pool
// references (create_mempool_handle_ref, get_device_mempool) are not counted.
// Read by the test suite to gate the per-test gc.collect(); atomic for
// free-threaded builds.
std::atomic<long long> g_live_owned_mempools{0};
}  // namespace

// Helper to clear peer access before destroying a memory pool.
// Works around nvbug 5698116: recycled pool handles inherit peer access state.
// The driver validates a request as a whole and rejects it if any entry cannot
// be applied: the pool's own device, a device without memory-map support, or
// devices of more than one kind in one request. Each device is therefore
// revoked with its own request, and the owning device (-1 if the pool has
// none) is not requested at all.
// Must be noexcept since it's called from a shared_ptr deleter.
static void
clear_mempool_peer_access(CUmemoryPool pool, int owner_device) noexcept
{
    try {
        int device_count = 0;
        if (DRIVER_CALL(cuDeviceGetCount, &device_count) != CUDA_SUCCESS || device_count <= 0) {
            return;
        }

        CUmemAccessDesc revoke{};
        revoke.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        revoke.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
        for (int i = 0; i < device_count; ++i) {
            if (i == owner_device) {
                continue;
            }
            revoke.location.id = i;
            DRIVER_CALL(cuMemPoolSetAccess, pool, &revoke, 1);  // Best effort
        }
    }
    catch (...) {  // NOLINT(bugprone-empty-catch)
        // Swallow exceptions - this is best-effort cleanup in destructor context
    }
}

// owner_device is the ordinal of a device-located pool, or -1 when there is no
// device to skip (host pools) or the location is unknown (imported pools).
static MemoryPoolHandle
wrap_mempool_owned(CUmemoryPool pool, int owner_device)
{
    auto *raw = new MemoryPoolBox{pool};
    // Count before constructing the shared_ptr: if its control-block allocation
    // throws, it runs the deleter on raw, whose decrement then balances this.
    g_live_owned_mempools.fetch_add(1, std::memory_order_relaxed);
    // clang-format off
    auto box = std::shared_ptr<const MemoryPoolBox>(
        raw,
        [owner_device](const MemoryPoolBox *b) {
            GILReleaseGuard gil;
            clear_mempool_peer_access(b->resource, owner_device);
            pw_cuMemPoolDestroy(b->resource);
            g_live_owned_mempools.fetch_sub(1, std::memory_order_relaxed);
            delete b;
        }
    );
    // clang-format on
    return MemoryPoolHandle(box, &box->resource);
}

// Test-suite instrumentation; see g_live_owned_mempools.
long long
live_owned_mempool_count() noexcept
{
    return g_live_owned_mempools.load(std::memory_order_relaxed);
}

MemoryPoolHandle
create_mempool_handle(const CUmemPoolProps &props)
{
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemPoolCreate, &pool, &props))) {
        return {};
    }
    int owner_device = props.location.type == CU_MEM_LOCATION_TYPE_DEVICE ? props.location.id : -1;
    return wrap_mempool_owned(pool, owner_device);
}

MemoryPoolHandle
create_mempool_handle_ref(CUmemoryPool pool)
{
    auto box = std::make_shared<const MemoryPoolBox>(MemoryPoolBox{pool});
    return MemoryPoolHandle(box, &box->resource);
}

MemoryPoolHandle
get_device_mempool(int device_id)
{
    GILReleaseGuard gil;
    CUmemoryPool pool;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuDeviceGetMemPool, &pool, device_id))) {
        return {};
    }
    return create_mempool_handle_ref(pool);
}

// Importing a pool that is already imported in this process returns the same
// CUmemoryPool and adds a driver reference: each import needs its own
// cuMemPoolDestroy, and one made while allocations imported from the pool are
// outstanding fails without dropping its reference. Access settings belong to
// the pool, not to an import, so revoking them on behalf of one import
// (nvbug 5698116) would remove access that another import relies on (#3510).
// All imports of a pool therefore share one owner, found through
// ipc_pool_registry. It counts the imports and, when its last reference is
// released, revokes access once and then destroys every import. An import that
// finds the owner's release in flight waits for it, even when the last
// reference was dropped by a buffer and not by a resource the importer could
// synchronize with: the revocation would remove access granted by a new owner,
// and a destroy fails once the new owner has imported a buffer.
namespace {
struct IpcMemoryPoolBox {
    CUmemoryPool resource;
    // The imports this box owns: 0 until it becomes the pool's owner, then
    // one for every import of the pool.
    mutable std::atomic<std::size_t> imports{0};
};

const IpcMemoryPoolBox *
get_ipc_box(const MemoryPoolHandle &h)
{
    const CUmemoryPool *p = h.get();
    return reinterpret_cast<const IpcMemoryPoolBox *>(
        reinterpret_cast<const char *>(p) - offsetof(IpcMemoryPoolBox, resource)
    );
}
}  // namespace

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static ReleasingHandleRegistry<CUmemoryPool, MemoryPoolHandle> ipc_pool_registry;

MemoryPoolHandle
create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type)
{
    // Released before the registry can wait: the owner's deleter may need it.
    GILReleaseGuard gil;
    // Allocate before importing: a failed allocation then has nothing to undo.
    // clang-format off
    auto box = std::shared_ptr<IpcMemoryPoolBox>(
        new IpcMemoryPoolBox{},
        [](const IpcMemoryPoolBox *b) {
            // Shared ownership synchronizes the increments with this load.
            std::size_t imports = b->imports.load(std::memory_order_relaxed);
            if (imports > 0) {
                GILReleaseGuard gil;
                clear_mempool_peer_access(b->resource, -1);
                const std::size_t total = imports;
                std::size_t failed = 0;
                CUresult first_failure = CUDA_SUCCESS;
                for (; imports > 0; --imports) {
                    if (CUresult s = DRIVER_CALL(cuMemPoolDestroy, b->resource); s != CUDA_SUCCESS) {
                        if (failed++ == 0) {
                            first_failure = s;
                        }
                    }
                }
                g_live_owned_mempools.fetch_sub(1, std::memory_order_relaxed);
                // Wake waiting importers before reporting: a report can run
                // user code, which could import this pool again.
                ipc_pool_registry.release(b->resource, &b->resource);
                if (failed > 0) {
                    char operation[kOperationNameSize];
                    format_operation(operation, sizeof(operation), "cuMemPoolDestroy", handle_bits(b->resource));
                    char detail[kErrorMessageSize];
                    std::snprintf(detail, sizeof(detail), "failed %zu of %zu times, first with", failed, total);
                    report_cuda_error(operation, first_failure, total > 1 ? detail : nullptr);
                }
            }
            delete b;
        }
    );
    // clang-format on
    auto *handle_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(fd));
    if (CUDA_SUCCESS !=
        (err = DRIVER_CALL(cuMemPoolImportFromShareableHandle, &box->resource, handle_ptr, handle_type, 0))) {
        return {};
    }

    MemoryPoolHandle h(box, &box->resource);
    MemoryPoolHandle owner;
    try {
        owner = ipc_pool_registry.get_or_create(box->resource, [&]() noexcept {
            box->imports.store(1, std::memory_order_relaxed);
            return h;
        });
    }
    catch (...) {
        // The box owns no import yet, so its deleter would not drop this one.
        pw_cuMemPoolDestroy(box->resource);
        throw;
    }
    if (owner == h) {
        g_live_owned_mempools.fetch_add(1, std::memory_order_relaxed);
    }
    else {
        // An earlier import's box owns the pool; it now holds this import's
        // driver reference too.
        get_ipc_box(owner)->imports.fetch_add(1, std::memory_order_relaxed);
    }
    return owner;
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

// The box behind a VirtualMemoryResource buffer: the range of mappings it
// owns (VMM_DESIGN.md). Every handle on a VirtualMemoryBuffer comes from
// deviceptr_create_vmm, so vmm_range() downcasts without a tag; there are no
// virtual functions, so other boxes pay nothing. A size-zero buffer has an
// empty range.
struct VmmDevicePtrBox : DevicePtrBox {
    VmmRangeHandle range;
};
}  // namespace

// Recovers the owning DevicePtrBox from the aliased CUdeviceptr pointer.
// This works because DevicePtrHandle is a shared_ptr alias pointing to
// &box->resource, so we can compute the containing struct using offsetof.
// The const_cast is safe because we only use this to access the mutable
// deallocation member or in the deleter (where the box is being destroyed).
static DevicePtrBox *
get_box(const DevicePtrHandle &h)
{
    const CUdeviceptr *p = h.get();
    return reinterpret_cast<DevicePtrBox *>(
        reinterpret_cast<char *>(const_cast<CUdeviceptr *>(p)) - offsetof(DevicePtrBox, resource)
    );
}

// Return the stream that orders a device pointer's deallocation; empty for an
// empty handle, as set_deallocation_stream rejects one.
StreamHandle
deallocation_stream(const DevicePtrHandle &h) noexcept
{
    return h ? get_box(h)->deallocation.h_stream : StreamHandle{};
}

// Replace the stream that orders a device pointer's deallocation.
CUresult
set_deallocation_stream(const DevicePtrHandle &h, const StreamHandle &h_stream)
{
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

DevicePtrHandle
deviceptr_alloc_from_pool(size_t size, const MemoryPoolHandle &h_pool, const StreamHandle &h_stream)
{
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemAllocFromPoolAsync, &ptr, size, *h_pool, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_pool](DevicePtrBox *b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync", handle_bits(b->resource),
                [&]() noexcept {
                    return DRIVER_CALL(cuMemFreeAsync,
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle
deviceptr_alloc_async(size_t size, const StreamHandle &h_stream)
{
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemAllocAsync, &ptr, size, as_cu(h_stream)))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [](DevicePtrBox *b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync", handle_bits(b->resource),
                [&]() noexcept {
                    return DRIVER_CALL(cuMemFreeAsync,
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

// Allocate device memory synchronously with the provided context current.
CUresult
deviceptr_alloc_raw(CUdeviceptr *ptr, size_t size, const ContextHandle &h_context) noexcept
{
    GILReleaseGuard gil;
    return invoke_in_context_or_undo(
        h_context,
        [&]() noexcept { return DRIVER_CALL(cuMemAlloc, ptr, size); },
        [&]() noexcept { pw_cuMemFree(*ptr); },
        /*undo_requires_target_context=*/false
    );
}

DevicePtrHandle
deviceptr_alloc_host(size_t size)
{
    GILReleaseGuard gil;
    void *ptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemAllocHost, &ptr, size))) {
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{reinterpret_cast<CUdeviceptr>(ptr), DeallocationStream{}},
        [](DevicePtrBox *b) {
            GILReleaseGuard gil;
            pw_cuMemFreeHost(reinterpret_cast<void *>(b->resource));
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle
deviceptr_create_ref(CUdeviceptr ptr)
{
    auto box = std::make_shared<DevicePtrBox>(DevicePtrBox{ptr, DeallocationStream{}});
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle
deviceptr_create_with_owner(CUdeviceptr ptr, PyObject *owner)
{
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
    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, DeallocationStream{}},
        [owner](DevicePtrBox *b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                Py_DECREF(owner);
            }
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

DevicePtrHandle
deviceptr_create_mapped_graphics(
    CUdeviceptr ptr, const GraphicsResourceHandle &h_resource, const StreamHandle &h_stream
)
{
    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        return {};
    }
    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_resource](DevicePtrBox *b) {
            GILReleaseGuard gil;
            CUgraphicsResource resource = as_cu(h_resource);
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuGraphicsUnmapResources", handle_bits(resource),
                [&]() noexcept {
                    return DRIVER_CALL(cuGraphicsUnmapResources,
                        1, &resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// Virtual memory ranges (VMM_DESIGN.md)
// ============================================================================

DevicePtrHandle
deviceptr_create_vmm(CUdeviceptr base, const VmmRangeHandle &range)
{
    if (!range) {  // a null handle; an empty range is valid
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }
    // clang-format off
    auto box = std::shared_ptr<VmmDevicePtrBox>(
        new VmmDevicePtrBox{{base, DeallocationStream{}}, range},
        [](VmmDevicePtrBox *b) {
            GILReleaseGuard gil;
            // Order the release on this buffer's recorded stream, then free
            // the box, which drops the range: every mapping this buffer was
            // the last to hold unmaps, and its reservation and allocation
            // follow.
            vmm_sync_before_release(b->deallocation);
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

VmmRangeHandle
vmm_range(const DevicePtrHandle &h) noexcept
{
    // Only for handles from deviceptr_create_vmm; the VirtualMemoryBuffer
    // class guarantees that for its callers.
    return h ? static_cast<VmmDevicePtrBox *>(get_box(h))->range : VmmRangeHandle{};
}

// ============================================================================
// MemoryResource-owned Device Pointer Handles
// ============================================================================

static MRDeallocCallback mr_dealloc_cb = nullptr;

void
register_mr_dealloc_callback(MRDeallocCallback cb)
{
    mr_dealloc_cb = cb;
}

DevicePtrHandle
deviceptr_create_with_mr(CUdeviceptr ptr, size_t size, PyObject *mr)
{
    if (!mr) {
        return deviceptr_create_ref(ptr);
    }
    // GIL required when mr is provided
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return deviceptr_create_ref(ptr);
    }
    Py_INCREF(mr);
    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, DeallocationStream{}},
        [mr, size](DevicePtrBox *b) {
            GILAcquireGuard gil;
            if (gil.acquired()) {
                // The last reference may go while an exception propagates
                // through the releasing caller; deallocate() must run with a
                // clean error state and leave that exception in place.
                PendingExceptionGuard pending;
                if (mr_dealloc_cb) {
                    const DeallocationStream& stream = b->deallocation;
                    cleanup_in_context(
                        deallocation_context(stream), "MemoryResource.deallocate", handle_bits(b->resource),
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
    // clang-format on
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
static bool
use_ipc_ptr_cache()
{
    return true;
}

namespace {
// Wrapper for CUmemPoolPtrExportData to use as map key
struct ExportDataKey {
    CUmemPoolPtrExportData data;

    bool
    operator==(const ExportDataKey &other) const
    {
        return std::memcmp(&data, &other.data, sizeof(data)) == 0;
    }
};

struct ExportDataKeyHash {
    std::size_t
    operator()(const ExportDataKey &key) const
    {
        // Simple hash of the bytes
        std::size_t h = 0;
        const auto *bytes = reinterpret_cast<const unsigned char *>(&key.data);
        for (std::size_t i = 0; i < sizeof(key.data); ++i) {
            h = (h * 31) + bytes[i];  // NOLINT(readability-magic-numbers)
        }
        return h;
    }
};

}  // namespace

// An importer that finds an expired entry waits until the previous handle's
// deleter has freed the mapping: a re-import before then would share that
// mapping, and the deleter's cuMemFreeAsync would unmap it under the new
// handle (nvbug 5570902).
static ReleasingHandleRegistry<ExportDataKey, DevicePtrHandle, ExportDataKeyHash> ipc_ptr_cache;

DevicePtrHandle
deviceptr_import_ipc(const MemoryPoolHandle &h_pool, const void *export_data, const StreamHandle &h_stream)
{
    auto *data = const_cast<CUmemPoolPtrExportData *>(reinterpret_cast<const CUmemPoolPtrExportData *>(export_data));

    // Resolve the table before you take any lock: a fill acquires the GIL, and
    // nothing under the cache's mutex may acquire it (#2840). The raw p_ calls
    // below rely on this.
    if (!ensure_fn_table(FnTable::driver)) {
        report_unavailable_fn(FnTable::driver, "cuMemPoolImportPointer");
        err = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }

    if (use_ipc_ptr_cache()) {
        ExportDataKey key;
        std::memcpy(&key.data, data, sizeof(key.data));

        // Release the GIL before the cache's mutex is taken: a thread blocked
        // on it while holding the GIL would deadlock with a holder that needs
        // the GIL back (#2840).
        GILReleaseGuard gil;
        // Allocate before taking the mutex: a failed allocation then cannot run
        // a deleter under it. The box owns nothing until it holds an import.
        // clang-format off
        auto box = std::shared_ptr<DevicePtrBox>(
            new DevicePtrBox{0, DeallocationStream{}},
            [h_pool, key](DevicePtrBox *b) {
                if (b->resource != 0) {
                    GILReleaseGuard gil;
                    const DeallocationStream& stream = b->deallocation;
                    // Release the entry once the mapping is freed, and before
                    // reporting: a report can run user code.
                    cleanup_in_context(
                        deallocation_context(stream), "cuMemFreeAsync", handle_bits(b->resource),
                        [&]() noexcept {
                            return DRIVER_CALL(cuMemFreeAsync, b->resource, as_cu(stream.h_stream));
                        },
                        [&]() noexcept { ipc_ptr_cache.release(key, &b->resource); });
                }
                delete b;
            }
        );
        // clang-format on
        DevicePtrHandle h(box, &box->resource);

        // The factory runs under the cache's mutex, which makes lookup, import
        // and registration one step: two threads cannot import the same
        // descriptor twice. Nothing under the mutex may acquire the GIL, so a
        // failed discard is reported only after the lock is released (see
        // DESIGN.md).
        CUresult discard_status = CUDA_SUCCESS;
        CUdeviceptr discarded = 0;
        auto report_failed_discard = [&]() noexcept {
            if (discard_status != CUDA_SUCCESS) {
                char operation[kOperationNameSize];
                format_operation(operation, sizeof(operation), "cuMemFreeAsync", discarded);
                report_cuda_error(
                    operation,
                    discard_status,
                    "failed while discarding an IPC import that could not record a "
                    "deallocation stream; the mapping leaked"
                );
            }
        };
        DevicePtrHandle imported;
        try {
            imported = ipc_ptr_cache.get_or_create(key, [&]() -> DevicePtrHandle {
                CUdeviceptr ptr;
                if (CUDA_SUCCESS !=
                    (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {  // raw: under the cache's mutex
                    return {};
                }
                // No deallocation stream could be recorded: discard the import
                // with the raw call (a pw_ report would acquire the GIL) before
                // the mutex is released, or a concurrent import would share the
                // mapping this free unmaps.
                bool recorded;
                try {
                    recorded = make_deallocation_stream(h_stream, box->deallocation);
                }
                catch (...) {
                    discard_status = p_cuMemFreeAsync(ptr, as_cu(h_stream));  // raw: under the cache's mutex
                    discarded = ptr;
                    throw;
                }
                if (!recorded) {
                    discard_status = p_cuMemFreeAsync(ptr, as_cu(h_stream));  // raw: under the cache's mutex
                    discarded = ptr;
                    return {};
                }
                box->resource = ptr;
                return h;
            });
        }
        catch (...) {
            report_failed_discard();
            throw;
        }
        if (imported) {
            return imported;
        }
        report_failed_discard();
        return {};
    }
    GILReleaseGuard gil;
    CUdeviceptr ptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuMemPoolImportPointer, &ptr, *h_pool, data))) {
        return {};
    }

    DeallocationStream ds;
    if (!make_deallocation_stream(h_stream, ds)) {
        pw_cuMemFreeAsync(ptr, as_cu(h_stream));
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<DevicePtrBox>(
        new DevicePtrBox{ptr, std::move(ds)},
        [h_pool](DevicePtrBox *b) {
            GILReleaseGuard gil;
            const DeallocationStream& stream = b->deallocation;
            cleanup_in_context(
                deallocation_context(stream), "cuMemFreeAsync", handle_bits(b->resource),
                [&]() noexcept {
                    return DRIVER_CALL(cuMemFreeAsync,
                        b->resource, as_cu(stream.h_stream));
                });
            delete b;
        }
    );
    // clang-format on
    return DevicePtrHandle(box, &box->resource);
}

// ============================================================================
// File Descriptor Handles
// ============================================================================

FileDescriptorHandle
create_fd_handle([[maybe_unused]] int fd)
{
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle is not supported on Windows");
#else
    // clang-format off
    return FileDescriptorHandle(
        new int(fd),
        [](const int *p) {
            if (::close(*p) != 0) {
                report_message("close() failed for an IPC file descriptor; the descriptor may have leaked");
            }
            delete p;
        }
    );
    // clang-format on
#endif
}

FileDescriptorHandle
create_fd_handle_ref([[maybe_unused]] int fd)
{
#ifdef _WIN32
    throw std::runtime_error("create_fd_handle_ref is not supported on Windows");
#else
    return std::make_shared<const int>(fd);
#endif
}

}  // namespace cuda_core::rt
