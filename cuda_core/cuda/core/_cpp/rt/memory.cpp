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
#include <stdexcept>
#include <utility>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace cuda_core::rt {

using namespace detail;

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

namespace {
// Deleter of the box behind a cached IPC import. The box is allocated before
// the import, so it may never have held one: a box whose import failed, or
// that found an earlier import in the cache, has nothing to free.
struct DeleteIpcPtrBox {
    MemoryPoolHandle h_pool;  // keeps the pool alive until the mapping is freed
    ExportDataKey key;

    static bool
    holds_import(const DevicePtrBox &box) noexcept
    {
        return box.resource != 0;
    }

    void
    operator()(DevicePtrBox *box) const
    {
        if (holds_import(*box)) {
            free_import(*box);
        }
        delete box;
    }

   private:
    void
    free_import(const DevicePtrBox &box) const
    {
        GILReleaseGuard gil;
        const DeallocationStream &stream = box.deallocation;
        // Release the entry once the mapping is freed, and before reporting:
        // a report can run user code.
        cleanup_in_context(
            deallocation_context(stream),
            "cuMemFreeAsync",
            handle_bits(box.resource),
            [&]() noexcept { return DRIVER_CALL(cuMemFreeAsync, box.resource, as_cu(stream.h_stream)); },
            [&]() noexcept { ipc_ptr_cache.release(key, &box.resource); }
        );
    }
};

// An import that was discarded under the cache's mutex because no
// deallocation stream could be recorded for it. Nothing under that mutex may
// acquire the GIL, so the discard uses the raw p_ call and keeps its status;
// report() runs after the mutex is released (see DESIGN.md).
struct DiscardedIpcImport {
    CUdeviceptr ptr = 0;
    CUresult status = CUDA_SUCCESS;

    static DiscardedIpcImport
    discard(CUdeviceptr ptr, const StreamHandle &h_stream) noexcept
    {
        return {ptr, p_cuMemFreeAsync(ptr, as_cu(h_stream))};  // raw: under the cache's mutex
    }

    void
    report() const noexcept
    {
        if (status == CUDA_SUCCESS) {
            return;
        }
        char operation[kOperationNameSize];
        format_operation(operation, sizeof(operation), "cuMemFreeAsync", ptr);
        report_cuda_error(
            operation,
            status,
            "failed while discarding an IPC import that could not record a "
            "deallocation stream; the mapping leaked"
        );
    }
};
}  // namespace

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
        auto box =
            std::shared_ptr<DevicePtrBox>(new DevicePtrBox{0, DeallocationStream{}}, DeleteIpcPtrBox{h_pool, key});
        DevicePtrHandle h(box, &box->resource);

        // Resolve the deallocation stream before the mutex is taken: for a
        // default-stream token it looks up the current context, which may pin a
        // primary context and report a failure (get_current_context).
        // Only a fresh import uses it, so a failure is kept for that path, and
        // a cached import is returned as before. May throw std::bad_alloc;
        // nothing has been imported yet.
        DeallocationStream ds;
        const CUresult status_before = err;
        const bool ds_ok = make_deallocation_stream(h_stream, ds);
        const CUresult ds_status = err;
        err = status_before;

        // The factory runs under the cache's mutex, which makes lookup, import
        // and registration one step: two threads cannot import the same
        // descriptor twice. It therefore calls the driver through the raw p_
        // pointers only, and hands a failed discard out to be reported here.
        DiscardedIpcImport discarded;
        DevicePtrHandle imported;
        try {
            imported = ipc_ptr_cache.get_or_create(key, [&]() -> DevicePtrHandle {
                CUdeviceptr ptr;
                if (CUDA_SUCCESS !=
                    (err = p_cuMemPoolImportPointer(&ptr, *h_pool, data))) {  // raw: under the cache's mutex
                    return {};
                }
                // No deallocation stream could be recorded: discard the import
                // before the mutex is released, or a concurrent import would
                // share the mapping this free unmaps.
                if (!ds_ok) {
                    err = ds_status;
                    discarded = DiscardedIpcImport::discard(ptr, h_stream);
                    return {};
                }
                // Moves two handles; no deleter runs under the mutex.
                box->deallocation = std::move(ds);
                box->resource = ptr;
                return h;
            });
        }
        catch (...) {
            discarded.report();
            throw;
        }
        if (!imported) {
            discarded.report();
        }
        return imported;
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
