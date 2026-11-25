// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

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

// See set_ipc_pool_release_delay_for_testing.
std::atomic<unsigned int> g_ipc_pool_release_delay_us{0};
}  // namespace

void
set_ipc_pool_release_delay_for_testing(unsigned int microseconds) noexcept
{
    g_ipc_pool_release_delay_us.store(microseconds, std::memory_order_relaxed);
}

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

    // False for a box that found an earlier import's box registered: that
    // owner took over its import, and this box has nothing to release.
    // Shared ownership synchronizes the increments with this load.
    bool
    owns_pool() const noexcept
    {
        return imports.load(std::memory_order_relaxed) > 0;
    }
};

const IpcMemoryPoolBox *
get_ipc_box(const MemoryPoolHandle &h)
{
    const CUmemoryPool *p = h.get();
    return reinterpret_cast<const IpcMemoryPoolBox *>(
        reinterpret_cast<const char *>(p) - offsetof(IpcMemoryPoolBox, resource)
    );
}

// The outcome of dropping every driver reference an owner holds on its pool.
struct DestroyedImports {
    std::size_t total = 0;
    std::size_t failed = 0;
    CUresult first_failure = CUDA_SUCCESS;
};

// One cuMemPoolDestroy for each import. Does not report: the caller still has
// to wake waiting importers first.
DestroyedImports
destroy_imports(CUmemoryPool pool, std::size_t imports) noexcept
{
    DestroyedImports result;
    result.total = imports;
    for (; imports > 0; --imports) {
        if (CUresult s = DRIVER_CALL(cuMemPoolDestroy, pool); s != CUDA_SUCCESS) {
            if (result.failed++ == 0) {
                result.first_failure = s;
            }
        }
    }
    return result;
}

void
report_failed_destroys(CUmemoryPool pool, const DestroyedImports &result) noexcept
{
    if (result.failed == 0) {
        return;
    }
    char operation[kOperationNameSize];
    format_operation(operation, sizeof(operation), "cuMemPoolDestroy", handle_bits(pool));
    char detail[kErrorMessageSize];
    std::snprintf(detail, sizeof(detail), "failed %zu of %zu times, first with", result.failed, result.total);
    report_cuda_error(operation, result.first_failure, result.total > 1 ? detail : nullptr);
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
ReleasingHandleRegistry<CUmemoryPool, MemoryPoolHandle> ipc_pool_registry;

// The last reference to the pool's owner is gone. The order matters: revoke
// access once, destroy every import, let waiting importers proceed, and only
// then report.
void
release_owned_ipc_pool(const IpcMemoryPoolBox &box)
{
    GILReleaseGuard gil;
    if (unsigned int delay = g_ipc_pool_release_delay_us.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
    }
    clear_mempool_peer_access(box.resource, -1);
    const DestroyedImports destroyed = destroy_imports(box.resource, box.imports.load(std::memory_order_relaxed));
    g_live_owned_mempools.fetch_sub(1, std::memory_order_relaxed);
    // Wake waiting importers before reporting: a report can run user code,
    // which could import this pool again.
    ipc_pool_registry.release(box.resource, &box.resource);
    report_failed_destroys(box.resource, destroyed);
}

// Deleter of every IpcMemoryPoolBox.
void
delete_ipc_pool_box(const IpcMemoryPoolBox *box)
{
    if (box->owns_pool()) {
        release_owned_ipc_pool(*box);
    }
    delete box;
}
}  // namespace

MemoryPoolHandle
create_mempool_handle_ipc(int fd, CUmemAllocationHandleType handle_type)
{
    // Released before the registry can wait: the owner's deleter may need it.
    GILReleaseGuard gil;
    // Allocate before importing: a failed allocation then has nothing to undo.
    auto box = std::shared_ptr<IpcMemoryPoolBox>(new IpcMemoryPoolBox{}, delete_ipc_pool_box);
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
    const bool became_owner = owner == h;
    if (became_owner) {
        g_live_owned_mempools.fetch_add(1, std::memory_order_relaxed);
    }
    else {
        // An earlier import's box owns the pool; it now holds this import's
        // driver reference too.
        get_ipc_box(owner)->imports.fetch_add(1, std::memory_order_relaxed);
    }
    return owner;
}

}  // namespace cuda_core::rt
