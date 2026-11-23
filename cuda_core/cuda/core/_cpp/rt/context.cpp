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
#include <memory>
#include <utility>
#include <vector>

namespace cuda_core::rt {

using namespace detail;

namespace {
// Thread-local fault injected into the next context restoration (tests only).
thread_local CUresult context_restore_fault = CUDA_SUCCESS;
}  // namespace

void
set_context_restore_fault_for_testing(CUresult status) noexcept
{
    context_restore_fault = status;
}

namespace detail {
// Make a context current and record the state needed to restore it.
// An empty handle is a no-op: the operation runs in the caller's current
// context, and nothing is restored on exit. invoke_in_context and
// invoke_in_context_or_undo reject empty handles before getting here; only
// graph_node_set_params relies on the no-op (pre-13.2 node updates run in the
// caller's context).
CUresult
enter_context(const ContextHandle &h_context, CUcontext *previous, int *changed) noexcept
{
    *previous = nullptr;
    *changed = 0;
    clear_last_error_detail();
    CUcontext target = as_cu(h_context);
    if (!target) {
        return CUDA_SUCCESS;
    }

    GILReleaseGuard gil;
    CUresult status = DRIVER_CALL(cuCtxGetCurrent, previous);
    if (status != CUDA_SUCCESS || *previous == target) {
        return status;
    }
    status = DRIVER_CALL(cuCtxSetCurrent, target);
    *changed = status == CUDA_SUCCESS;
    return status;
}

// Restore the caller's context. Returns the restoration status.
CUresult
restore_context(CUcontext previous) noexcept
{
    if (context_restore_fault != CUDA_SUCCESS) {
        // Test hook: behave as if cuCtxSetCurrent(previous) failed, leaving the
        // target context current exactly as a real failure would.
        CUresult fault = context_restore_fault;
        context_restore_fault = CUDA_SUCCESS;
        return fault;
    }
    GILReleaseGuard gil;
    return DRIVER_CALL(cuCtxSetCurrent, previous);
}
// Restore the previous context and preserve an earlier operation error. The
// operation error, if any, is returned; otherwise the restoration status is.
// Either way a restoration failure is recorded as the detail of the returned
// status, so the eventual CUDAError explains it (see take_last_error_detail()).
CUresult
exit_context(CUcontext previous, int changed, CUresult operation_status) noexcept
{
    CUresult restore_status = changed ? restore_context(previous) : CUDA_SUCCESS;
    if (restore_status == CUDA_SUCCESS) {
        return operation_status;
    }
    note_context_not_restored(previous, operation_status, restore_status);
    return operation_status != CUDA_SUCCESS ? operation_status : restore_status;
}
}  // namespace detail

// Synchronize the provided context.
CUresult
context_synchronize(const ContextHandle &h_context) noexcept
{
    GILReleaseGuard gil;
    return invoke_in_context(h_context, []() noexcept { return DRIVER_FN(cuCtxSynchronize)(); });
}

// Query the stream priority range for the provided context.
CUresult
context_get_stream_priority_range(const ContextHandle &h_context, int *least_priority, int *greatest_priority) noexcept
{
    GILReleaseGuard gil;
    return invoke_in_context(h_context, [&]() noexcept {
        return DRIVER_CALL(cuCtxGetStreamPriorityRange, least_priority, greatest_priority);
    });
}

// Query the device of the provided context.
CUresult
context_get_device(const ContextHandle &h_context, CUdevice *device) noexcept
{
    return invoke_in_context(h_context, [&]() noexcept { return DRIVER_CALL(cuCtxGetDevice, device); });
}

// ============================================================================
// Context Handles
// ============================================================================

namespace {
struct ContextBox {
    CUcontext resource;
    GreenCtxHandle h_green_ctx;
    bool retains_primary = false;  // Owns one cuDevicePrimaryCtxRetain.
};

struct GreenCtxBox {
    CUgreenCtx resource;
};

const ContextBox *
get_box(const ContextHandle &h) noexcept
{
    const CUcontext *p = h.get();
    return reinterpret_cast<const ContextBox *>(reinterpret_cast<const char *>(p) - offsetof(ContextBox, resource));
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
HandleRegistry<CUcontext, ContextHandle> context_registry;

// A box keeps its context alive if it holds a primary-context retain or the
// owning green-context handle. Reads only immutable box fields, so it is safe
// under the registry's mutex.
bool
owns_context(const ContextHandle &h) noexcept
{
    const ContextBox *b = get_box(h);
    return b->retains_primary || b->h_green_ctx;
}

// Registry replacement policy for contexts: an owning candidate replaces a
// non-owning entry, and nothing replaces an owning entry. This keeps one
// owning box per live context, and a lookup never trades it for a plain one.
auto
replace_if_upgrade(const ContextHandle &candidate) noexcept
{
    return [owning = owns_context(candidate)](const ContextHandle &existing) noexcept {
        return owning && !owns_context(existing);
    };
}

// Create a context handle reference, with optional green context as source.
ContextHandle
create_context_handle_ref(CUcontext ctx, GreenCtxHandle h_green_ctx)
{
    if (!ctx) {
        return {};
    }
    if (auto h = context_registry.lookup(ctx)) {
        // An owning entry for ctx is either this green context's own box or a
        // live primary context. A green context cannot share a live primary
        // context's CUcontext, so returning it never drops a needed green owner.
        if (!h_green_ctx || owns_context(h)) {
            return h;
        }
    }
    // clang-format off
    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx, std::move(h_green_ctx)},
        [](const ContextBox *b) {
            context_registry.unregister_handle(b->resource, &b->resource);
            delete b;
        }
    );
    // clang-format on
    ContextHandle candidate(box, &box->resource);
    // A losing candidate is dropped by the caller, outside the registry's lock;
    // its deleter's ownership-checked erase is then a no-op.
    return context_registry.insert_or_get(ctx, candidate, replace_if_upgrade(candidate));
}
}  // namespace

ContextHandle
create_context_handle_ref(CUcontext ctx)
{
    return create_context_handle_ref(ctx, {});
}

ContextHandle
create_context_handle_from_green_ctx(const GreenCtxHandle &h_green_ctx)
{
    GILReleaseGuard gil;
    if (!h_green_ctx) {
        return {};
    }
    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuCtxFromGreenCtx, &ctx, as_cu(h_green_ctx)))) {
        return {};
    }

    return create_context_handle_ref(ctx, h_green_ctx);
}

GreenCtxHandle
get_context_green_ctx(const ContextHandle &h) noexcept
{
    if (!h) {
        return {};
    }
    return get_box(h)->h_green_ctx;
}

GreenCtxHandle
create_green_ctx_handle(CUdevResource *resources, unsigned int nbResources, CUdevice dev, unsigned int flags)
{
    GILReleaseGuard gil;
    CUdevResourceDesc desc = nullptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuDevResourceGenerateDesc, &desc, resources, nbResources))) {
        return {};
    }

    CUgreenCtx green_ctx = nullptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuGreenCtxCreate, &green_ctx, desc, dev, flags))) {
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<const GreenCtxBox>(
        new GreenCtxBox{green_ctx},
        [](const GreenCtxBox *b) {
            GILReleaseGuard gil;
            pw_cuGreenCtxDestroy(b->resource);
            delete b;
        }
    );
    // clang-format on
    return GreenCtxHandle(box, &box->resource);
}

GreenCtxHandle
create_green_ctx_handle_ref(CUgreenCtx green_ctx)
{
    if (!green_ctx) {
        return {};
    }
    auto box = std::make_shared<const GreenCtxBox>(GreenCtxBox{green_ctx});
    return GreenCtxHandle(box, &box->resource);
}

namespace {
// Set when the calling thread's primary-context cache is torn down. A plain
// bool needs no destructor, so it stays readable for the whole thread exit.
thread_local bool primary_context_cache_closed = false;

// Thread-local cache of primary contexts indexed by device ID. A fast path in
// front of the registry: every thread caches the same process-wide box.
struct PrimaryContextCache {
    std::vector<ContextHandle> entries;

    PrimaryContextCache() = default;
    PrimaryContextCache(const PrimaryContextCache &) = delete;
    PrimaryContextCache &operator=(const PrimaryContextCache &) = delete;

    // Releasing the last entry can run a CUDAWarning hook on this thread, and
    // the hook may call get_primary_context(). Close the cache first and drop
    // the entries from a local, so such a call bypasses the cache instead of
    // reading one that is being destroyed.
    ~PrimaryContextCache()
    {
        primary_context_cache_closed = true;
        std::vector<ContextHandle> doomed;
        doomed.swap(entries);
    }
};

thread_local PrimaryContextCache primary_context_cache;
}  // namespace

ContextHandle
get_primary_context(int device_id)
{
    const bool use_cache = !primary_context_cache_closed;
    auto idx = static_cast<size_t>(device_id);

    // Check thread-local cache
    if (use_cache && idx < primary_context_cache.entries.size()) {
        if (auto cached = primary_context_cache.entries[idx]) {
            return cached;
        }
    }

    // Cache miss - acquire primary context from driver
    GILReleaseGuard gil;
    CUcontext ctx;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuDevicePrimaryCtxRetain, &ctx, device_id))) {
        return {};
    }

    // clang-format off
    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx, {}, true},
        [device_id](const ContextBox *b) {
            context_registry.unregister_handle(b->resource, &b->resource);
            // During interpreter shutdown, leave primary-context cleanup to
            // process teardown. Reporting a failure needs Python too.
            if (Py_IsInitialized() && !py_is_finalizing()) {
                GILReleaseGuard gil;
                pw_cuDevicePrimaryCtxRelease(device_id);
            }
            delete b;
        }
    );
    // clang-format on
    ContextHandle candidate(box, &box->resource);
    // Keep one retaining box per primary context per process. If another
    // thread's box is live, use it, and dropping our candidate releases the
    // extra retain taken above; the live box still holds its own retain, so the
    // driver's count never reaches zero. This runs with the GIL released and
    // after insert_or_get has released the registry's lock, so the deleter may
    // report through a pw_ wrapper.
    ContextHandle h = context_registry.insert_or_get(ctx, candidate, replace_if_upgrade(candidate));
    candidate.reset();
    box.reset();

    // Update cache
    if (use_cache) {
        if (idx >= primary_context_cache.entries.size()) {
            primary_context_cache.entries.resize(idx + 1);
        }
        primary_context_cache.entries[idx] = h;
    }
    return h;
}

ContextHandle
get_current_context()
{
    GILReleaseGuard gil;
    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = DRIVER_CALL(cuCtxGetCurrent, &ctx))) {
        return {};
    }
    if (!ctx) {
        return {};  // No current context (not an error)
    }
    return create_context_handle_ref(ctx);
}

}  // namespace cuda_core::rt
