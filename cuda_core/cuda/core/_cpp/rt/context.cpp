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
#include <mutex>
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
    bool retains_primary = false;  // Pinned primary context (see pin_primary_context).
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

// A box keeps its context alive if it is a pinned primary context or holds the
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

// ----------------------------------------------------------------------------
// Primary contexts
//
// cuda.core retains each device's primary context at most once per process and
// never gives that retain back, as the CUDA runtime does. The driver destroys
// a primary context when its last retain is released, so a context that
// cuda.core has handed out can only be destroyed by code outside cuda.core
// (cuDevicePrimaryCtxReset, cudaDeviceReset).
//
// The retain is held by one box per device, pinned in a process-wide table
// that is never freed: no thread's exit and no handle's release can drop the
// box's last reference, so its deleter never runs.
// ----------------------------------------------------------------------------

std::mutex primary_context_mutex;

// Pinned primary contexts indexed by device ordinal; guarded by
// primary_context_mutex. Leaked on purpose: other threads may still use it
// while the process exits.
std::vector<ContextHandle> &
primary_context_table()
{
    static auto *table = new std::vector<ContextHandle>();  // NOLINT(cppcoreguidelines-owning-memory)
    return *table;
}

// Return the pinned primary context of device_id, pinning it first if needed.
//
// With expected == nullptr, retain the device's primary context (creating it
// if it is not active) and pin it. On failure, return an empty handle and
// store the status in *status.
//
// With expected != nullptr, pin and return the primary context only if it is
// active and is `expected`; otherwise return an empty handle. A primary
// context that is not active is not retained, because that would create it. If
// the active primary context is another one, the retain taken to learn its
// address is given back: the context was active without it, so that release
// cannot destroy a context that its other owners still hold.
ContextHandle
pin_primary_context(int device_id, CUcontext expected, CUresult *status)
{
    *status = CUDA_SUCCESS;
    // The GIL is the outermost lock, and filling the table acquires it: do both
    // before the mutex is taken. The raw p_ calls below rely on the fill.
    GILReleaseGuard gil;
    if (!ensure_fn_table(FnTable::driver)) {
        report_unavailable_fn(FnTable::driver, "cuDevicePrimaryCtxRetain");
        *status = CUDA_ERROR_NOT_INITIALIZED;
        return {};
    }

    const auto idx = static_cast<size_t>(device_id);
    ContextHandle h;
    bool release_probe = false;
    {
        std::scoped_lock lock(primary_context_mutex);
        auto &table = primary_context_table();
        if (idx < table.size() && table[idx]) {
            if (expected && as_cu(table[idx]) != expected) {
                return {};
            }
            return table[idx];
        }
        if (expected) {
            unsigned int flags = 0;
            int active = 0;
            *status = p_cuDevicePrimaryCtxGetState(device_id, &flags, &active);  // raw: under primary_context_mutex
            if (*status != CUDA_SUCCESS || !active) {
                return {};
            }
        }
        CUcontext ctx = nullptr;
        *status = p_cuDevicePrimaryCtxRetain(&ctx, device_id);  // raw: under primary_context_mutex
        if (*status != CUDA_SUCCESS) {
            return {};
        }
        if (expected && ctx != expected) {
            release_probe = true;
        }
        else {
            // clang-format off
            auto box = std::shared_ptr<const ContextBox>(
                new ContextBox{ctx, {}, true},
                [](const ContextBox *b) {
                    // Unreachable while the table pins the box. The retain is
                    // deliberately not released.
                    context_registry.unregister_handle(b->resource, &b->resource);
                    delete b;
                }
            );
            // clang-format on
            h = ContextHandle(box, &box->resource);
            if (idx >= table.size()) {
                table.resize(idx + 1);
            }
            table[idx] = h;
        }
    }
    if (release_probe) {
        // Reports through Python on failure, so it runs after the unlock.
        pw_cuDevicePrimaryCtxRelease(device_id);
        return {};
    }
    // The driver has just returned this address as the device's primary
    // context, so a live entry under it is a plain box or a stale one: replace
    // it. Registering after the unlock keeps a displaced box's deleter out of
    // primary_context_mutex. A lookup that misses the registry in between finds
    // the box in the table (see adopt_current_primary_context).
    context_registry.insert_or_get(as_cu(h), h, [](const ContextHandle &) noexcept { return true; });
    return h;
}

// If ctx is the calling thread's current context and the primary context of
// its device, pin it and return the pinned handle; otherwise return an empty
// handle. Only the current context is examined, because its device can be
// queried without switching contexts, which a lookup must not do.
ContextHandle
adopt_current_primary_context(CUcontext ctx)
{
    // The lookup falls back to a handle that holds no retain, and its callers do
    // not check for an error, so each failure is reported.
    constexpr const char *detail = "failed while looking up the current context";
    GILReleaseGuard gil;
    CUcontext current = nullptr;
    CUresult status = DRIVER_CALL(cuCtxGetCurrent, &current);
    if (status != CUDA_SUCCESS) {
        report_cuda_error("cuCtxGetCurrent", status, detail);
        return {};
    }
    if (current != ctx) {
        return {};
    }
    CUdevice device = 0;
    status = DRIVER_CALL(cuCtxGetDevice, &device);
    if (status != CUDA_SUCCESS) {
        report_cuda_error("cuCtxGetDevice", status, detail);
        return {};
    }
    ContextHandle h = pin_primary_context(device, ctx, &status);
    if (status != CUDA_SUCCESS) {
        report_cuda_error("cuDevicePrimaryCtxRetain", status, detail);
    }
    return h;
}

// Create a context handle reference, with optional green context as source.
ContextHandle
create_context_handle_ref(CUcontext ctx, GreenCtxHandle h_green_ctx)
{
    if (!ctx) {
        return {};
    }
    ContextHandle registered = context_registry.lookup(ctx);
    // An owning entry for ctx is either this green context's own box or a
    // pinned primary context. A green context cannot share a live primary
    // context's CUcontext, so returning it never drops a needed green owner.
    if (registered && owns_context(registered)) {
        return registered;
    }
    if (!h_green_ctx) {
        // A primary context that cuda.core has not retained yet, for example one
        // made current by another library: pin it, so that the handle keeps it
        // alive like every other primary-context handle.
        if (ContextHandle h = adopt_current_primary_context(ctx)) {
            return h;
        }
        if (registered) {
            return registered;
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

// Thread-local cache of pinned primary contexts indexed by device ordinal: a
// fast path in front of primary_context_mutex. Its entries share the pinned
// boxes, so destroying it at thread exit only drops references.
static thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle
get_primary_context(int device_id)
{
    const auto idx = static_cast<size_t>(device_id);
    if (idx < primary_context_cache.size()) {
        if (auto cached = primary_context_cache[idx]) {
            return cached;
        }
    }

    ContextHandle h = pin_primary_context(device_id, nullptr, &err);
    if (!h) {
        return {};
    }
    if (idx >= primary_context_cache.size()) {
        primary_context_cache.resize(idx + 1);
    }
    primary_context_cache[idx] = h;
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
