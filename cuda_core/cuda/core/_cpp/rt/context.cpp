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

void set_context_restore_fault_for_testing(CUresult status) noexcept {
    context_restore_fault = status;
}

namespace detail {
// Make a context current and record the state needed to restore it.
// An empty handle is a no-op: the operation runs in the caller's current
// context, and nothing is restored on exit. invoke_in_context and
// invoke_in_context_or_undo reject empty handles before getting here; only
// graph_node_set_params relies on the no-op (pre-13.2 node updates run in the
// caller's context).
CUresult enter_context(const ContextHandle& h_context, CUcontext* previous, int* changed) noexcept {
    *previous = nullptr;
    *changed = 0;
    clear_last_error_detail();
    CUcontext target = as_cu(h_context);
    if (!target) {
        return CUDA_SUCCESS;
    }

    GILReleaseGuard gil;
    CUresult status = p_cuCtxGetCurrent(previous);
    if (status != CUDA_SUCCESS || *previous == target) {
        return status;
    }
    status = p_cuCtxSetCurrent(target);
    *changed = status == CUDA_SUCCESS;
    return status;
}

// Restore the caller's context. Returns the restoration status.
CUresult restore_context(CUcontext previous) noexcept {
    if (context_restore_fault != CUDA_SUCCESS) {
        // Test hook: behave as if cuCtxSetCurrent(previous) failed, leaving the
        // target context current exactly as a real failure would.
        CUresult fault = context_restore_fault;
        context_restore_fault = CUDA_SUCCESS;
        return fault;
    }
    GILReleaseGuard gil;
    return p_cuCtxSetCurrent(previous);
}
// Restore the previous context and preserve an earlier operation error. The
// operation error, if any, is returned; otherwise the restoration status is.
// Either way a restoration failure is recorded as the detail of the returned
// status, so the eventual CUDAError explains it (see take_last_error_detail()).
CUresult exit_context(CUcontext previous, int changed, CUresult operation_status) noexcept {
    CUresult restore_status = changed ? restore_context(previous) : CUDA_SUCCESS;
    if (restore_status == CUDA_SUCCESS) {
        return operation_status;
    }
    note_context_not_restored(previous, operation_status, restore_status);
    return operation_status != CUDA_SUCCESS ? operation_status : restore_status;
}
}  // namespace detail

// Synchronize the provided context.
CUresult context_synchronize(const ContextHandle& h_context) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context(h_context, []() noexcept {
        return p_cuCtxSynchronize();
    });
}

// Query the stream priority range for the provided context.
CUresult context_get_stream_priority_range(const ContextHandle& h_context,
                                           int* least_priority,
                                           int* greatest_priority) noexcept {
    GILReleaseGuard gil;
    return invoke_in_context(h_context, [&]() noexcept {
        return p_cuCtxGetStreamPriorityRange(least_priority, greatest_priority);
    });
}

// Query the device of the provided context.
CUresult context_get_device(const ContextHandle& h_context, CUdevice* device) noexcept {
    return invoke_in_context(h_context, [&]() noexcept {
        return p_cuCtxGetDevice(device);
    });
}

// ============================================================================
// Context Handles
// ============================================================================

namespace {
struct ContextBox {
    CUcontext resource;
    GreenCtxHandle h_green_ctx;
};

struct GreenCtxBox {
    CUgreenCtx resource;
};

static const ContextBox* get_box(const ContextHandle& h) noexcept {
    const CUcontext* p = h.get();
    return reinterpret_cast<const ContextBox*>(
        reinterpret_cast<const char*>(p) - offsetof(ContextBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUcontext, ContextHandle> context_registry;

// Create a context handle reference, with optional green context as source.
ContextHandle create_context_handle_ref(CUcontext ctx, GreenCtxHandle h_green_ctx) {
    if (!ctx) {
        return {};
    }
    if (auto h = context_registry.lookup(ctx)) {
        return h;
    }
    auto box = std::shared_ptr<const ContextBox>(
        new ContextBox{ctx, std::move(h_green_ctx)},
        [](const ContextBox* b) {
            context_registry.unregister_handle(b->resource);
            delete b;
        }
    );
    ContextHandle h(box, &box->resource);
    context_registry.register_handle(ctx, h);
    return h;
}
}  // namespace

ContextHandle create_context_handle_ref(CUcontext ctx) {
    return create_context_handle_ref(ctx, {});
}

ContextHandle create_context_handle_from_green_ctx(const GreenCtxHandle& h_green_ctx) {
    GILReleaseGuard gil;
    if (!h_green_ctx) {
        return {};
    }
    if (!p_cuCtxFromGreenCtx) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    CUcontext ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuCtxFromGreenCtx(&ctx, as_cu(h_green_ctx)))) {
        return {};
    }

    return create_context_handle_ref(ctx, h_green_ctx);
}

GreenCtxHandle get_context_green_ctx(const ContextHandle& h) noexcept {
    if (!h) {
        return {};
    }
    return get_box(h)->h_green_ctx;
}

GreenCtxHandle create_green_ctx_handle(CUdevResource* resources, unsigned int nbResources,
                                       CUdevice dev, unsigned int flags) {
    GILReleaseGuard gil;
    if (!p_cuDevResourceGenerateDesc || !p_cuGreenCtxCreate || !p_cuGreenCtxDestroy) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    CUdevResourceDesc desc = nullptr;
    if (CUDA_SUCCESS != (err = p_cuDevResourceGenerateDesc(&desc, resources, nbResources))) {
        return {};
    }

    CUgreenCtx green_ctx = nullptr;
    if (CUDA_SUCCESS != (err = p_cuGreenCtxCreate(&green_ctx, desc, dev, flags))) {
        return {};
    }

    auto box = std::shared_ptr<const GreenCtxBox>(
        new GreenCtxBox{green_ctx},
        [](const GreenCtxBox* b) {
            GILReleaseGuard gil;
            pw_cuGreenCtxDestroy(b->resource);
            delete b;
        }
    );
    return GreenCtxHandle(box, &box->resource);
}

GreenCtxHandle create_green_ctx_handle_ref(CUgreenCtx green_ctx) {
    if (!green_ctx) {
        return {};
    }
    auto box = std::make_shared<const GreenCtxBox>(GreenCtxBox{green_ctx});
    return GreenCtxHandle(box, &box->resource);
}

// Thread-local cache of primary contexts indexed by device ID
static thread_local std::vector<ContextHandle> primary_context_cache;

ContextHandle get_primary_context(int device_id) {
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
        new ContextBox{ctx, {}},
        [device_id](const ContextBox* b) {
            context_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            p_cuDevicePrimaryCtxRelease(device_id);
            delete b;
        }
    );
    auto h = ContextHandle(box, &box->resource);
    context_registry.register_handle(ctx, h);

    // Update cache
    if (static_cast<size_t>(device_id) >= primary_context_cache.size()) {
        primary_context_cache.resize(device_id + 1);
    }
    primary_context_cache[device_id] = h;
    return h;
}

ContextHandle get_current_context() {
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

}  // namespace cuda_core::rt
