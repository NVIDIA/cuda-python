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

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Event Handles
// ============================================================================

namespace {
struct EventBox {
    CUevent resource;
    bool timing_enabled;
    bool is_blocking_sync;
    bool ipc_enabled;
    int device_id;
    ContextHandle h_context;
};
}  // namespace

static const EventBox* get_box(const EventHandle& h) {
    const CUevent* p = h.get();
    return reinterpret_cast<const EventBox*>(
        reinterpret_cast<const char*>(p) - offsetof(EventBox, resource)
    );
}

bool get_event_timing_enabled(const EventHandle& h) noexcept {
    return h ? get_box(h)->timing_enabled : false;
}

bool get_event_is_blocking_sync(const EventHandle& h) noexcept {
    return h ? get_box(h)->is_blocking_sync : false;
}

bool get_event_ipc_enabled(const EventHandle& h) noexcept {
    return h ? get_box(h)->ipc_enabled : false;
}

int get_event_device_id(const EventHandle& h) noexcept {
    return h ? get_box(h)->device_id : -1;
}

// Return the context retained by an event handle.
ContextHandle get_event_context(const EventHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUevent, EventHandle> event_registry;

EventHandle create_event_handle(const ContextHandle& h_ctx, unsigned int flags,
                                bool timing_enabled, bool is_blocking_sync,
                                bool ipc_enabled, int device_id) {
    GILReleaseGuard gil;
    CUevent event = nullptr;
    err = invoke_in_context_or_undo(
        h_ctx,
        [&]() noexcept { return p_cuEventCreate(&event, flags); },
        [&]() noexcept { pw_cuEventDestroy(event); },
        /*undo_requires_target_context=*/false);
    if (err != CUDA_SUCCESS) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event, timing_enabled, is_blocking_sync, ipc_enabled, device_id, h_ctx},
        [](const EventBox* b) {
            event_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuEventDestroy(b->resource);
            delete b;
        }
    );
    EventHandle h(box, &box->resource);
    event_registry.register_handle(event, h);
    return h;
}

EventHandle create_event_handle_for_stream(CUstream stream, unsigned int flags) {
    // Resolve the stream's owning context (for default-stream tokens this is
    // the current context, per cuStreamGetCtx) and create the event there, so
    // it can be recorded on `stream` no matter which context is current.
    CUcontext ctx = nullptr;
    {
        GILReleaseGuard gil;
        err = p_cuStreamGetCtx(stream, &ctx);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }
    if (!ctx) {
        err = CUDA_ERROR_INVALID_CONTEXT;
        return {};
    }
    return create_event_handle(create_context_handle_ref(ctx), flags, false, false, false, -1);
}

EventHandle create_event_handle_ref(CUevent event) {
    if (auto h = event_registry.lookup(event)) {
        return h;
    }
    auto box = std::make_shared<const EventBox>(EventBox{event, false, false, false, -1, {}});
    return EventHandle(box, &box->resource);
}

EventHandle create_event_handle_ipc(const CUipcEventHandle& ipc_handle,
                                    bool is_blocking_sync) {
    GILReleaseGuard gil;
    CUevent event;
    if (CUDA_SUCCESS != (err = p_cuIpcOpenEventHandle(&event, ipc_handle))) {
        return {};
    }

    auto box = std::shared_ptr<const EventBox>(
        new EventBox{event, false, is_blocking_sync, true, -1, {}},
        [](const EventBox* b) {
            event_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuEventDestroy(b->resource);
            delete b;
        }
    );
    EventHandle h(box, &box->resource);
    event_registry.register_handle(event, h);
    return h;
}

}  // namespace cuda_core::rt
