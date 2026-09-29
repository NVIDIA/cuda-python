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
#include <thread>
#include <utility>

namespace cuda_core::rt {

using namespace detail;

namespace {
// Return whether a stream handle needs a current context to resolve it.
bool is_default_stream(CUstream stream) noexcept {
    return stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD;
}
}  // namespace

namespace detail {
// Return the context a deallocation-stream token must run under. Real streams
// resolve their own context; default-stream tokens use the context bound at
// allocation time. Warn when PTDS deallocation crosses host threads.
ContextHandle deallocation_context(const DeallocationStream& stream) noexcept {
    if (!is_default_stream(as_cu(stream.h_stream))) {
        return {};
    }
    if (stream.ptds_tid != std::thread::id{}
            && stream.ptds_tid != std::this_thread::get_id()) {
        report_message(
            "Buffer deallocation for a per-thread default stream "
            "is running on a different host thread than the one that recorded "
            "the deallocation stream; ordering relative to the allocating "
            "thread's PTDS is not preserved");
    }
    return get_stream_context(stream.h_stream);
}
}  // namespace detail

// ============================================================================
// Stream Handles
// ============================================================================

namespace {
struct StreamBox {
    CUstream resource;
    ContextHandle h_context;
};

static const StreamBox* get_box(const StreamHandle& h) noexcept {
    const CUstream* p = h.get();
    return reinterpret_cast<const StreamBox*>(
        reinterpret_cast<const char*>(p) - offsetof(StreamBox, resource)
    );
}

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
static HandleRegistry<CUstream, StreamHandle> stream_registry;
}  // namespace

StreamHandle create_stream_handle(const ContextHandle& h_ctx, unsigned int flags, int priority) {
    GILReleaseGuard gil;
    CUstream stream = nullptr;
    GreenCtxHandle h_green = get_context_green_ctx(h_ctx);
    if (h_green) {
        err = p_cuGreenCtxStreamCreate
            ? p_cuGreenCtxStreamCreate(&stream, as_cu(h_green), flags, priority)
            : CUDA_ERROR_NOT_SUPPORTED;
    } else {
        err = invoke_in_context_or_undo(
            h_ctx,
            [&]() noexcept { return p_cuStreamCreateWithPriority(&stream, flags, priority); },
            [&]() noexcept { pw_cuStreamDestroy(stream); },
            /*undo_requires_target_context=*/false);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }

    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, h_ctx},
        [](const StreamBox* b) {
            stream_registry.unregister_handle(b->resource);
            GILReleaseGuard gil;
            pw_cuStreamDestroy(b->resource);
            delete b;
        }
    );
    StreamHandle h(box, &box->resource);
    stream_registry.register_handle(stream, h);
    return h;
}

StreamHandle create_stream_handle_ref(CUstream stream) {
    if (auto h = stream_registry.lookup(stream)) {
        return h;
    }
    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, {}},
        [](const StreamBox* b) {
            stream_registry.unregister_handle(b->resource);
            delete b;
        }
    );
    StreamHandle h(box, &box->resource);
    stream_registry.register_handle(stream, h);
    return h;
}

StreamHandle create_stream_handle_with_owner(CUstream stream, PyObject* owner) {
    if (auto h = stream_registry.lookup(stream)) {
        // Reuse handles that already carry structural context metadata, e.g.
        // cuda-core-owned streams.
        if (get_box(h)->h_context) {
            return h;
        }
    }
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
    // Owner-backed handles are NOT registered in the stream registry to avoid
    // corruption when multiple owners wrap the same CUstream (each stacks its
    // own Py_INCREF/Py_DECREF independently).
    auto box = std::shared_ptr<const StreamBox>(
        new StreamBox{stream, {}},
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

// Return the context retained by a stream handle.
ContextHandle get_stream_context(const StreamHandle& h) noexcept {
    return h ? get_box(h)->h_context : ContextHandle{};
}

StreamHandle get_legacy_stream() {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_LEGACY);
    return handle;
}

StreamHandle get_per_thread_stream() {
    static StreamHandle handle = create_stream_handle_ref(CU_STREAM_PER_THREAD);
    return handle;
}

StreamHandle create_context_bound_legacy_stream(const ContextHandle& h_context) {
    if (!h_context) {
        return {};
    }
    // Default deleter: this handle never owns CU_STREAM_LEGACY, so nothing
    // needs to run when the last reference is released.
    auto box = std::make_shared<const StreamBox>(StreamBox{CU_STREAM_LEGACY, h_context});
    return StreamHandle(box, &box->resource);
}

// ============================================================================
// Deallocation streams
//
// A DeallocationStream is a StreamHandle used for ordering frees. It differs
// from an ordinary StreamHandle only for default-stream tokens, for which it
// stores the (de)allocation context. Ordinarily, the LEGACY and PER_THREAD
// default streams resolve to whichever context is active at the time they are
// used, but for storing deallocation recipes we need to pin the context. With
// the PER_THREAD token, it is not possible to restore the original stream when
// deallocation runs on a different thread. Therefore, in that case the
// allocating host thread id is also stored so that cross-thread frees can be
// detected and warnings can be issued.
// ============================================================================

namespace detail {
// Real streams are copied unchanged. Default-stream tokens without an embedded
// context are bound to the current context. Returns false (and sets err) when a
// default-stream token cannot be bound because no context is current.
bool make_deallocation_stream(
        const StreamHandle& h, DeallocationStream& out) noexcept {
    out = {};
    if (!h) {
        return true;
    }

    const CUstream stream = as_cu(h);
    if (!is_default_stream(stream)) {
        out = DeallocationStream{h, {}};
        return true;
    }

    StreamHandle h_bound = h;
    if (!get_stream_context(h)) {
        ContextHandle h_ctx = get_current_context();
        if (!h_ctx) {
            if (err == CUDA_SUCCESS) {
                err = CUDA_ERROR_INVALID_CONTEXT;
            }
            return false;
        }
        // Do not register in stream_registry: the token value alone is not
        // a unique stream identity (context is part of the meaning).
        auto box = std::shared_ptr<const StreamBox>(
            new StreamBox{stream, h_ctx});
        h_bound = StreamHandle(box, &box->resource);
    }

    std::thread::id ptds_tid{};
    if (stream == CU_STREAM_PER_THREAD) {
        ptds_tid = std::this_thread::get_id();
    }
    out = DeallocationStream{std::move(h_bound), ptds_tid};
    return true;
}
}  // namespace detail

}  // namespace cuda_core::rt
