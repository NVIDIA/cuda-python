// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// Graph Exec Handles
// ============================================================================

namespace {

// Append-only owners introduced by individual executable-node updates. CUDA
// owns this payload through a user object propagated into the CUgraphExec.
struct ExecAttachments : DeferredCleanupItem {
    CUuserObject object = nullptr;
    std::vector<OpaqueHandle> owners;
};

struct GraphExecBox {
    CUgraphExec resource = nullptr;
    ExecAttachments* attachments = nullptr;  // Non-owning.

    ~GraphExecBox() noexcept {
        if (resource) {
            GILReleaseGuard gil;
            pw_cuGraphExecDestroy(resource);
        }
        // The accumulator fields may be dangling after exec destruction.
        retry_deferred_cleanup();
    }
};

GraphExecBox* get_exec_box(const GraphExecHandle& h) noexcept {
    return const_cast<GraphExecBox*>(
        reinterpret_cast<const GraphExecBox*>(h.get()));
}

GraphExecHandle make_graph_exec_handle(
        CUgraphExec graph_exec, ExecAttachments* attachments) {
    struct RawGraphExecGuard {
        CUgraphExec resource;

        ~RawGraphExecGuard() noexcept {
            if (resource) {
                GILReleaseGuard gil;
                pw_cuGraphExecDestroy(resource);
            }
            retry_deferred_cleanup();
        }
    } guard{graph_exec};

    auto box = std::make_shared<GraphExecBox>();
    box->resource = graph_exec;
    box->attachments = attachments;
    guard.resource = nullptr;
    return GraphExecHandle(box, &box->resource);
}

// Holds a fresh accumulator retained on the source graph across a CUDA call
// that propagates user objects into an exec. Releasing drops the source's
// reference: after successful propagation the exec keeps the accumulator
// alive, and otherwise this drops its last reference.
struct ExecAttachmentStaging {
    GraphHandle h_source;
    ExecAttachments* accumulator = nullptr;

    ~ExecAttachmentStaging() noexcept {
        report_cuda_error("cuGraphReleaseUserObject", release(),
                          "failed while dropping a staged graph attachment");
    }

    CUresult release() noexcept {
        if (!h_source || !accumulator) {
            return CUDA_SUCCESS;
        }
        const CUuserObject object = accumulator->object;
        const GraphHandle source = std::move(h_source);
        accumulator = nullptr;
        GILReleaseGuard gil;
        return p_cuGraphReleaseUserObject(*source, object, 1);
    }
};

// Create an accumulator and retain it on h_source, so that a following
// instantiation or whole-graph update propagates a reference into the exec.
CUresult stage_exec_attachments(
        const GraphHandle& h_source, ExecAttachmentStaging* out_staging) {
    if (!p_cuUserObjectCreate || !p_cuUserObjectRelease ||
        !p_cuGraphRetainUserObject || !p_cuGraphReleaseUserObject) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    ensure_deferred_cleanup_ready();
    auto* accumulator = new ExecAttachments;

    CUuserObject object = nullptr;
    CUresult status;
    {
        GILReleaseGuard gil;
        status = p_cuUserObjectCreate(
            &object,
            static_cast<DeferredCleanupItem*>(accumulator),
            reinterpret_cast<CUhostFn>(enqueue_cleanup),
            1,
            CU_USER_OBJECT_NO_DESTRUCTOR_SYNC);
        if (status != CUDA_SUCCESS) {
            delete accumulator;
            return status;
        }
        accumulator->object = object;
        status = p_cuGraphRetainUserObject(
            *h_source, object, 1, CU_GRAPH_USER_OBJECT_MOVE);
        if (status != CUDA_SUCCESS) {
            // Dropping the last reference retires the accumulator.
            pw_cuUserObjectRelease(object, 1);
            return status;
        }
    }

    out_staging->h_source = h_source;
    out_staging->accumulator = accumulator;
    return CUDA_SUCCESS;
}

}  // namespace

// State held by PreparedExecAttachment between preparation and commit. It keeps
// the exec alive and remembers the accumulator size before the append, so that
// rollback can drop owners staged for a mutation that CUDA rejected.
struct PreparedExecAttachmentState {
    GraphExecHandle h_exec;
    ExecAttachments* attachments = nullptr;
    size_t original_size = 0;

    PreparedExecAttachmentState(
            GraphExecHandle h_exec_,
            ExecAttachments* attachments_,
            size_t original_size_)
        : h_exec(std::move(h_exec_)),
          attachments(attachments_),
          original_size(original_size_) {}
};

void rollback_prepared_exec_attachment(
        PreparedExecAttachmentState* state) noexcept {
    if (!state) {
        return;
    }
    if (state->attachments) {
        while (state->attachments->owners.size() > state->original_size) {
            state->attachments->owners.pop_back();
        }
    }
    delete state;
}

GraphExecHandle create_graph_exec_handle(
        const GraphHandle& h_source,
        CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    if (!h_source || !*h_source || !params) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }
    if (!p_cuGraphInstantiateWithParams) {
        err = CUDA_ERROR_NOT_SUPPORTED;
        return {};
    }

    ExecAttachmentStaging staging;
    if (CUDA_SUCCESS != (err = stage_exec_attachments(h_source, &staging))) {
        return {};
    }

    CUgraphExec graph_exec = nullptr;
    {
        GILReleaseGuard gil;
        err = p_cuGraphInstantiateWithParams(&graph_exec, *h_source, params);
    }
    if (err != CUDA_SUCCESS) {
        return {};
    }
    // CUDA can report a specific failure while returning success. The exec is
    // then unusable, so it stays unadopted for the caller to diagnose from
    // params->result_out.
    if (params->result_out != CUDA_GRAPH_INSTANTIATE_SUCCESS) {
        return {};
    }
    if (!graph_exec) {
        err = CUDA_ERROR_INVALID_VALUE;
        return {};
    }

    GraphExecHandle h_exec = make_graph_exec_handle(
        graph_exec, staging.accumulator);
    if (CUDA_SUCCESS != (err = staging.release())) {
        return {};
    }
    return h_exec;
}

CUresult graph_exec_update(
        const GraphExecHandle& h_exec,
        const GraphHandle& h_source,
        CUgraphExecUpdateResultInfo* result_info) {
    if (!h_exec || !h_source || !*h_source || !result_info) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphExecUpdate) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    GraphExecBox* box = get_exec_box(h_exec);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    ExecAttachmentStaging staging;
    CUresult status = stage_exec_attachments(h_source, &staging);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    {
        GILReleaseGuard gil;
        status = p_cuGraphExecUpdate(box->resource, *h_source, result_info);
    }
    if (status != CUDA_SUCCESS) {
        return status;
    }

    // CUDA may already have retired the old accumulator. Publish the new one
    // before releasing the source graph's temporary reference.
    box->attachments = staging.accumulator;
    return staging.release();
}

CUresult graph_prepare_exec_attachment(
        const GraphExecHandle& h_exec,
        OpaqueHandle owner0,
        OpaqueHandle owner1,
        PreparedExecAttachment* out_prepared) {
    if (!out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();
    if (!h_exec) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphExecBox* box = get_exec_box(h_exec);
    if (!box->resource || !box->attachments) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    ExecAttachments* attachments = box->attachments;
    const size_t original_size = attachments->owners.size();
    const size_t additions =
        static_cast<size_t>(static_cast<bool>(owner0)) +
        static_cast<size_t>(static_cast<bool>(owner1));
    // Reserve before staging so that rollback and commit cannot allocate.
    attachments->owners.reserve(original_size + additions);
    PreparedExecAttachment prepared(
        new PreparedExecAttachmentState(h_exec, attachments, original_size),
        PreparedExecAttachmentDeleter{rollback_prepared_exec_attachment});
    if (owner0) {
        attachments->owners.emplace_back(std::move(owner0));
    }
    if (owner1) {
        attachments->owners.emplace_back(std::move(owner1));
    }
    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

void graph_commit_exec_attachment(
        PreparedExecAttachment& prepared) noexcept {
    delete prepared.release();
}

}  // namespace cuda_core::rt
