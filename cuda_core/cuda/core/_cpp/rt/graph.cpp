// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "context_scope.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuda_core::rt {

using namespace detail;

// Set a graph node's parameters with h_context current (an empty handle runs in
// the caller's context). Returns the cuGraphNodeSetParams status. A failure to
// restore the caller's context is returned separately in *restore_status so the
// caller can publish the metadata that depends on the successful update before
// raising it; if the update itself failed, its status is returned with the
// restoration failure recorded as its detail and *restore_status is CUDA_SUCCESS.
CUresult graph_node_set_params(CUgraphNode node, CUgraphNodeParams* params,
                               const ContextHandle& h_context,
                               CUresult* restore_status) noexcept {
    *restore_status = CUDA_SUCCESS;
    if (!p_cuGraphNodeSetParams) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUcontext previous = nullptr;
    int changed = 0;
    CUresult status = enter_context(h_context, &previous, &changed);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    {
        GILReleaseGuard gil;
        status = p_cuGraphNodeSetParams(node, params);
    }
    if (!changed) {
        return status;
    }
    CUresult restored = restore_context(previous);
    if (restored == CUDA_SUCCESS) {
        return status;
    }
    note_context_not_restored(previous, status, restored);
    if (status == CUDA_SUCCESS) {
        *restore_status = restored;
    }
    return status;
}

// ============================================================================
// Graph Handles
// ============================================================================

namespace {

struct NodeAttachment;
using GraphAttachmentMap = std::map<CUgraphNode, NodeAttachment*>;

struct GraphHierarchy;

// Standard-layout alias target for GraphHandle.
struct GraphBoxBase {
    CUgraph resource = nullptr;
};

// Canonical state for one CUgraph. Its GraphHandle aliases resource, whose
// address remains stable for the lifetime of the hierarchy.
struct GraphBox : GraphBoxBase {
    GraphHierarchy* hierarchy = nullptr;  // Non-owning back-reference.
    GraphBox* parent = nullptr;           // Null for the root graph.
    CUgraphNode owner_node = nullptr;     // Node in parent that owns this graph.
    GraphAttachmentMap attachments;       // Non-owning attachment index.
    HandleRegistry<CUgraphNode, GraphNodeHandle> node_handles;

    GraphBox(
            CUgraph resource_,
            GraphHierarchy* hierarchy_,
            GraphBox* parent_ = nullptr,
            CUgraphNode owner_node_ = nullptr) noexcept
        : GraphBoxBase{resource_},
          hierarchy(hierarchy_),
          parent(parent_),
          owner_node(owner_node_) {}
};

// Shared owner of stable GraphBox storage. Every GraphHandle aliases the same
// control block, so any graph handle keeps the entire hierarchy alive.
struct GraphHierarchy {
    std::list<GraphBox> graphs;  // Parent boxes precede their descendants.
    std::list<GraphBox> graveyard;  // Retired child graph tombstones.

    GraphBox* root() noexcept {
        return graphs.empty() ? nullptr : &graphs.front();
    }
};

// See REGISTRY_DESIGN.md (Level 1: Driver Handle -> Resource Handle)
using GraphRegistry = HandleRegistry<CUgraph, GraphHandle>;
static GraphRegistry graph_registry;

// Immutable resource owners for one version of a graph node's parameters.
// Inheriting DeferredCleanupItem lets CUDA's user-object destructor enqueue
// the payload without destroying owners on the callback thread.
struct NodeAttachment : DeferredCleanupItem {
    CUuserObject object = nullptr;
    std::array<OpaqueHandle, 2> owners;

    NodeAttachment(OpaqueHandle owner0, OpaqueHandle owner1)
        : owners{std::move(owner0), std::move(owner1)} {}
};

// shared_ptr deleters for the payloads that need one. Typed handles convert to
// OpaqueHandle by assignment and reuse their own control block, so they need no
// deleter here. The Python deleter follows the owner-release pattern used by
// the stream/deviceptr handles above.
void py_deleter(const void* p) noexcept {
    GILAcquireGuard gil;
    if (gil.acquired()) {
        Py_DECREF(const_cast<PyObject*>(static_cast<const PyObject*>(p)));
    }
}

void free_deleter(const void* p) noexcept {
    std::free(const_cast<void*>(p));
}

GraphBox* get_box(const GraphHandle& h) noexcept {
    auto* value = reinterpret_cast<const GraphBoxBase*>(h.get());
    return const_cast<GraphBox*>(
        static_cast<const GraphBox*>(value));
}

// Rekey a staged attachment map from source nodes to their cloned nodes.
// The caller must release the GIL before calling this function.
CUresult rekey_attachments(
        GraphAttachmentMap& attachments, CUgraph cloned_graph) {
    if (!cloned_graph) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphNodeFindInClone) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    GraphAttachmentMap remapped;
    while (!attachments.empty()) {
        auto attachment = attachments.extract(attachments.begin());
        CUgraphNode cloned_node = nullptr;
        CUresult status = p_cuGraphNodeFindInClone(
            &cloned_node, attachment.key(), cloned_graph);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        attachment.key() = cloned_node;
        if (!remapped.insert(std::move(attachment)).inserted) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    attachments.swap(remapped);
    return CUDA_SUCCESS;
}

struct StagedGraphMetadata {
    const GraphBox* source;
    GraphBox* clone;
    GraphAttachmentMap* attachments;
};
using StagedGraphMetadataList = std::vector<StagedGraphMetadata>;

// Copy a source hierarchy into detached metadata before CUDA mutation.
void stage_graph_metadata(
        const GraphBox& source,
        GraphBox& clone,
        GraphAttachmentMap& attachments,
        std::list<GraphBox>& subgraphs,
        StagedGraphMetadataList& staged) {
    attachments = source.attachments;
    staged.push_back({&source, &clone, &attachments});

    for (const GraphBox& source_child : source.hierarchy->graphs) {
        if (source_child.parent != &source || !source_child.resource) {
            continue;
        }
        GraphBox& cloned_child = subgraphs.emplace_back(
            nullptr,
            clone.hierarchy,
            &clone,
            nullptr);
        stage_graph_metadata(
            source_child,
            cloned_child,
            cloned_child.attachments,
            subgraphs,
            staged);
    }
}

// Bind staged metadata to a CUDA-cloned hierarchy. The root clone resource
// must be populated before entry. The caller must release the GIL.
CUresult rekey_graph_metadata(
        StagedGraphMetadataList& staged) {
    if (!p_cuGraphNodeFindInClone || !p_cuGraphChildGraphNodeGetGraph) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUresult status;
    for (size_t i = 0; i < staged.size(); ++i) {
        const GraphBox& source = *staged[i].source;
        GraphBox& clone = *staged[i].clone;
        if (i != 0) {
            CUgraphNode cloned_owner = nullptr;
            status = p_cuGraphNodeFindInClone(
                &cloned_owner,
                source.owner_node,
                clone.parent->resource);
            if (status == CUDA_SUCCESS) {
                status = p_cuGraphChildGraphNodeGetGraph(
                    cloned_owner, &clone.resource);
            }
            if (status != CUDA_SUCCESS) {
                return status;
            }
            clone.owner_node = cloned_owner;
        }

        status = rekey_attachments(
            *staged[i].attachments, clone.resource);
        if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    return CUDA_SUCCESS;
}

}  // namespace

OpaqueHandle make_opaque_py(PyObject* obj) {
    Py_INCREF(obj);
    return OpaqueHandle(static_cast<const void*>(obj), py_deleter);
}

OpaqueHandle make_opaque_malloc(void* buf) {
    return OpaqueHandle(static_cast<const void*>(buf), free_deleter);
}

// State held by PreparedAttachment between preparation and commit. It keeps the
// graph alive, tracks the graph-retained replacement, and holds a preallocated
// map entry so commit cannot allocate. Destroying PreparedAttachment rolls back
// the staged user-object retain unless graph_commit_attachment publishes it.
struct PreparedAttachmentState {
    GraphHandle h_graph;
    NodeAttachment* replacement = nullptr;
    GraphAttachmentMap::node_type replacement_entry;

    explicit PreparedAttachmentState(GraphHandle h_graph_)
        : h_graph(std::move(h_graph_)) {}
};

void rollback_prepared_attachment(
        PreparedAttachmentState* state) noexcept {
    if (!state) {
        return;
    }
    if (state->replacement) {
        GraphBox* box = get_box(state->h_graph);
        if (box->resource) {
            GILReleaseGuard gil;
            pw_cuGraphReleaseUserObject(
                box->resource, state->replacement->object, 1);
        }
    }
    delete state;
}

// Detached metadata for a replacement embedded graph hierarchy. Preparation
// copies every attachment map and allocates every GraphBox before CUDA destroys
// the old embedded graph. Commit only rekeys and publishes it.
struct PreparedChildGraphUpdateState {
    GraphHandle h_parent;
    GraphHandle h_source;
    GraphBox* old_root = nullptr;
    CUgraphNode owner_node = nullptr;
    std::list<GraphBox> replacement;
    StagedGraphMetadataList staged;
    std::vector<GraphHandle> handles;

    PreparedChildGraphUpdateState(
            GraphHandle h_parent_,
            GraphHandle h_source_,
            GraphBox* old_root_,
            CUgraphNode owner_node_)
        : h_parent(std::move(h_parent_)),
          h_source(std::move(h_source_)),
          old_root(old_root_),
          owner_node(owner_node_) {}
};

GraphHandle create_graph_handle(CUgraph graph) {
    if (!graph) {
        return {};
    }

    auto hierarchy = std::shared_ptr<GraphHierarchy>(
        new GraphHierarchy{},
        [](GraphHierarchy* hierarchy) {
            for (const GraphBox& box : hierarchy->graphs) {
                if (box.resource) {
                    graph_registry.unregister_handle(box.resource);
                }
            }
            GraphBox* root = hierarchy->root();
            if (root && root->resource) {
                GILReleaseGuard gil;
                pw_cuGraphDestroy(root->resource);
            }
            retry_deferred_cleanup();
            delete hierarchy;
        }
    );
    GraphBox& root = hierarchy->graphs.emplace_back(
        graph, hierarchy.get());

    GraphHandle h_graph(hierarchy, &root.resource);
    graph_registry.register_handle(graph, h_graph);
    return h_graph;
}

GraphHandle create_child_graph_handle(
        CUgraph child_graph, const GraphHandle& h_parent,
        CUgraphNode owner_node) {
    if (!child_graph || !h_parent || !owner_node) {
        return {};
    }
    if (GraphHandle h_graph = graph_registry.lookup(child_graph)) {
        return h_graph;
    }

    GraphBox* parent = get_box(h_parent);
    GraphHierarchy* hierarchy = parent->hierarchy;
    GraphBox& child = hierarchy->graphs.emplace_back(
        child_graph, hierarchy, parent, owner_node);

    GraphHandle h_child(h_parent, &child.resource);
    graph_registry.register_handle(child_graph, h_child);
    return h_child;
}

CUresult graph_prepare_child_graph_update(
        const GraphHandle& h_parent,
        const GraphHandle& h_old_child,
        CUgraphNode owner_node,
        const GraphHandle& h_source,
        PreparedChildGraphUpdate* out_prepared) {
    if (!h_parent || !h_old_child || !owner_node ||
        !h_source || !out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();

    GraphBox* parent = get_box(h_parent);
    GraphBox* old_root = get_box(h_old_child);
    GraphBox* source = get_box(h_source);
    // A source from the destination hierarchy can include the old embedded
    // subtree whose raw node keys CUDA destroys during replacement.
    if (!parent->resource || !old_root->resource || !source->resource ||
        old_root->parent != parent ||
        old_root->owner_node != owner_node ||
        source->hierarchy == parent->hierarchy) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    PreparedChildGraphUpdate prepared =
        std::make_shared<PreparedChildGraphUpdateState>(
            h_parent, h_source, old_root, owner_node);

    GraphBox& replacement_root =
        prepared->replacement.emplace_back(
            nullptr, parent->hierarchy, parent, owner_node);
    stage_graph_metadata(
        *source,
        replacement_root,
        replacement_root.attachments,
        prepared->replacement,
        prepared->staged);

    const size_t graph_count = prepared->staged.size();
    prepared->handles.reserve(graph_count);
    for (const StagedGraphMetadata& graph : prepared->staged) {
        prepared->handles.emplace_back(
            h_parent, &graph.clone->resource);
    }

    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

void publish_child_graph_update(
        PreparedChildGraphUpdateState& state,
        GraphHandle* out_child) {
    GraphBox* parent = get_box(state.h_parent);
    parent->hierarchy->graphs.splice(
        parent->hierarchy->graphs.end(), state.replacement);
    *out_child = state.handles.front();
    graph_registry.register_handles(state.handles);
}

CUresult graph_commit_child_graph_update(
        PreparedChildGraphUpdate& prepared,
        GraphHandle* out_child) {
    if (!prepared || !out_child) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_child->reset();

    PreparedChildGraphUpdateState& state = *prepared;
    GraphBox* parent = get_box(state.h_parent);
    if (!parent->resource || !state.old_root->resource) {
        prepared.reset();
        return CUDA_ERROR_INVALID_VALUE;
    }

    CUresult status = CUDA_ERROR_NOT_SUPPORTED;
    CUgraph cloned_root = nullptr;
    if (p_cuGraphChildGraphNodeGetGraph) {
        GILReleaseGuard gil;
        status = p_cuGraphChildGraphNodeGetGraph(
            state.owner_node, &cloned_root);
        if (status == CUDA_SUCCESS) {
            state.staged.front().clone->resource = cloned_root;
            status = rekey_graph_metadata(state.staged);
        }
    }

    // CUDA has already destroyed the old embedded graph. No replacement
    // metadata is visible yet, so this selects only the old generation.
    invalidate_child_graph_state(
        state.h_parent, state.owner_node);

    if (status != CUDA_SUCCESS) {
        prepared.reset();
        throw std::runtime_error(
            "failed to update graph metadata after child graph replacement");
    }

    publish_child_graph_update(state, out_child);
    prepared.reset();
    return status;
}

CUresult graph_get_attachment(
        const GraphHandle& h_graph, CUgraphNode node,
        OpaqueHandle* owner0, OpaqueHandle* owner1) {
    if (!h_graph || !node || (!owner0 && !owner1)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (owner0) {
        owner0->reset();
    }
    if (owner1) {
        owner1->reset();
    }

    GraphBox* box = get_box(h_graph);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto it = box->attachments.find(node);
    if (it != box->attachments.end()) {
        if (owner0) {
            *owner0 = it->second->owners[0];
        }
        if (owner1) {
            *owner1 = it->second->owners[1];
        }
    }
    return CUDA_SUCCESS;
}

CUresult graph_prepare_attachment(
        const GraphHandle& h_graph,
        OpaqueHandle owner0,
        OpaqueHandle owner1,
        PreparedAttachment* out_prepared) {
    if (!out_prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    out_prepared->reset();
    if (!h_graph) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphBox* box = get_box(h_graph);
    if (!box->resource) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!p_cuGraphReleaseUserObject) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    PreparedAttachment prepared(
        new PreparedAttachmentState(h_graph),
        PreparedAttachmentDeleter{rollback_prepared_attachment});
    if (owner0 || owner1) {
        if (!p_cuUserObjectCreate || !p_cuUserObjectRelease ||
            !p_cuGraphRetainUserObject) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        ensure_deferred_cleanup_ready();
        prepared->replacement = new NodeAttachment(
            std::move(owner0), std::move(owner1));
        GraphAttachmentMap staged;
        try {
            staged.emplace(nullptr, prepared->replacement);
            prepared->replacement_entry =
                staged.extract(staged.begin());
        } catch (...) {
            delete prepared->replacement;
            prepared->replacement = nullptr;
            throw;
        }
        auto* cleanup_item =
            static_cast<DeferredCleanupItem*>(
                prepared->replacement);

        CUuserObject object = nullptr;
        CUresult status;
        {
            GILReleaseGuard gil;
            status = p_cuUserObjectCreate(
                &object, cleanup_item,
                reinterpret_cast<CUhostFn>(enqueue_cleanup),
                1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC);
            if (status != CUDA_SUCCESS) {
                prepared->replacement_entry.mapped() = nullptr;
                delete prepared->replacement;
                prepared->replacement = nullptr;
                return status;
            }
            prepared->replacement->object = object;
            status = p_cuGraphRetainUserObject(
                box->resource, object, 1, CU_GRAPH_USER_OBJECT_MOVE);
            if (status != CUDA_SUCCESS) {
                prepared->replacement_entry.mapped() = nullptr;
                prepared->replacement = nullptr;
                pw_cuUserObjectRelease(object, 1);
                return status;
            }
        }
    }

    *out_prepared = std::move(prepared);
    return CUDA_SUCCESS;
}

CUresult graph_commit_attachment(
        PreparedAttachment& prepared,
        CUgraphNode node) {
    if (!prepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphHandle h_graph = prepared->h_graph;
    GraphBox* box = get_box(h_graph);
    if (!box->resource || (!node && !prepared->replacement)) {
        delete prepared.release();
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!node) {
        delete prepared.release();
        return CUDA_SUCCESS;
    }

    // Publish the replacement or removal before releasing the previous graph
    // reference; that release can make the previous payload eligible for
    // destruction.
    NodeAttachment* previous = nullptr;
    auto it = box->attachments.find(node);
    if (it == box->attachments.end()) {
        if (prepared->replacement) {
            prepared->replacement_entry.key() = node;
            auto result = box->attachments.insert(
                std::move(prepared->replacement_entry));
            if (!result.inserted) {
                prepared->replacement_entry =
                    std::move(result.node);
                delete prepared.release();
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    } else {
        previous = it->second;
        if (prepared->replacement) {
            it->second = prepared->replacement;
        } else {
            box->attachments.erase(it);
        }
    }

    delete prepared.release();
    if (!previous) {
        return CUDA_SUCCESS;
    }
    GILReleaseGuard gil;
    return p_cuGraphReleaseUserObject(
        box->resource, previous->object, 1);
}

CUresult graph_clone_attachments(
        const GraphHandle& h_clone,
        const GraphHandle& h_source) {
    if (!h_clone || !h_source) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    GraphBox* clone = get_box(h_clone);
    GraphBox* source = get_box(h_source);
    if (!clone->resource || !source->resource ||
        !clone->attachments.empty()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Build and rekey the clone metadata off-hierarchy so a CUDA mapping error
    // cannot partially publish it.
    GraphAttachmentMap attachments;
    std::list<GraphBox> subgraphs;
    StagedGraphMetadataList staged;
    stage_graph_metadata(
        *source, *clone, attachments, subgraphs, staged);

    std::vector<GraphHandle> handles;
    handles.reserve(subgraphs.size());
    for (GraphBox& graph : subgraphs) {
        handles.emplace_back(h_clone, &graph.resource);
    }

    CUresult status;
    {
        GILReleaseGuard gil;
        status = rekey_graph_metadata(staged);
    }
    if (status != CUDA_SUCCESS) {
        return status;
    }

    clone->attachments.swap(attachments);
    if (subgraphs.empty()) {
        return CUDA_SUCCESS;
    }

    clone->hierarchy->graphs.splice(
        clone->hierarchy->graphs.end(), subgraphs);
    graph_registry.register_handles(handles);
    return CUDA_SUCCESS;
}

namespace {
struct GraphNodeBox {
    mutable CUgraphNode resource;
    GraphHandle h_graph;
};
}  // namespace

static const GraphNodeBox* get_box(const GraphNodeHandle& h) {
    const CUgraphNode* p = h.get();
    return reinterpret_cast<const GraphNodeBox*>(
        reinterpret_cast<const char*>(p) - offsetof(GraphNodeBox, resource)
    );
}

// graphs is ordered parent-before-child. Nulling a selected box marks its
// later descendants, whose parent pointers remain valid after list splicing.
// This permits one allocation-free sweep of the hierarchy.
void invalidate_child_graph_state(
        const GraphHandle& h_parent,
        CUgraphNode owner_node) noexcept {
    if (!h_parent || !owner_node) {
        return;
    }

    GraphBox* parent = get_box(h_parent);
    if (!parent->resource) {
        return;
    }
    GraphHierarchy& hierarchy = *parent->hierarchy;
    for (auto it = hierarchy.graphs.begin();
         it != hierarchy.graphs.end();) {
        auto graph = it++;
        bool is_owned_root = graph->parent == parent &&
                             graph->owner_node == owner_node;
        bool is_descendant = graph->parent &&
                             !graph->parent->resource;
        if (!is_owned_root && !is_descendant) {
            continue;
        }

        // Empty node_handles and invalidate each one.
        for (auto& entry : graph->node_handles.drain()) {
            if (GraphNodeHandle h_node = entry.second.lock()) {
                get_box(h_node)->resource = nullptr;
            }
        }
        graph_registry.unregister_handle(graph->resource);
        graph->resource = nullptr;
        graph->attachments.clear();
        hierarchy.graveyard.splice(
            hierarchy.graveyard.end(), hierarchy.graphs, graph);
    }
}

GraphNodeHandle create_graph_node_handle(CUgraphNode node, const GraphHandle& h_graph) {
    if (!node) {
        auto box = std::make_shared<const GraphNodeBox>(
            GraphNodeBox{nullptr, h_graph});
        return GraphNodeHandle(box, &box->resource);
    }

    GraphBox* graph = get_box(h_graph);
    return graph->node_handles.get_or_create(
        node,
        [node, &h_graph] {
            auto box = std::make_shared<const GraphNodeBox>(
                GraphNodeBox{node, h_graph});
            return GraphNodeHandle(box, &box->resource);
        });
}

GraphHandle graph_node_get_graph(const GraphNodeHandle& h) noexcept {
    return h ? get_box(h)->h_graph : GraphHandle{};
}

void invalidate_graph_node(const GraphNodeHandle& h) noexcept {
    if (!h) {
        return;
    }

    const GraphNodeBox* node_box = get_box(h);
    CUgraphNode node = node_box->resource;
    if (!node) {
        return;
    }
    GraphBox* graph = get_box(node_box->h_graph);
    graph->node_handles.unregister_handle(node);
    node_box->resource = nullptr;
}

}  // namespace cuda_core::rt
