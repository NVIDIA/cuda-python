# Graph Node Attachments

## Purpose

CUDA user objects tie externally managed resource lifetimes to CUDA graphs.
cuda-core uses them to keep resources referenced by graph node parameters
alive. It also keeps non-owning metadata that maps each node to the
user-object attachment backing its current parameters. This mapping lets
cuda-core update the correct user-object references during graph changes and
reproduce the node associations in graph clones.

A `CUgraph` is a mutable graph definition made of nodes. Each node has a
complete parameter set that may refer to kernels, memory, events, callbacks, or
other external resources. CUDA does not normally take ownership of those
resources.

A definition can be cloned, embedded as a child graph, or instantiated as a
`CUgraphExec`. Each operation copies the definition's current nodes and
parameters. The source and copy then have independent lifetimes: changing the
source does not change an existing clone or executable. An embedded clone is
owned by its parent node and is destroyed when that node is removed or
replaced.

Definitions can add, delete, or replace nodes and their parameters. Executable
graphs can adopt a new definition through a whole-graph update: success
replaces their inherited state, while failure leaves the old state intact. An
in-flight launch may continue using an older parameter set even after its
source node or executable has changed or been destroyed. Direct
executable-node updates require separate executable-specific ownership and are
not tracked by this definition metadata.

Several versions of one logical node's parameters may therefore remain live at
the same time. Without version-specific ownership, replacing or deleting a node
could release memory, kernels, callbacks, or other resources that an existing
clone, executable, or launch still uses.

cuda-core prevents this by giving each parameter version an immutable
`NodeAttachment` owned through a graph-retained CUDA user object. Non-owning
metadata records the current attachment for each graph node so cuda-core can
update the correct user-object references as graphs change.

This document explains that ownership model and provides a map to the data
structures and functions that implement it.

## Ownership model

For each resource-bearing node version, cuda-core:

1. Collects its resources in an immutable `NodeAttachment`.
2. Creates a CUDA user object that owns the attachment.
3. Retains one user-object reference on the `CUgraph`.

CUDA copies graph-owned user-object references when it clones or instantiates a
graph. It also keeps the references needed by in-flight launches. Replacing or
deleting a source node therefore releases only the source graph's reference.
Clones, executable graphs, and launches retain the old attachment for as long
as they need it.

```text
CUgraph definition ─┐
CUgraph clone ──────┼── retains ──> CUuserObject ── owns ──> NodeAttachment
CUgraphExec ────────┤                                      │
in-flight launch ───┘                                      └── owns resources

GraphAttachmentMap ── non-owning lookup ──> current NodeAttachment
```

The CUDA user-object reference count controls the attachment lifetime.
`GraphAttachmentMap` only lets cuda-core find the attachment currently
associated with a node.

Each `NodeAttachment` contains two type-erased `OpaqueHandle` owners:

- kernel: kernel and argument storage
- host callback: callback and copied user data
- memcpy: destination and source
- memset or event: destination or event in the first owner

`OpaqueHandle` is `shared_ptr<const void>`. Existing cuda-core handles reuse
their shared ownership when converted to it. Python objects and copied callback
data use custom deleters.

## Graph hierarchy state

One `GraphHierarchy` represents a root graph and every CUDA-owned child or
conditional-body graph below it. Every graph in the hierarchy has one
canonical `GraphBox`.

A `GraphHandle` points at `GraphBox::resource` while sharing ownership of the
whole `GraphHierarchy`. Holding any root or child handle therefore keeps every
box and the root CUDA graph alive. Only the root graph is destroyed with
`cuGraphDestroy`; CUDA owns and destroys embedded child graphs.

`GraphBox` stores:

- the `CUgraph`
- its parent box and owning node, when it is a child graph
- a `GraphAttachmentMap` from `CUgraphNode` to `NodeAttachment*`
- a per-graph weak registry of canonical `GraphNodeHandle` objects

`GraphHierarchy::graphs` stores live boxes in parent-before-child order. When
CUDA destroys a child graph, cuda-core unregisters its raw handle, nulls the
resource, clears its non-owning metadata, and moves the box to
`GraphHierarchy::graveyard`. Keeping this tombstone at a stable address lets
existing aliasing handles safely observe that the graph is invalid. Removing
the registry entry also lets a future CUDA graph reuse the raw handle value.

The process-wide graph registry maps a live `CUgraph` to its canonical
`GraphHandle`. Node handles are scoped to their `GraphBox`, where they can all
be invalidated when CUDA destroys that graph.

## Code map

The C++ implementation is in
[resource_handles.cpp](resource_handles.cpp). Its public C++ declarations are
in [resource_handles.hpp](resource_handles.hpp), and the Cython declarations
are in [_resource_handles.pxd](../_resource_handles.pxd).

The main operations are:

- `create_graph_handle`: create the root box and hierarchy
- `create_child_graph_handle`: register a CUDA-owned child graph and return an
  aliasing handle
- `graph_get_attachment`: copy either or both current owners
- `graph_prepare_attachment`: graph-retain a staged replacement before mutation
- `graph_commit_attachment`: publish or anonymously retain the staged attachment
- `graph_clone_attachments`: copy attachment metadata into an embedded clone
- `invalidate_child_graph_state`: invalidate boxes and node handles after CUDA
  destroys child graphs

Private helpers `rekey_attachments` and `copy_attachments` implement recursive
clone mapping.

The Cython graph code creates CUDA nodes and then calls these operations to keep
the C++ metadata synchronized with the driver.

## Common operations

### Adding a node

Cython keeps the parameter owners alive while it calls the CUDA node-creation
API. Before the call, `graph_prepare_attachment` creates and graph-retains one
user object for the complete owner bundle. It also preallocates the metadata
entry needed by commit.

After CUDA returns the new node handle, `graph_commit_attachment` publishes the
preallocated entry without allocating. If preparation or node creation fails,
destruction of the `PreparedAttachment` automatically releases the staged graph
reference. CUDA therefore never receives node parameters whose owners could
not be retained.

### Reading and replacing attachments

`graph_get_attachment` copies the requested owners. Either output pointer may
be null when the caller needs only one owner. A missing attachment returns
empty handles.

`graph_prepare_attachment` creates and graph-retains a complete replacement
bundle before the CUDA parameter update. After the update succeeds,
`graph_commit_attachment` publishes the replacement before releasing the
graph's reference to the old user object, so metadata never points at a
released payload. Preparing two empty owners stages removal of the current
attachment.

A partial parameter update first gets the current owners, replaces the values
covered by the update, prepares the resulting complete attachment, asks CUDA
to apply the new complete parameter set, and then commits the attachment.

### Embedding a child graph

`cuGraphAddChildGraphNode` clones the source graph into a node owned by the
parent. CUDA copies all user-object references into that embedded clone.

After obtaining the embedded `CUgraph`, cuda-core creates its canonical child
box and calls `graph_clone_attachments`. This operation:

1. Copies the source graph's non-owning attachment map.
2. Uses `cuGraphNodeFindInClone` to replace source node keys with cloned node
   keys.
3. Finds nested cloned child graphs from their mapped owner nodes.
4. Recursively creates staged boxes and copies their maps.
5. Publishes the root map with `swap` and the descendant boxes with
   `list::splice`.
6. Registers canonical handles for the published descendant graphs.

All CUDA mapping happens before publication. A mapping error leaves the live
metadata unchanged.

### Deleting a node

cuda-core asks CUDA to destroy the node before changing any wrapper or
attachment state. If CUDA rejects deletion, all cuda-core state remains
unchanged. An empty `PreparedAttachment` is created before the call and
committed only after deletion succeeds.

After a successful deletion, cuda-core:

1. Clears the node attachment and releases that graph's user-object reference.
2. Finds child graphs owned by the node.
3. Invalidates their graph and node handles.
4. Clears their non-owning metadata and moves their boxes to the graveyard.
5. Invalidates the deleted node handle.

CUDA already destroyed those child graphs and released their user-object
references, so cuda-core does not release the child attachments again.

### Captured host callbacks

`cuLaunchHostFunc` can add a host node during stream capture but does not return
its `CUgraphNode`. cuda-core normally recovers the node from the stream's
capture dependencies and records the callback attachment there.

If CUDA accepted the callback but its node cannot be identified, cuda-core
commits the prepared attachment with a null node. This retains the owners
anonymously on the graph without publishing node metadata. The owners then
remain alive until graph destruction, preventing a dangling callback pointer.

### Deferred cleanup

CUDA invokes a user-object destructor on an internal thread where CUDA API
calls are forbidden. Destroying an attachment there could release handles whose
deleters call CUDA or run Python finalizers.

`NodeAttachment` therefore inherits from `DeferredCleanupItem`. The CUDA
destructor callback only adds the attachment to the process-lifetime
`DeferredCleanupQueue` and requests a `Py_AddPendingCall`.

One pending call drains all queued attachments from Python's main thread. The
queue coalesces work because CPython's pending-call queue is bounded. If
scheduling fails, attachments stay queued and a later enqueue or safe cuda-core
entry retries. Graph and executable-graph destruction and explicit close paths
provide additional retry points. During Python finalization, scheduling stops
and unreclaimable attachments are intentionally leaked rather than destroyed
in an unsafe context.

## Invariants

1. A published `NodeAttachment` is immutable.
2. CUDA user-object references, not metadata pointers, own attachments.
3. Metadata is removed or replaced before its graph reference is released.
4. Fallible attachment setup and metadata allocation happen before the CUDA
   graph mutation they support.
5. Every live cuda-core `CUgraph` has one canonical `GraphBox` and registry
   entry.
6. Graph boxes remain in parent-before-child order.
7. Destroyed child boxes remain at stable addresses in the graveyard.
8. A raw graph handle is unregistered before its box becomes a tombstone.
9. CUDA callbacks only enqueue attachments; they never release owners or call
   CUDA.
10. Graph mutations and their metadata updates require the same external
   synchronization as the underlying CUDA graph.

## Scope

- Attachment metadata tracks graph mutations performed through cuda-core.
- Raw driver clones receive the CUDA user-object references needed for safe
  execution, but cuda-core cannot reconstruct their node-to-attachment map.
- Executable graphs rely on CUDA's inherited user-object references; they do
  not use `GraphAttachmentMap`.
- Direct executable-node updates require separate executable ownership and are
  not tracked by definition attachment metadata.
- Stream capture explicitly retains host callbacks. Other captured operations
  keep their documented caller-owned lifetime contract.
