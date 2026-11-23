# Handle and Object Registries

When Python-managed objects round-trip through the CUDA driver (e.g.,
querying a graph's nodes and getting back raw `CUgraphNode` pointers),
we need to recover the original Python object rather than creating a
duplicate.

This document describes the approach used to achieve this. The pattern
is driven mainly by needs arising in the context of CUDA graphs, but
it is general and can be extended to other object types as needs arise.

This solves the same problem as pybind11's `registered_instances` map
and is sometimes called the Identity Map pattern. Two registries work
together to map a raw driver handle all the way back to the original
Python object. Both use weak references so they
do not prevent cleanup. Entries are removed either explicitly (via
`destroy()` or a Box destructor) or implicitly when the weak reference
expires.

## Level 1: Driver Handle -> Resource Handle (C++)

`HandleRegistry` in `_cpp/rt/internal.hpp` maps a raw CUDA handle
(e.g., `CUevent`, `CUkernel`, `CUgraph`) to a `weak_ptr`
for its owning resource handle. When a `_ref` constructor receives a raw
handle, it checks the registry first. If found, it returns the existing
`shared_ptr`, preserving the Box and its metadata (e.g., `EventBox`
carries timing/IPC flags, `KernelBox` carries the library dependency).

Without this level, a round-tripped handle would produce a new Box
with default metadata, losing information that was set at creation.

Instances: `context_registry`, `stream_registry`, `event_registry`,
`kernel_registry`, and, per `GraphBox`, one registry for its direct child
graphs and one for its nodes.

### Removing and replacing entries

More than one Box can exist for one raw handle, for example a plain `_ref`
Box and an owning one. An entry therefore records the address of the
element its handle points to, and `unregister_handle(key, resource)` erases
the entry only if `resource` matches. A Box destructor thus never erases the
entry of another, still-live Box registered under the same key.

`insert_or_get(key, candidate, replace)` registers a new Box without blindly
overwriting a live entry; `replace(existing)` decides. Build the candidate
before the call and drop a losing candidate after it, never under the
registry's lock, because its deleter takes that lock. `context_registry` uses
this to keep one owning Box per live context (#3533): a Box that holds a
primary-context retain or the owning green-context handle replaces a plain
one, and nothing replaces an owning one. `get_primary_context()` therefore
shares one retaining Box per primary context across all threads.

### Imports that wait for an in-flight release

IPC imports use `ReleasingHandleRegistry`, a variant whose entry outlives its
handle's last reference. Only the owner's deleter removes the entry, with
`release()`, after its last driver call on the resource. `get_or_create()`
waits while the entry under its key has expired, so a new owner never overlaps
the previous owner's cleanup, even when the last reference was dropped by a
dependent handle that the importing thread cannot synchronize with.

Instances: `ipc_pool_registry` maps an imported `CUmemoryPool` to its owner,
because every import of a pool in a process returns the same `CUmemoryPool`
(#3510); the owner's cleanup revokes access and destroys the imports.
`ipc_ptr_cache` maps a `CUmemPoolPtrExportData` descriptor to its imported
pointer, because a re-import before the previous mapping is freed would share
that mapping (nvbug 5570902).

### Reused graph handles

A `GraphHandle` weak reference locks whenever any handle of its hierarchy is
alive, so a lockable entry does not prove that a child graph is alive. CUDA
destroys child graphs inside node operations and capture invalidation, before
cuda.core retires their boxes. A process-wide map keyed by raw `CUgraph` could
therefore resolve a recycled address to another hierarchy's metadata.

Scoping the child registry to its parent `GraphBox` removes that case by
construction: `create_child_graph_handle` always knows the parent, and roots
are never looked up. Within the parent, an entry is keyed by the child's
`CUgraph` and its owner node, which together identify a child graph. If CUDA
destroys a child before cuda.core retires its box, the owner node is still
alive, so a newer child at the recycled address has a different owner and its
own entry. Retiring the stale box erases only the stale entry. Within one
owner, external synchronization and the retire-before-register order in
`graph_commit_child_graph_update` keep one live entry per key.

## Level 2: Resource Handle -> Python Object (Cython)

`_node_registry` in `_graph_node.pyx` is a `WeakValueDictionary`
mapping a resource address (`shared_ptr::get()`) to a Python
`GraphNode` object. When `GraphNode._create` receives a handle from
Level 1, it checks this registry. If found, it returns the existing
Python object.

Without this level, each driver round-trip would produce a distinct
Python object for the same logical node, resulting in surprising
behavior:

```python
a = g.empty()
a.succ = {b}
(b2,) = a.succ  # queries driver, gets back CUgraphNode for b
assert b2 is b  # fails without Level 2 registry
```
