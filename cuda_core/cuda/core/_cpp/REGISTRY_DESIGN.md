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

`HandleRegistry` in `resource_handles.cpp` maps a raw CUDA handle
(e.g., `CUevent`, `CUkernel`, `CUgraphNode`) to the `weak_ptr` that
owns it. When a `_ref` constructor receives a raw handle, it
checks the registry first. If found, it returns the existing
`shared_ptr`, preserving the Box and its metadata (e.g., `EventBox`
carries timing/IPC flags, `KernelBox` carries the library dependency).

Without this level, a round-tripped handle would produce a new Box
with default metadata, losing information that was set at creation.

Instances: `event_registry`, `kernel_registry`, `graph_node_registry`.

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
b2, = a.succ     # queries driver, gets back CUgraphNode for b
assert b2 is b   # fails without Level 2 registry
```
