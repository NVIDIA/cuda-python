# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Private module for explicit CUDA graph construction.

This module provides GraphDef and a Node class hierarchy for building CUDA
graphs explicitly (as opposed to stream capture). Both approaches produce
the same public Graph type for execution.

Node hierarchy:
    Node (base — also used for the internal entry point)
    ├── EmptyNode         (synchronization / join point)
    ├── KernelNode        (kernel launch)
    ├── AllocNode         (memory allocation, exposes dptr and bytesize)
    ├── FreeNode          (memory free, exposes dptr)
    ├── MemsetNode        (memory set, exposes dptr, value, element_size, etc.)
    ├── MemcpyNode        (memory copy, exposes dst, src, size)
    ├── ChildGraphNode    (embedded sub-graph)
    ├── EventRecordNode   (record an event)
    ├── EventWaitNode     (wait for an event)
    ├── HostCallbackNode  (host CPU callback)
    └── ConditionalNode   (conditional execution — base for reconstruction)
        ├── IfNode        (if-then conditional, 1 branch)
        ├── IfElseNode    (if-then-else conditional, 2 branches)
        ├── WhileNode     (while-loop conditional, 1 branch)
        └── SwitchNode    (switch conditional, N branches)
"""

from __future__ import annotations

from cpython.ref cimport Py_INCREF

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset as c_memset, memcpy as c_memcpy

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from cuda.core._event cimport Event
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._module cimport Kernel
from cuda.core._resource_handles cimport (
    EventHandle,
    GraphHandle,
    KernelHandle,
    GraphNodeHandle,
    as_cu,
    as_intptr,
    as_py,
    create_event_handle_ref,
    create_graph_handle,
    create_graph_handle_ref,
    create_kernel_handle_ref,
    create_graph_node_handle,
    graph_node_get_graph,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, _parse_fill_value

from dataclasses import dataclass

from cuda.core import Device
from cuda.core._utils.cuda_utils import driver, handle_return

__all__ = [
    "Condition",
    "GraphAllocOptions",
    "GraphDef",
    "Node",
    "EmptyNode",
    "KernelNode",
    "AllocNode",
    "FreeNode",
    "MemsetNode",
    "MemcpyNode",
    "ChildGraphNode",
    "EventRecordNode",
    "EventWaitNode",
    "HostCallbackNode",
    "ConditionalNode",
    "IfNode",
    "IfElseNode",
    "WhileNode",
    "SwitchNode",
]


cdef bint _has_cuGraphNodeGetParams = False
cdef bint _version_checked = False

cdef bint _check_node_get_params():
    global _has_cuGraphNodeGetParams, _version_checked
    if not _version_checked:
        ver = handle_return(driver.cuDriverGetVersion())
        _has_cuGraphNodeGetParams = ver >= 13020
        _version_checked = True
    return _has_cuGraphNodeGetParams


cdef extern from "Python.h":
    void _py_decref "Py_DECREF" (void*)


cdef void _py_host_trampoline(void* data) noexcept with gil:
    (<object>data)()


cdef void _py_host_destructor(void* data) noexcept with gil:
    _py_decref(data)


cdef void _destroy_event_handle_copy(void* ptr) noexcept nogil:
    cdef EventHandle* p = <EventHandle*>ptr
    del p


cdef void _destroy_kernel_handle_copy(void* ptr) noexcept nogil:
    cdef KernelHandle* p = <KernelHandle*>ptr
    del p


cdef void _attach_user_object(
        cydriver.CUgraph graph, void* ptr,
        cydriver.CUhostFn destroy) except *:
    """Create a CUDA user object and transfer ownership to the graph.

    On success the graph owns the resource (via MOVE semantics).
    On failure the destroy callback is invoked to clean up ptr,
    then a CUDAError is raised — callers need no try/except.
    """
    cdef cydriver.CUuserObject user_obj = NULL
    cdef cydriver.CUresult ret
    with nogil:
        ret = cydriver.cuUserObjectCreate(
            &user_obj, ptr, destroy, 1,
            cydriver.CU_USER_OBJECT_NO_DESTRUCTOR_SYNC)
        if ret == cydriver.CUDA_SUCCESS:
            ret = cydriver.cuGraphRetainUserObject(
                graph, user_obj, 1, cydriver.CU_GRAPH_USER_OBJECT_MOVE)
            if ret != cydriver.CUDA_SUCCESS:
                cydriver.cuUserObjectRelease(user_obj, 1)
    if ret != cydriver.CUDA_SUCCESS:
        if user_obj == NULL:
            destroy(ptr)
        HANDLE_RETURN(ret)


cdef class Condition:
    """Wraps a CUgraphConditionalHandle.

    Created by :meth:`GraphDef.create_condition` and passed to
    conditional-node builder methods (``if_cond``, ``if_else``,
    ``while_loop``, ``switch``). The underlying value is set at
    runtime by device code via ``cudaGraphSetConditional``.
    """

    def __repr__(self) -> str:
        return f"<Condition handle=0x{<unsigned long long>self._c_handle:x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self._c_handle == (<Condition>other)._c_handle

    def __hash__(self) -> int:
        return hash(<unsigned long long>self._c_handle)

    @property
    def handle(self) -> int:
        """The raw CUgraphConditionalHandle as an int."""
        return <unsigned long long>self._c_handle


cdef ConditionalNode _make_conditional_node(
        Node pred,
        Condition condition,
        cydriver.CUgraphConditionalNodeType cond_type,
        unsigned int size,
        type node_cls):
    if not isinstance(condition, Condition):
        raise TypeError(
            f"condition must be a Condition object (from "
            f"GraphDef.create_condition()), got {type(condition).__name__}")
    cdef cydriver.CUgraphNodeParams params
    cdef cydriver.CUgraphNode new_node = NULL

    c_memset(&params, 0, sizeof(params))
    params.type = cydriver.CU_GRAPH_NODE_TYPE_CONDITIONAL
    params.conditional.handle = condition._c_handle
    params.conditional.type = cond_type
    params.conditional.size = size

    cdef cydriver.CUcontext ctx = NULL
    cdef GraphHandle h_graph = graph_node_get_graph(pred._h_node)
    cdef cydriver.CUgraphNode pred_node = as_cu(pred._h_node)
    cdef cydriver.CUgraphNode* deps = NULL
    cdef size_t num_deps = 0

    if pred_node != NULL:
        deps = &pred_node
        num_deps = 1

    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    params.conditional.ctx = ctx

    with nogil:
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            HANDLE_RETURN(cydriver.cuGraphAddNode(
                &new_node, as_cu(h_graph), deps, NULL, num_deps, &params))
        ELSE:
            HANDLE_RETURN(cydriver.cuGraphAddNode(
                &new_node, as_cu(h_graph), deps, num_deps, &params))

    # cuGraphAddNode sets phGraph_out to an internal array of body
    # graphs (it replaces the pointer, not writing into a caller array).
    cdef list branch_list = []
    cdef unsigned int i
    cdef cydriver.CUgraph bg
    cdef GraphHandle h_branch
    for i in range(size):
        bg = params.conditional.phGraph_out[i]
        h_branch = create_graph_handle_ref(bg, h_graph)
        branch_list.append(GraphDef._from_handle(h_branch))
    cdef tuple branches = tuple(branch_list)

    cdef ConditionalNode n = node_cls.__new__(node_cls)
    n._h_node = create_graph_node_handle(new_node, h_graph)
    n._condition = condition
    n._cond_type = cond_type
    n._branches = branches

    pred._succ_cache = None
    return n


@dataclass
class GraphAllocOptions:
    """Options for graph memory allocation nodes.

    Attributes
    ----------
    device : int or Device, optional
        The device on which to allocate memory. If None (default),
        uses the current CUDA context's device.
    memory_type : str, optional
        Type of memory to allocate. One of:

        - ``"device"`` (default): Pinned device memory, optimal for GPU kernels.
        - ``"host"``: Pinned host memory, accessible from both host and device.
          Useful for graphs containing host callback nodes. Note: may not be
          supported on all systems/drivers.
        - ``"managed"``: Managed/unified memory that automatically migrates
          between host and device. Useful for mixed host/device access patterns.

    peer_access : list of int or Device, optional
        List of devices that should have read-write access to the
        allocated memory. If None (default), only the allocating
        device has access.

    Notes
    -----
    - IPC (inter-process communication) is not supported for graph
      memory allocation nodes per CUDA documentation.
    - The allocation uses the device's default memory pool.
    """

    device: int | Device | None = None
    memory_type: str = "device"
    peer_access: list | None = None


cdef class GraphDef:
    """Represents a CUDA graph definition (CUgraph).

    A GraphDef is used to construct a graph explicitly by adding nodes
    and specifying dependencies. Once construction is complete, call
    instantiate() to obtain an executable Graph.
    """

    def __init__(self):
        """Create a new empty graph definition."""
        cdef cydriver.CUgraph graph = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphCreate(&graph, 0))
        self._h_graph = create_graph_handle(graph)

    @staticmethod
    cdef GraphDef _from_handle(GraphHandle h_graph):
        """Create a GraphDef from an existing GraphHandle (internal use)."""
        cdef GraphDef g = GraphDef.__new__(GraphDef)
        g._h_graph = h_graph
        return g

    def __repr__(self) -> str:
        return f"<GraphDef handle=0x{as_intptr(self._h_graph):x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphDef):
            return NotImplemented
        return as_intptr(self._h_graph) == as_intptr((<GraphDef>other)._h_graph)

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_graph))

    @property
    def _entry(self) -> Node:
        """Return the internal entry-point Node (no dependencies)."""
        cdef Node n = Node.__new__(Node)
        n._h_node = create_graph_node_handle(<cydriver.CUgraphNode>NULL, self._h_graph)
        return n

    def alloc(self, size_t size, options: GraphAllocOptions | None = None) -> AllocNode:
        """Add an entry-point memory allocation node (no dependencies).

        See :meth:`Node.alloc` for full documentation.
        """
        return self._entry.alloc(size, options)

    def free(self, dptr) -> FreeNode:
        """Add an entry-point memory free node (no dependencies).

        See :meth:`Node.free` for full documentation.
        """
        return self._entry.free(dptr)

    def memset(self, dst, value, size_t width, size_t height=1, size_t pitch=0) -> MemsetNode:
        """Add an entry-point memset node (no dependencies).

        See :meth:`Node.memset` for full documentation.
        """
        return self._entry.memset(dst, value, width, height, pitch)

    def launch(self, config, kernel, *args) -> KernelNode:
        """Add an entry-point kernel launch node (no dependencies).

        See :meth:`Node.launch` for full documentation.
        """
        return self._entry.launch(config, kernel, *args)

    def join(self, *nodes) -> EmptyNode:
        """Create an empty node that depends on all given nodes.

        Parameters
        ----------
        *nodes : Node
            Nodes to merge.

        Returns
        -------
        EmptyNode
            A new EmptyNode that depends on all input nodes.
        """
        return self._entry.join(*nodes)

    def memcpy(self, dst, src, size_t size) -> MemcpyNode:
        """Add an entry-point memcpy node (no dependencies).

        See :meth:`Node.memcpy` for full documentation.
        """
        return self._entry.memcpy(dst, src, size)

    def embed(self, child: GraphDef) -> ChildGraphNode:
        """Add an entry-point child graph node (no dependencies).

        See :meth:`Node.embed` for full documentation.
        """
        return self._entry.embed(child)

    def record_event(self, event: Event) -> EventRecordNode:
        """Add an entry-point event record node (no dependencies).

        See :meth:`Node.record_event` for full documentation.
        """
        return self._entry.record_event(event)

    def wait_event(self, event: Event) -> EventWaitNode:
        """Add an entry-point event wait node (no dependencies).

        See :meth:`Node.wait_event` for full documentation.
        """
        return self._entry.wait_event(event)

    def callback(self, fn, *, user_data=None) -> HostCallbackNode:
        """Add an entry-point host callback node (no dependencies).

        See :meth:`Node.callback` for full documentation.
        """
        return self._entry.callback(fn, user_data=user_data)

    def create_condition(self, default_value: int | None = None) -> Condition:
        """Create a condition variable for use with conditional nodes.

        The returned :class:`Condition` object is passed to conditional-node
        builder methods. Its value is controlled at runtime by device code
        via ``cudaGraphSetConditional``.

        Parameters
        ----------
        default_value : int, optional
            The default value to assign to the condition.
            If None, no default is assigned.

        Returns
        -------
        Condition
            A condition variable for controlling conditional execution.
        """
        cdef cydriver.CUgraphConditionalHandle c_handle
        cdef unsigned int flags = 0
        cdef unsigned int default_val = 0

        if default_value is not None:
            default_val = <unsigned int>default_value
            flags = cydriver.CU_GRAPH_COND_ASSIGN_DEFAULT

        cdef cydriver.CUcontext ctx = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
            HANDLE_RETURN(cydriver.cuGraphConditionalHandleCreate(
                &c_handle, as_cu(self._h_graph), ctx, default_val, flags))

        cdef Condition cond = Condition.__new__(Condition)
        cond._c_handle = c_handle
        return cond

    def if_cond(self, condition: Condition) -> IfNode:
        """Add an entry-point if-conditional node (no dependencies).

        See :meth:`Node.if_cond` for full documentation.
        """
        return self._entry.if_cond(condition)

    def if_else(self, condition: Condition) -> IfElseNode:
        """Add an entry-point if-else conditional node (no dependencies).

        See :meth:`Node.if_else` for full documentation.
        """
        return self._entry.if_else(condition)

    def while_loop(self, condition: Condition) -> WhileNode:
        """Add an entry-point while-loop conditional node (no dependencies).

        See :meth:`Node.while_loop` for full documentation.
        """
        return self._entry.while_loop(condition)

    def switch(self, condition: Condition, unsigned int count) -> SwitchNode:
        """Add an entry-point switch conditional node (no dependencies).

        See :meth:`Node.switch` for full documentation.
        """
        return self._entry.switch(condition, count)

    def instantiate(self):
        """Instantiate the graph definition into an executable Graph.

        Returns
        -------
        Graph
            An executable graph that can be launched on a stream.
        """
        from cuda.core._graph import Graph
        from cuda.core._utils.cuda_utils import handle_return

        graph_exec = handle_return(driver.cuGraphInstantiate(
            driver.CUgraph(as_intptr(self._h_graph)), 0))
        return Graph._init(graph_exec)

    def debug_dot_print(self, path: str, options=None) -> None:
        """Write a GraphViz DOT representation of the graph to a file.

        Parameters
        ----------
        path : str
            File path for the DOT output.
        options : GraphDebugPrintOptions, optional
            Customizable options for the debug print.
        """
        from cuda.core._graph import GraphDebugPrintOptions

        cdef unsigned int flags = 0
        if options is not None:
            if not isinstance(options, GraphDebugPrintOptions):
                raise TypeError("options must be a GraphDebugPrintOptions instance")
            flags = options._to_flags()

        cdef bytes path_bytes = path.encode('utf-8')
        cdef const char* c_path = path_bytes
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphDebugDotPrint(as_cu(self._h_graph), c_path, flags))

    def nodes(self) -> tuple:
        """Return all nodes in the graph.

        Returns
        -------
        tuple of Node
            All nodes in the graph.
        """
        cdef size_t num_nodes = 0

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(as_cu(self._h_graph), NULL, &num_nodes))

        if num_nodes == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] nodes_vec
        nodes_vec.resize(num_nodes)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(as_cu(self._h_graph), nodes_vec.data(), &num_nodes))

        return tuple(Node._create(self._h_graph, nodes_vec[i]) for i in range(num_nodes))

    def edges(self) -> tuple:
        """Return all edges in the graph as (from_node, to_node) pairs.

        Returns
        -------
        tuple of tuple
            Each element is a (from_node, to_node) pair representing
            a dependency edge in the graph.
        """
        cdef size_t num_edges = 0

        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(as_cu(self._h_graph), NULL, NULL, NULL, &num_edges))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(as_cu(self._h_graph), NULL, NULL, &num_edges))

        if num_edges == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] from_nodes
        cdef vector[cydriver.CUgraphNode] to_nodes
        from_nodes.resize(num_edges)
        to_nodes.resize(num_edges)
        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(
                    as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), NULL, &num_edges))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(
                    as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), &num_edges))

        return tuple(
            (Node._create(self._h_graph, from_nodes[i]),
             Node._create(self._h_graph, to_nodes[i]))
            for i in range(num_edges)
        )

    @property
    def handle(self) -> int:
        """Return the underlying CUgraph handle."""
        return as_py(self._h_graph)


cdef class Node:
    """Base class for all graph nodes.

    Nodes are created by calling builder methods on GraphDef (for
    entry-point nodes with no dependencies) or on other Nodes (for
    nodes that depend on a predecessor).
    """

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Factory: dispatch to the right subclass based on node type."""
        if node == NULL:
            n = Node.__new__(Node)
            (<Node>n)._h_node = create_graph_node_handle(node, h_graph)
            return n

        cdef GraphNodeHandle h_node = create_graph_node_handle(node, h_graph)
        cdef cydriver.CUgraphNodeType node_type
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetType(node, &node_type))

        if node_type == cydriver.CU_GRAPH_NODE_TYPE_EMPTY:
            return EmptyNode._create_impl(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_KERNEL:
            return KernelNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEM_ALLOC:
            return AllocNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEM_FREE:
            return FreeNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEMSET:
            return MemsetNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEMCPY:
            return MemcpyNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_GRAPH:
            return ChildGraphNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            return EventRecordNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_WAIT_EVENT:
            return EventWaitNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_HOST:
            return HostCallbackNode._create_from_driver(h_node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_CONDITIONAL:
            return ConditionalNode._create_from_driver(h_node)
        else:
            n = Node.__new__(Node)
            (<Node>n)._h_node = h_node
            return n

    def __repr__(self) -> str:
        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        if node == NULL:
            return "<Node entry>"
        return f"<Node handle=0x{<uintptr_t>node:x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        cdef Node o = <Node>other
        return as_intptr(self._h_node) == as_intptr(o._h_node)

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_node))

    @property
    def type(self):
        """Return the CUDA graph node type.

        Returns
        -------
        CUgraphNodeType or None
            The node type enum value, or None for the entry node.
        """
        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        if node == NULL:
            return None
        cdef cydriver.CUgraphNodeType node_type
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetType(node, &node_type))
        return driver.CUgraphNodeType(<int>node_type)

    @property
    def graph(self) -> GraphDef:
        """Return the GraphDef this node belongs to."""
        return GraphDef._from_handle(graph_node_get_graph(self._h_node))

    @property
    def handle(self) -> int | None:
        """Return the underlying CUgraphNode handle as an int.

        Returns None for the entry node.
        """
        return as_py(self._h_node)

    @property
    def pred(self) -> tuple:
        """Return the predecessor nodes (dependencies) of this node.

        Results are cached since a node's dependencies are immutable
        once created.

        Returns
        -------
        tuple of Node
            The nodes that this node depends on.
        """
        if self._pred_cache is not None:
            return self._pred_cache

        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        if node == NULL:
            self._pred_cache = ()
            return self._pred_cache

        cdef size_t num_deps = 0

        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, NULL, NULL, &num_deps))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, NULL, &num_deps))

        if num_deps == 0:
            self._pred_cache = ()
            return self._pred_cache

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, deps.data(), NULL, &num_deps))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, deps.data(), &num_deps))

        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        self._pred_cache = tuple(Node._create(h_graph, deps[i]) for i in range(num_deps))
        return self._pred_cache

    @property
    def succ(self) -> tuple:
        """Return the successor nodes (dependents) of this node.

        Results are cached and automatically invalidated when new
        dependent nodes are added via builder methods.

        Returns
        -------
        tuple of Node
            The nodes that depend on this node.
        """
        if self._succ_cache is not None:
            return self._succ_cache

        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        if node == NULL:
            self._succ_cache = ()
            return self._succ_cache

        cdef size_t num_deps = 0

        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, NULL, NULL, &num_deps))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, NULL, &num_deps))

        if num_deps == 0:
            self._succ_cache = ()
            return self._succ_cache

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, deps.data(), NULL, &num_deps))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, deps.data(), &num_deps))

        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        self._succ_cache = tuple(Node._create(h_graph, deps[i]) for i in range(num_deps))
        return self._succ_cache

    def launch(self, config: LaunchConfig, kernel: Kernel, *args) -> KernelNode:
        """Add a kernel launch node depending on this node.

        Parameters
        ----------
        config : LaunchConfig
            Launch configuration (grid, block, shared memory, etc.)
        kernel : Kernel
            The kernel to launch.
        *args
            Kernel arguments.

        Returns
        -------
        KernelNode
            A new KernelNode representing the kernel launch.
        """
        cdef LaunchConfig conf = config
        cdef Kernel ker = <Kernel>kernel
        cdef ParamHolder ker_args = ParamHolder(args)

        cdef cydriver.CUDA_KERNEL_NODE_PARAMS node_params
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        node_params.kern = as_cu(ker._h_kernel)
        node_params.func = <cydriver.CUfunction>NULL
        node_params.gridDimX = conf.grid[0]
        node_params.gridDimY = conf.grid[1]
        node_params.gridDimZ = conf.grid[2]
        node_params.blockDimX = conf.block[0]
        node_params.blockDimY = conf.block[1]
        node_params.blockDimZ = conf.block[2]
        node_params.sharedMemBytes = conf.shmem_size
        node_params.kernelParams = <void**><uintptr_t>(ker_args.ptr)
        node_params.extra = NULL
        node_params.ctx = <cydriver.CUcontext>NULL

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddKernelNode(
                &new_node, as_cu(h_graph), deps, num_deps, &node_params))

        _attach_user_object(as_cu(h_graph), <void*>new KernelHandle(ker._h_kernel),
                            <cydriver.CUhostFn>_destroy_kernel_handle_copy)

        self._succ_cache = None
        return KernelNode._create_with_params(
            create_graph_node_handle(new_node, h_graph),
            conf.grid, conf.block, conf.shmem_size,
            ker._h_kernel)

    def join(self, *nodes: Node) -> EmptyNode:
        """Create an empty node that depends on this node and all given nodes.

        This is used to synchronize multiple branches of execution.

        Parameters
        ----------
        *nodes : Node
            Additional nodes to depend on.

        Returns
        -------
        EmptyNode
            A new EmptyNode that depends on all input nodes.
        """
        cdef vector[cydriver.CUgraphNode] deps
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef Node other
        cdef cydriver.CUgraphNode* deps_ptr = NULL
        cdef size_t num_deps = 0
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)

        if pred_node != NULL:
            deps.push_back(pred_node)
        for other in nodes:
            if as_cu((<Node>other)._h_node) != NULL:
                deps.push_back(as_cu((<Node>other)._h_node))

        num_deps = deps.size()
        if num_deps > 0:
            deps_ptr = deps.data()

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEmptyNode(
                &new_node, as_cu(h_graph), deps_ptr, num_deps))

        self._succ_cache = None
        for other in nodes:
            (<Node>other)._succ_cache = None
        return EmptyNode._create_impl(create_graph_node_handle(new_node, h_graph))

    def alloc(self, size_t size, options: GraphAllocOptions | None = None) -> AllocNode:
        """Add a memory allocation node depending on this node.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.
        options : GraphAllocOptions, optional
            Allocation options. If None, allocates on the current device.

        Returns
        -------
        AllocNode
            A new AllocNode representing the allocation. Access the allocated
            device pointer via the dptr property.
        """
        cdef int device_id
        cdef cydriver.CUdevice dev

        if options is None or options.device is None:
            with nogil:
                HANDLE_RETURN(cydriver.cuCtxGetDevice(&dev))
            device_id = <int>dev
        else:
            device_id = getattr(options.device, 'device_id', options.device)

        cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS alloc_params
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        cdef vector[cydriver.CUmemAccessDesc] access_descs
        cdef int peer_id
        cdef list peer_ids = []

        if options is not None and options.peer_access is not None:
            for peer_dev in options.peer_access:
                peer_id = getattr(peer_dev, 'device_id', peer_dev)
                peer_ids.append(peer_id)
                access_descs.push_back(cydriver.CUmemAccessDesc_st(
                    cydriver.CUmemLocation_st(
                        cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                        peer_id
                    ),
                    cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                ))

        cdef str memory_type = "device"
        if options is not None and options.memory_type is not None:
            memory_type = options.memory_type

        c_memset(&alloc_params, 0, sizeof(alloc_params))
        alloc_params.poolProps.handleTypes = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        alloc_params.bytesize = size

        if memory_type == "device":
            alloc_params.poolProps.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            alloc_params.poolProps.location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            alloc_params.poolProps.location.id = device_id
        elif memory_type == "host":
            alloc_params.poolProps.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            alloc_params.poolProps.location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            alloc_params.poolProps.location.id = 0
        elif memory_type == "managed":
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                alloc_params.poolProps.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED
                alloc_params.poolProps.location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                alloc_params.poolProps.location.id = device_id
            ELSE:
                raise ValueError("memory_type='managed' requires CUDA 13.0 or later")
        else:
            raise ValueError(f"Invalid memory_type: {memory_type!r}. "
                           "Must be 'device', 'host', or 'managed'.")

        if access_descs.size() > 0:
            alloc_params.accessDescs = access_descs.data()
            alloc_params.accessDescCount = access_descs.size()

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemAllocNode(
                &new_node, as_cu(h_graph), deps, num_deps, &alloc_params))

        self._succ_cache = None
        return AllocNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), alloc_params.dptr, size,
            device_id, memory_type, tuple(peer_ids))

    def free(self, dptr: int) -> FreeNode:
        """Add a memory free node depending on this node.

        Parameters
        ----------
        dptr : int
            Device pointer to free (typically from AllocNode.dptr).

        Returns
        -------
        FreeNode
            A new FreeNode representing the free operation.
        """
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0
        cdef cydriver.CUdeviceptr c_dptr = <cydriver.CUdeviceptr>dptr

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemFreeNode(
                &new_node, as_cu(h_graph), deps, num_deps, c_dptr))

        self._succ_cache = None
        return FreeNode._create_with_params(create_graph_node_handle(new_node, h_graph), c_dptr)

    def memset(self, dst: int, value, size_t width, size_t height=1, size_t pitch=0) -> MemsetNode:
        """Add a memset node depending on this node.

        Parameters
        ----------
        dst : int
            Destination device pointer.
        value : int or buffer-protocol object
            Fill value. int for 1-byte fill (range [0, 256)),
            or buffer-protocol object of 1, 2, or 4 bytes.
        width : int
            Width of the row in elements.
        height : int, optional
            Number of rows (default 1).
        pitch : int, optional
            Pitch of destination in bytes (default 0, unused if height is 1).

        Returns
        -------
        MemsetNode
            A new MemsetNode representing the memset operation.
        """
        cdef unsigned int val
        cdef unsigned int elem_size
        val, elem_size = _parse_fill_value(value)

        cdef cydriver.CUDA_MEMSET_NODE_PARAMS memset_params
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        cdef cydriver.CUdeviceptr c_dst = <cydriver.CUdeviceptr>dst
        cdef cydriver.CUcontext ctx = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))

        c_memset(&memset_params, 0, sizeof(memset_params))
        memset_params.dst = c_dst
        memset_params.value = val
        memset_params.elementSize = elem_size
        memset_params.width = width
        memset_params.height = height
        memset_params.pitch = pitch

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemsetNode(
                &new_node, as_cu(h_graph), deps, num_deps,
                &memset_params, ctx))

        self._succ_cache = None
        return MemsetNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), c_dst,
            val, elem_size, width, height, pitch)

    def memcpy(self, dst: int, src: int, size_t size) -> MemcpyNode:
        """Add a memcpy node depending on this node.

        Copies ``size`` bytes from ``src`` to ``dst``. Memory types are
        auto-detected via the driver, so both device and pinned host
        pointers are supported.

        Parameters
        ----------
        dst : int
            Destination pointer (device or pinned host).
        src : int
            Source pointer (device or pinned host).
        size : int
            Number of bytes to copy.

        Returns
        -------
        MemcpyNode
            A new MemcpyNode representing the copy operation.
        """
        cdef cydriver.CUdeviceptr c_dst = <cydriver.CUdeviceptr>dst
        cdef cydriver.CUdeviceptr c_src = <cydriver.CUdeviceptr>src

        cdef unsigned int dst_mem_type = cydriver.CU_MEMORYTYPE_DEVICE
        cdef unsigned int src_mem_type = cydriver.CU_MEMORYTYPE_DEVICE
        cdef cydriver.CUresult ret
        with nogil:
            ret = cydriver.cuPointerGetAttribute(
                &dst_mem_type,
                cydriver.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                c_dst)
            if ret != cydriver.CUDA_SUCCESS and ret != cydriver.CUDA_ERROR_INVALID_VALUE:
                HANDLE_RETURN(ret)
            ret = cydriver.cuPointerGetAttribute(
                &src_mem_type,
                cydriver.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                c_src)
            if ret != cydriver.CUDA_SUCCESS and ret != cydriver.CUDA_ERROR_INVALID_VALUE:
                HANDLE_RETURN(ret)

        cdef cydriver.CUmemorytype c_dst_type = <cydriver.CUmemorytype>dst_mem_type
        cdef cydriver.CUmemorytype c_src_type = <cydriver.CUmemorytype>src_mem_type

        cdef cydriver.CUDA_MEMCPY3D params
        c_memset(&params, 0, sizeof(params))

        params.srcMemoryType = c_src_type
        params.dstMemoryType = c_dst_type
        if c_src_type == cydriver.CU_MEMORYTYPE_HOST:
            params.srcHost = <const void*><uintptr_t>c_src
        else:
            params.srcDevice = c_src
        if c_dst_type == cydriver.CU_MEMORYTYPE_HOST:
            params.dstHost = <void*><uintptr_t>c_dst
        else:
            params.dstDevice = c_dst
        params.WidthInBytes = size
        params.Height = 1
        params.Depth = 1

        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        cdef cydriver.CUcontext ctx = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
            HANDLE_RETURN(cydriver.cuGraphAddMemcpyNode(
                &new_node, as_cu(h_graph), deps, num_deps, &params, ctx))

        self._succ_cache = None
        return MemcpyNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), c_dst, c_src, size,
            c_dst_type, c_src_type)

    def embed(self, child: GraphDef) -> ChildGraphNode:
        """Add a child graph node depending on this node.

        Embeds a clone of the given graph definition as a sub-graph node.
        The child graph must not contain allocation, free, or conditional
        nodes.

        Parameters
        ----------
        child : GraphDef
            The graph definition to embed (will be cloned).

        Returns
        -------
        ChildGraphNode
            A new ChildGraphNode representing the embedded sub-graph.
        """
        cdef GraphDef child_def = <GraphDef>child
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddChildGraphNode(
                &new_node, as_cu(h_graph), deps, num_deps, as_cu(child_def._h_graph)))

        cdef cydriver.CUgraph embedded_graph = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphChildGraphNodeGetGraph(
                new_node, &embedded_graph))

        cdef GraphHandle h_embedded = create_graph_handle_ref(embedded_graph, h_graph)

        self._succ_cache = None
        return ChildGraphNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), h_embedded)

    def record_event(self, event: Event) -> EventRecordNode:
        """Add an event record node depending on this node.

        Parameters
        ----------
        event : Event
            The event to record.

        Returns
        -------
        EventRecordNode
            A new EventRecordNode representing the event record operation.
        """
        cdef Event ev = <Event>event
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEventRecordNode(
                &new_node, as_cu(h_graph), deps, num_deps, as_cu(ev._h_event)))

        _attach_user_object(as_cu(h_graph), <void*>new EventHandle(ev._h_event),
                            <cydriver.CUhostFn>_destroy_event_handle_copy)

        self._succ_cache = None
        return EventRecordNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), ev._h_event)

    def wait_event(self, event: Event) -> EventWaitNode:
        """Add an event wait node depending on this node.

        Parameters
        ----------
        event : Event
            The event to wait for.

        Returns
        -------
        EventWaitNode
            A new EventWaitNode representing the event wait operation.
        """
        cdef Event ev = <Event>event
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEventWaitNode(
                &new_node, as_cu(h_graph), deps, num_deps, as_cu(ev._h_event)))

        _attach_user_object(as_cu(h_graph), <void*>new EventHandle(ev._h_event),
                            <cydriver.CUhostFn>_destroy_event_handle_copy)

        self._succ_cache = None
        return EventWaitNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), ev._h_event)

    def callback(self, fn, *, user_data=None) -> HostCallbackNode:
        """Add a host callback node depending on this node.

        The callback runs on the host CPU when the graph reaches this node.
        Two modes are supported:

        - **Python callable**: Pass any callable. The GIL is acquired
          automatically. The callable must take no arguments; use closures
          or ``functools.partial`` to bind state.
        - **ctypes function pointer**: Pass a ``ctypes.CFUNCTYPE`` instance.
          The function receives a single ``void*`` argument (the
          ``user_data``). The caller must keep the ctypes wrapper alive
          for the lifetime of the graph.

        .. warning::

            Callbacks must not call CUDA API functions. Doing so may
            deadlock or corrupt driver state.

        Parameters
        ----------
        fn : callable or ctypes function pointer
            The callback function.
        user_data : int or bytes-like, optional
            Only for ctypes function pointers. If ``int``, passed as a raw
            pointer (caller manages lifetime). If bytes-like, the data is
            copied and its lifetime is tied to the graph.

        Returns
        -------
        HostCallbackNode
            A new HostCallbackNode representing the callback.
        """
        import ctypes as ct

        cdef cydriver.CUDA_HOST_NODE_PARAMS node_params
        cdef cydriver.CUgraphNode new_node = NULL
        cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
        cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0
        cdef void* c_user_data = NULL
        cdef object callable_obj = None
        cdef void* fn_pyobj = NULL

        if pred_node != NULL:
            deps = &pred_node
            num_deps = 1

        if isinstance(fn, ct._CFuncPtr):
            node_params.fn = <cydriver.CUhostFn><uintptr_t>ct.cast(
                fn, ct.c_void_p).value

            if user_data is not None:
                if isinstance(user_data, int):
                    c_user_data = <void*><uintptr_t>user_data
                else:
                    buf = bytes(user_data)
                    c_user_data = malloc(len(buf))
                    if c_user_data == NULL:
                        raise MemoryError(
                            "failed to allocate user_data buffer")
                    c_memcpy(c_user_data, <const char*>buf, len(buf))
                    _attach_user_object(
                        as_cu(h_graph), c_user_data,
                        <cydriver.CUhostFn>free)

            node_params.userData = c_user_data
        else:
            if user_data is not None:
                raise ValueError(
                    "user_data is only supported with ctypes "
                    "function pointers")
            callable_obj = fn
            Py_INCREF(fn)
            fn_pyobj = <void*>fn
            node_params.fn = <cydriver.CUhostFn>_py_host_trampoline
            node_params.userData = fn_pyobj
            _attach_user_object(
                as_cu(h_graph), fn_pyobj,
                <cydriver.CUhostFn>_py_host_destructor)

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddHostNode(
                &new_node, as_cu(h_graph), deps, num_deps, &node_params))

        self._succ_cache = None
        return HostCallbackNode._create_with_params(
            create_graph_node_handle(new_node, h_graph), callable_obj,
            node_params.fn, node_params.userData)

    def if_cond(self, condition: Condition) -> IfNode:
        """Add an if-conditional node depending on this node.

        The body graph executes only when the condition evaluates to
        a non-zero value at runtime.

        Parameters
        ----------
        condition : Condition
            Condition from :meth:`GraphDef.create_condition`.

        Returns
        -------
        IfNode
            A new IfNode with one branch accessible via ``.then``.
        """
        return _make_conditional_node(
            self, condition,
            cydriver.CU_GRAPH_COND_TYPE_IF, 1, IfNode)

    def if_else(self, condition: Condition) -> IfElseNode:
        """Add an if-else conditional node depending on this node.

        Two body graphs: the first executes when the condition is
        non-zero, the second when it is zero.

        Parameters
        ----------
        condition : Condition
            Condition from :meth:`GraphDef.create_condition`.

        Returns
        -------
        IfElseNode
            A new IfElseNode with branches accessible via
            ``.then`` and ``.else_``.
        """
        return _make_conditional_node(
            self, condition,
            cydriver.CU_GRAPH_COND_TYPE_IF, 2, IfElseNode)

    def while_loop(self, condition: Condition) -> WhileNode:
        """Add a while-loop conditional node depending on this node.

        The body graph executes repeatedly while the condition
        evaluates to a non-zero value.

        Parameters
        ----------
        condition : Condition
            Condition from :meth:`GraphDef.create_condition`.

        Returns
        -------
        WhileNode
            A new WhileNode with body accessible via ``.body``.
        """
        return _make_conditional_node(
            self, condition,
            cydriver.CU_GRAPH_COND_TYPE_WHILE, 1, WhileNode)

    def switch(self, condition: Condition, unsigned int count) -> SwitchNode:
        """Add a switch conditional node depending on this node.

        The condition value selects which branch to execute. If the
        value is out of range, no branch executes.

        Parameters
        ----------
        condition : Condition
            Condition from :meth:`GraphDef.create_condition`.
        count : int
            Number of switch cases (branches).

        Returns
        -------
        SwitchNode
            A new SwitchNode with branches accessible via ``.branches``.
        """
        return _make_conditional_node(
            self, condition,
            cydriver.CU_GRAPH_COND_TYPE_SWITCH, count, SwitchNode)


# =============================================================================
# Node subclasses
# =============================================================================


cdef class EmptyNode(Node):
    """A synchronization / join node with no operation."""

    @staticmethod
    cdef EmptyNode _create_impl(GraphNodeHandle h_node):
        cdef EmptyNode n = EmptyNode.__new__(EmptyNode)
        n._h_node = h_node
        return n

    def __repr__(self) -> str:
        cdef Py_ssize_t n = len(self.pred)
        return f"<EmptyNode with {n} {'pred' if n == 1 else 'preds'}>"


cdef class KernelNode(Node):
    """A kernel launch node.

    Properties
    ----------
    grid : tuple of int
        Grid dimensions (gridDimX, gridDimY, gridDimZ).
    block : tuple of int
        Block dimensions (blockDimX, blockDimY, blockDimZ).
    shmem_size : int
        Dynamic shared memory size in bytes.
    kernel : Kernel
        The kernel object for this launch node.
    config : LaunchConfig
        A LaunchConfig reconstructed from this node's parameters.
    """

    @staticmethod
    cdef KernelNode _create_with_params(GraphNodeHandle h_node,
                                        tuple grid, tuple block, unsigned int shmem_size,
                                        KernelHandle h_kernel):
        """Create from known params (called by launch() builder)."""
        cdef KernelNode n = KernelNode.__new__(KernelNode)
        n._h_node = h_node
        n._grid = grid
        n._block = block
        n._shmem_size = shmem_size
        n._h_kernel = h_kernel
        return n

    @staticmethod
    cdef KernelNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUDA_KERNEL_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphKernelNodeGetParams(node, &params))
        cdef KernelHandle h_kernel = create_kernel_handle_ref(params.kern)
        return KernelNode._create_with_params(
            h_node,
            (params.gridDimX, params.gridDimY, params.gridDimZ),
            (params.blockDimX, params.blockDimY, params.blockDimZ),
            params.sharedMemBytes,
            h_kernel)

    def __repr__(self) -> str:
        return (f"<KernelNode grid={self._grid} block={self._block}>")

    @property
    def grid(self) -> tuple:
        """Grid dimensions as a 3-tuple (gridDimX, gridDimY, gridDimZ)."""
        return self._grid

    @property
    def block(self) -> tuple:
        """Block dimensions as a 3-tuple (blockDimX, blockDimY, blockDimZ)."""
        return self._block

    @property
    def shmem_size(self) -> int:
        """Dynamic shared memory size in bytes."""
        return self._shmem_size

    @property
    def kernel(self) -> Kernel:
        """The Kernel object for this launch node."""
        return Kernel._from_handle(self._h_kernel)

    @property
    def config(self) -> LaunchConfig:
        """A LaunchConfig reconstructed from this node's grid, block, and shmem_size.

        Note: cluster dimensions and cooperative_launch are not preserved
        by the CUDA driver's kernel node params, so they are not included.
        """
        return LaunchConfig(grid=self._grid, block=self._block,
                            shmem_size=self._shmem_size)


cdef class AllocNode(Node):
    """A memory allocation node.

    Properties
    ----------
    dptr : int
        The device pointer for the allocation.
    bytesize : int
        The number of bytes allocated.
    device_id : int
        The device on which the allocation was made.
    memory_type : str
        The type of memory allocated (``"device"``, ``"host"``, or ``"managed"``).
    peer_access : tuple of int
        Device IDs that have read-write access to this allocation.
    options : GraphAllocOptions
        A GraphAllocOptions reconstructed from this node's parameters.
    """

    @staticmethod
    cdef AllocNode _create_with_params(GraphNodeHandle h_node,
                                       cydriver.CUdeviceptr dptr, size_t bytesize,
                                       int device_id, str memory_type, tuple peer_access):
        """Create from known params (called by alloc() builder)."""
        cdef AllocNode n = AllocNode.__new__(AllocNode)
        n._h_node = h_node
        n._dptr = dptr
        n._bytesize = bytesize
        n._device_id = device_id
        n._memory_type = memory_type
        n._peer_access = peer_access
        return n

    @staticmethod
    cdef AllocNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemAllocNodeGetParams(node, &params))

        cdef str memory_type
        if params.poolProps.allocType == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED:
            if params.poolProps.location.type == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST:
                memory_type = "host"
            else:
                memory_type = "device"
        else:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                if params.poolProps.allocType == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED:
                    memory_type = "managed"
                else:
                    memory_type = "device"
            ELSE:
                memory_type = "device"

        cdef list peer_ids = []
        cdef size_t i
        for i in range(params.accessDescCount):
            peer_ids.append(<int>params.accessDescs[i].location.id)

        return AllocNode._create_with_params(
            h_node, params.dptr, params.bytesize,
            <int>params.poolProps.location.id, memory_type, tuple(peer_ids))

    def __repr__(self) -> str:
        return f"<AllocNode dptr=0x{self._dptr:x} size={self._bytesize}>"

    @property
    def dptr(self) -> int:
        """The device pointer for the allocation."""
        return self._dptr

    @property
    def bytesize(self) -> int:
        """The number of bytes allocated."""
        return self._bytesize

    @property
    def device_id(self) -> int:
        """The device on which the allocation was made."""
        return self._device_id

    @property
    def memory_type(self) -> str:
        """The type of memory: ``"device"``, ``"host"``, or ``"managed"``."""
        return self._memory_type

    @property
    def peer_access(self) -> tuple:
        """Device IDs with read-write access to this allocation."""
        return self._peer_access

    @property
    def options(self) -> GraphAllocOptions:
        """A GraphAllocOptions reconstructed from this node's parameters."""
        return GraphAllocOptions(
            device=self._device_id,
            memory_type=self._memory_type,
            peer_access=list(self._peer_access) if self._peer_access else None,
        )


cdef class FreeNode(Node):
    """A memory free node.

    Properties
    ----------
    dptr : int
        The device pointer being freed.
    """

    @staticmethod
    cdef FreeNode _create_with_params(GraphNodeHandle h_node,
                                      cydriver.CUdeviceptr dptr):
        """Create from known params (called by free() builder)."""
        cdef FreeNode n = FreeNode.__new__(FreeNode)
        n._h_node = h_node
        n._dptr = dptr
        return n

    @staticmethod
    cdef FreeNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUdeviceptr dptr
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemFreeNodeGetParams(node, &dptr))
        return FreeNode._create_with_params(h_node, dptr)

    def __repr__(self) -> str:
        return f"<FreeNode dptr=0x{self._dptr:x}>"

    @property
    def dptr(self) -> int:
        """The device pointer being freed."""
        return self._dptr


cdef class MemsetNode(Node):
    """A memory set node.

    Properties
    ----------
    dptr : int
        The destination device pointer.
    value : int
        The fill value.
    element_size : int
        Element size in bytes (1, 2, or 4).
    width : int
        Width of the row in elements.
    height : int
        Number of rows.
    pitch : int
        Pitch in bytes (unused if height is 1).
    """

    @staticmethod
    cdef MemsetNode _create_with_params(GraphNodeHandle h_node,
                                        cydriver.CUdeviceptr dptr, unsigned int value,
                                        unsigned int element_size, size_t width,
                                        size_t height, size_t pitch):
        """Create from known params (called by memset() builder)."""
        cdef MemsetNode n = MemsetNode.__new__(MemsetNode)
        n._h_node = h_node
        n._dptr = dptr
        n._value = value
        n._element_size = element_size
        n._width = width
        n._height = height
        n._pitch = pitch
        return n

    @staticmethod
    cdef MemsetNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUDA_MEMSET_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemsetNodeGetParams(node, &params))
        return MemsetNode._create_with_params(
            h_node, params.dst, params.value,
            params.elementSize, params.width, params.height, params.pitch)

    def __repr__(self) -> str:
        return (f"<MemsetNode dptr=0x{self._dptr:x} "
                f"value={self._value} elem={self._element_size}>")

    @property
    def dptr(self) -> int:
        """The destination device pointer."""
        return self._dptr

    @property
    def value(self) -> int:
        """The fill value."""
        return self._value

    @property
    def element_size(self) -> int:
        """Element size in bytes (1, 2, or 4)."""
        return self._element_size

    @property
    def width(self) -> int:
        """Width of the row in elements."""
        return self._width

    @property
    def height(self) -> int:
        """Number of rows."""
        return self._height

    @property
    def pitch(self) -> int:
        """Pitch in bytes (unused if height is 1)."""
        return self._pitch


cdef class MemcpyNode(Node):
    """A memory copy node.

    Properties
    ----------
    dst : int
        The destination pointer.
    src : int
        The source pointer.
    size : int
        The number of bytes copied.
    """

    @staticmethod
    cdef MemcpyNode _create_with_params(GraphNodeHandle h_node,
                                        cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src,
                                        size_t size, cydriver.CUmemorytype dst_type,
                                        cydriver.CUmemorytype src_type):
        """Create from known params (called by memcpy() builder)."""
        cdef MemcpyNode n = MemcpyNode.__new__(MemcpyNode)
        n._h_node = h_node
        n._dst = dst
        n._src = src
        n._size = size
        n._dst_type = dst_type
        n._src_type = src_type
        return n

    @staticmethod
    cdef MemcpyNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUDA_MEMCPY3D params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemcpyNodeGetParams(node, &params))

        cdef cydriver.CUdeviceptr dst
        cdef cydriver.CUdeviceptr src
        if params.dstMemoryType == cydriver.CU_MEMORYTYPE_HOST:
            dst = <cydriver.CUdeviceptr><uintptr_t>params.dstHost
        else:
            dst = params.dstDevice
        if params.srcMemoryType == cydriver.CU_MEMORYTYPE_HOST:
            src = <cydriver.CUdeviceptr><uintptr_t>params.srcHost
        else:
            src = params.srcDevice

        return MemcpyNode._create_with_params(
            h_node, dst, src, params.WidthInBytes,
            params.dstMemoryType, params.srcMemoryType)

    def __repr__(self) -> str:
        cdef str dt = "H" if self._dst_type == cydriver.CU_MEMORYTYPE_HOST else "D"
        cdef str st = "H" if self._src_type == cydriver.CU_MEMORYTYPE_HOST else "D"
        return (f"<MemcpyNode dst=0x{self._dst:x}({dt}) "
                f"src=0x{self._src:x}({st}) size={self._size}>")

    @property
    def dst(self) -> int:
        """The destination pointer."""
        return self._dst

    @property
    def src(self) -> int:
        """The source pointer."""
        return self._src

    @property
    def size(self) -> int:
        """The number of bytes copied."""
        return self._size


cdef class ChildGraphNode(Node):
    """A child graph (sub-graph) node.

    Properties
    ----------
    child_graph : GraphDef
        The embedded graph definition (non-owning wrapper).
    """

    @staticmethod
    cdef ChildGraphNode _create_with_params(GraphNodeHandle h_node,
                                            GraphHandle h_child_graph):
        """Create from known params (called by embed() builder)."""
        cdef ChildGraphNode n = ChildGraphNode.__new__(ChildGraphNode)
        n._h_node = h_node
        n._h_child_graph = h_child_graph
        return n

    @staticmethod
    cdef ChildGraphNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUgraph child_graph = NULL
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphChildGraphNodeGetGraph(node, &child_graph))
        cdef GraphHandle h_graph = graph_node_get_graph(h_node)
        cdef GraphHandle h_child = create_graph_handle_ref(child_graph, h_graph)
        return ChildGraphNode._create_with_params(h_node, h_child)

    def __repr__(self) -> str:
        cdef cydriver.CUgraph g = as_cu(self._h_child_graph)
        cdef size_t num_nodes = 0
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(g, NULL, &num_nodes))
        cdef Py_ssize_t n = <Py_ssize_t>num_nodes
        return f"<ChildGraphNode with {n} {'subnode' if n == 1 else 'subnodes'}>"

    @property
    def child_graph(self) -> GraphDef:
        """The embedded graph definition (non-owning wrapper)."""
        return GraphDef._from_handle(self._h_child_graph)


cdef class EventRecordNode(Node):
    """An event record node.

    Properties
    ----------
    event : Event
        The event being recorded.
    """

    @staticmethod
    cdef EventRecordNode _create_with_params(GraphNodeHandle h_node,
                                             EventHandle h_event):
        """Create from known params (called by record_event() builder)."""
        cdef EventRecordNode n = EventRecordNode.__new__(EventRecordNode)
        n._h_node = h_node
        n._h_event = h_event
        return n

    @staticmethod
    cdef EventRecordNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUevent event
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphEventRecordNodeGetEvent(node, &event))
        cdef EventHandle h_event = create_event_handle_ref(event)
        return EventRecordNode._create_with_params(h_node, h_event)

    def __repr__(self) -> str:
        return f"<EventRecordNode event=0x{as_intptr(self._h_event):x}>"

    @property
    def event(self) -> Event:
        """The event being recorded."""
        return Event._from_handle(self._h_event)


cdef class EventWaitNode(Node):
    """An event wait node.

    Properties
    ----------
    event : Event
        The event being waited on.
    """

    @staticmethod
    cdef EventWaitNode _create_with_params(GraphNodeHandle h_node,
                                           EventHandle h_event):
        """Create from known params (called by wait_event() builder)."""
        cdef EventWaitNode n = EventWaitNode.__new__(EventWaitNode)
        n._h_node = h_node
        n._h_event = h_event
        return n

    @staticmethod
    cdef EventWaitNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUevent event
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphEventWaitNodeGetEvent(node, &event))
        cdef EventHandle h_event = create_event_handle_ref(event)
        return EventWaitNode._create_with_params(h_node, h_event)

    def __repr__(self) -> str:
        return f"<EventWaitNode event=0x{as_intptr(self._h_event):x}>"

    @property
    def event(self) -> Event:
        """The event being waited on."""
        return Event._from_handle(self._h_event)


cdef class HostCallbackNode(Node):
    """A host callback node.

    Properties
    ----------
    callback_fn : callable or None
        The Python callable (None for ctypes function pointer callbacks).
    """

    @staticmethod
    cdef HostCallbackNode _create_with_params(GraphNodeHandle h_node,
                                              object callable_obj, cydriver.CUhostFn fn,
                                              void* user_data):
        """Create from known params (called by callback() builder)."""
        cdef HostCallbackNode n = HostCallbackNode.__new__(HostCallbackNode)
        n._h_node = h_node
        n._callable = callable_obj
        n._fn = fn
        n._user_data = user_data
        return n

    @staticmethod
    cdef HostCallbackNode _create_from_driver(GraphNodeHandle h_node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUgraphNode node = as_cu(h_node)
        cdef cydriver.CUDA_HOST_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphHostNodeGetParams(node, &params))

        cdef object callable_obj = None
        if params.fn == <cydriver.CUhostFn>_py_host_trampoline:
            callable_obj = <object>params.userData

        return HostCallbackNode._create_with_params(
            h_node, callable_obj, params.fn, params.userData)

    def __repr__(self) -> str:
        if self._callable is not None:
            name = getattr(self._callable, '__name__', '?')
            return f"<HostCallbackNode callback={name}>"
        return f"<HostCallbackNode cfunc=0x{<uintptr_t>self._fn:x}>"

    @property
    def callback_fn(self):
        """The Python callable, or None for ctypes function pointer callbacks."""
        return self._callable


cdef class ConditionalNode(Node):
    """Base class for conditional graph nodes.

    When created via builder methods (if_cond, if_else, while_loop, switch),
    a specific subclass (IfNode, IfElseNode, WhileNode, SwitchNode) is
    returned. When reconstructed from the driver on CUDA 13.2+, the
    correct subclass is determined via cuGraphNodeGetParams. On older
    drivers, this base class is used as a fallback.

    Properties
    ----------
    condition : Condition or None
        The condition variable controlling execution (None pre-13.2).
    cond_type : str or None
        The conditional type ("if", "while", or "switch"; None pre-13.2).
    branches : tuple of GraphDef
        The body graphs for each branch (empty pre-13.2).
    """

    @staticmethod
    cdef ConditionalNode _create_from_driver(GraphNodeHandle h_node):
        cdef ConditionalNode n
        if not _check_node_get_params():
            n = ConditionalNode.__new__(ConditionalNode)
            n._h_node = h_node
            n._condition = None
            n._cond_type = cydriver.CU_GRAPH_COND_TYPE_IF
            n._branches = ()
            return n

        cdef cydriver.CUgraphNode node = as_cu(h_node)
        params = handle_return(driver.cuGraphNodeGetParams(
            <uintptr_t>node))
        cond_params = params.conditional
        cdef int cond_type_int = int(cond_params.type)
        cdef unsigned int size = int(cond_params.size)

        cdef Condition condition = Condition.__new__(Condition)
        condition._c_handle = <cydriver.CUgraphConditionalHandle>(
            <unsigned long long>int(cond_params.handle))

        cdef GraphHandle h_graph = graph_node_get_graph(h_node)
        cdef list branch_list = []
        cdef unsigned int i
        cdef GraphHandle h_branch
        if cond_params.phGraph_out is not None:
            for i in range(size):
                h_branch = create_graph_handle_ref(
                    <cydriver.CUgraph><uintptr_t>int(cond_params.phGraph_out[i]),
                    h_graph)
                branch_list.append(GraphDef._from_handle(h_branch))
        cdef tuple branches = tuple(branch_list)

        cdef type cls
        if cond_type_int == <int>cydriver.CU_GRAPH_COND_TYPE_IF:
            if size == 1:
                cls = IfNode
            else:
                cls = IfElseNode
        elif cond_type_int == <int>cydriver.CU_GRAPH_COND_TYPE_WHILE:
            cls = WhileNode
        else:
            cls = SwitchNode

        n = cls.__new__(cls)
        n._h_node = h_node
        n._condition = condition
        n._cond_type = <cydriver.CUgraphConditionalNodeType>cond_type_int
        n._branches = branches
        return n

    def __repr__(self) -> str:
        return "<ConditionalNode>"

    @property
    def condition(self) -> Condition | None:
        """The condition variable controlling execution."""
        return self._condition

    @property
    def cond_type(self) -> str | None:
        """The conditional type as a string: 'if', 'while', or 'switch'.

        Returns None when reconstructed from the driver pre-CUDA 13.2,
        as the conditional type cannot be determined.
        """
        if self._condition is None:
            return None
        if self._cond_type == cydriver.CU_GRAPH_COND_TYPE_IF:
            return "if"
        elif self._cond_type == cydriver.CU_GRAPH_COND_TYPE_WHILE:
            return "while"
        else:
            return "switch"

    @property
    def branches(self) -> tuple:
        """The body graphs for each branch as a tuple of GraphDef.

        Returns an empty tuple when reconstructed from the driver
        pre-CUDA 13.2.
        """
        return self._branches


cdef class IfNode(ConditionalNode):
    """An if-conditional node (1 branch, executes when condition is non-zero)."""

    def __repr__(self) -> str:
        return f"<IfNode condition=0x{<unsigned long long>self._condition._c_handle:x}>"

    @property
    def then(self) -> GraphDef:
        """The 'then' branch graph."""
        return self._branches[0]


cdef class IfElseNode(ConditionalNode):
    """An if-else conditional node (2 branches)."""

    def __repr__(self) -> str:
        return f"<IfElseNode condition=0x{<unsigned long long>self._condition._c_handle:x}>"

    @property
    def then(self) -> GraphDef:
        """The 'then' branch graph (executed when condition is non-zero)."""
        return self._branches[0]

    @property
    def else_(self) -> GraphDef:
        """The 'else' branch graph (executed when condition is zero)."""
        return self._branches[1]


cdef class WhileNode(ConditionalNode):
    """A while-loop conditional node (1 branch, repeats while condition is non-zero)."""

    def __repr__(self) -> str:
        return f"<WhileNode condition=0x{<unsigned long long>self._condition._c_handle:x}>"

    @property
    def body(self) -> GraphDef:
        """The loop body graph."""
        return self._branches[0]


cdef class SwitchNode(ConditionalNode):
    """A switch conditional node (N branches, selected by condition value)."""

    def __repr__(self) -> str:
        cdef Py_ssize_t n = len(self._branches)
        return (f"<SwitchNode condition=0x{<unsigned long long>self._condition._c_handle:x}"
                f" with {n} {'branch' if n == 1 else 'branches'}>")
