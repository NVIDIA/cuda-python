# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GraphDefinition: explicit CUDA graph definition."""

from __future__ import annotations

from libc.stddef cimport size_t

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from cuda.core.graph._graph_node cimport GraphNode
from cuda.core._resource_handles cimport (
    GraphHandle,
    as_cu,
    as_intptr,
    as_py,
    create_graph_handle,
    create_graph_node_handle,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from dataclasses import dataclass

from cuda.core._utils.cuda_utils import driver

__all__ = ['GraphCondition', 'GraphAllocOptions', 'GraphDefinition']


cdef class GraphCondition:
    """A condition variable for conditional graph nodes.

    Created by :meth:`GraphDefinition.create_condition` (or
    :meth:`GraphBuilder.create_condition`) and passed to
    conditional-node builder methods (:meth:`~GraphDefinition.if_then`,
    :meth:`~GraphDefinition.if_else`, :meth:`~GraphDefinition.while_loop`,
    :meth:`~GraphDefinition.switch`). The underlying value is set at
    runtime by device code via ``cudaGraphSetConditional``.

    A :class:`GraphCondition` may be passed directly as a kernel
    argument to ``launch()``: the launcher unwraps it to the underlying
    ``CUgraphConditionalHandle`` value so device code can update the
    condition.
    """

    @staticmethod
    cdef GraphCondition _from_handle(cydriver.CUgraphConditionalHandle c_handle):
        cdef GraphCondition self = GraphCondition.__new__(GraphCondition)
        self._c_handle = c_handle
        return self

    def __repr__(self) -> str:
        return f"<GraphCondition handle=0x{<unsigned long long>self._c_handle:x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphCondition):
            return NotImplemented
        return self._c_handle == (<GraphCondition>other)._c_handle

    def __hash__(self) -> int:
        return hash(<unsigned long long>self._c_handle)

    @property
    def handle(self) -> driver.CUgraphConditionalHandle:
        """The raw CUgraphConditionalHandle as an int."""
        return <unsigned long long>self._c_handle


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

    device: int | "Device" | None = None
    memory_type: str = "device"
    peer_access: list | None = None


cdef class GraphDefinition:
    """A graph definition.

    A GraphDefinition is used to construct a graph explicitly by adding nodes
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
    cdef GraphDefinition _from_handle(GraphHandle h_graph):
        """Create a GraphDefinition from an existing GraphHandle (internal use)."""
        cdef GraphDefinition g = GraphDefinition.__new__(GraphDefinition)
        g._h_graph = h_graph
        return g

    def __repr__(self) -> str:
        return f"<GraphDefinition handle=0x{as_intptr(self._h_graph):x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphDefinition):
            return NotImplemented
        return as_intptr(self._h_graph) == as_intptr((<GraphDefinition>other)._h_graph)

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_graph))

    @property
    def _entry(self) -> "GraphNode":
        """Return the internal entry-point GraphNode (no dependencies)."""
        cdef GraphNode n = GraphNode.__new__(GraphNode)
        n._h_node = create_graph_node_handle(<cydriver.CUgraphNode>NULL, self._h_graph)
        return n

    def allocate(self, size_t size, options: GraphAllocOptions | None = None) -> "AllocNode":
        """Add an entry-point memory allocation node (no dependencies).

        See :meth:`GraphNode.allocate` for full documentation.
        """
        return self._entry.allocate(size, options)

    def deallocate(self, dptr) -> "FreeNode":
        """Add an entry-point memory free node (no dependencies).

        See :meth:`GraphNode.deallocate` for full documentation.
        """
        return self._entry.deallocate(dptr)

    def memset(self, dst, value, size_t width, size_t height=1, size_t pitch=0) -> "MemsetNode":
        """Add an entry-point memset node (no dependencies).

        See :meth:`GraphNode.memset` for full documentation.
        """
        return self._entry.memset(dst, value, width, height, pitch)

    def launch(self, config, kernel, *args) -> "KernelNode":
        """Add an entry-point kernel launch node (no dependencies).

        See :meth:`GraphNode.launch` for full documentation.
        """
        return self._entry.launch(config, kernel, *args)

    def empty(self) -> "EmptyNode":
        """Add an entry-point empty node (no dependencies).

        Returns
        -------
        EmptyNode
            A new EmptyNode with no dependencies.
        """
        return self._entry.join()

    def join(self, *nodes) -> "EmptyNode":
        """Create an empty node that depends on all given nodes.

        Parameters
        ----------
        *nodes : GraphNode
            Nodes to merge.

        Returns
        -------
        EmptyNode
            A new EmptyNode that depends on all input nodes.
        """
        return self._entry.join(*nodes)

    def memcpy(self, dst, src, size_t size) -> "MemcpyNode":
        """Add an entry-point memcpy node (no dependencies).

        See :meth:`GraphNode.memcpy` for full documentation.
        """
        return self._entry.memcpy(dst, src, size)

    def embed(self, child: GraphDefinition) -> "ChildGraphNode":
        """Add an entry-point child graph node (no dependencies).

        See :meth:`GraphNode.embed` for full documentation.
        """
        return self._entry.embed(child)

    def record(self, event) -> "EventRecordNode":
        """Add an entry-point event record node (no dependencies).

        See :meth:`GraphNode.record` for full documentation.
        """
        return self._entry.record(event)

    def wait(self, event) -> "EventWaitNode":
        """Add an entry-point event wait node (no dependencies).

        See :meth:`GraphNode.wait` for full documentation.
        """
        return self._entry.wait(event)

    def callback(self, fn, *, user_data=None) -> "HostCallbackNode":
        """Add an entry-point host callback node (no dependencies).

        See :meth:`GraphNode.callback` for full documentation.
        """
        return self._entry.callback(fn, user_data=user_data)

    def create_condition(self, default_value: int | None = None) -> GraphCondition:
        """Create a condition variable for use with conditional nodes.

        The returned :class:`GraphCondition` object is passed to conditional-node
        builder methods. Its value is controlled at runtime by device code
        via ``cudaGraphSetConditional``.

        Parameters
        ----------
        default_value : int, optional
            The default value to assign to the condition.
            If None, no default is assigned.

        Returns
        -------
        GraphCondition
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

        return GraphCondition._from_handle(c_handle)

    def if_then(self, condition: GraphCondition) -> "IfNode":
        """Add an entry-point if-conditional node (no dependencies).

        See :meth:`GraphNode.if_then` for full documentation.
        """
        return self._entry.if_then(condition)

    def if_else(self, condition: GraphCondition) -> "IfElseNode":
        """Add an entry-point if-else conditional node (no dependencies).

        See :meth:`GraphNode.if_else` for full documentation.
        """
        return self._entry.if_else(condition)

    def while_loop(self, condition: GraphCondition) -> "WhileNode":
        """Add an entry-point while-loop conditional node (no dependencies).

        See :meth:`GraphNode.while_loop` for full documentation.
        """
        return self._entry.while_loop(condition)

    def switch(self, condition: GraphCondition, unsigned int count) -> "SwitchNode":
        """Add an entry-point switch conditional node (no dependencies).

        See :meth:`GraphNode.switch` for full documentation.
        """
        return self._entry.switch(condition, count)

    def instantiate(self, options=None):
        """Instantiate the graph definition into an executable Graph.

        Parameters
        ----------
        options : :obj:`~graph.GraphCompleteOptions`, optional
            Customizable dataclass for graph instantiation options.

        Returns
        -------
        Graph
            An executable graph that can be launched on a stream.
        """
        from cuda.core.graph._graph_builder import _instantiate_graph

        return _instantiate_graph(
            driver.CUgraph(as_intptr(self._h_graph)), options)

    def debug_dot_print(self, path: str, options=None) -> None:
        """Write a GraphViz DOT representation of the graph to a file.

        Parameters
        ----------
        path : str
            File path for the DOT output.
        options : GraphDebugPrintOptions, optional
            Customizable options for the debug print.
        """
        from cuda.core.graph._graph_builder import GraphDebugPrintOptions

        cdef unsigned int flags = 0
        if options is not None:
            if not isinstance(options, GraphDebugPrintOptions):
                raise TypeError("options must be a GraphDebugPrintOptions instance")
            flags = options._to_flags()

        cdef bytes path_bytes = path.encode('utf-8')
        cdef const char* c_path = path_bytes
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphDebugDotPrint(as_cu(self._h_graph), c_path, flags))

    def nodes(self) -> set:
        """Return all nodes in the graph.

        Returns
        -------
        set of GraphNode
            All nodes in the graph.
        """
        cdef vector[cydriver.CUgraphNode] nodes_vec
        nodes_vec.resize(128)
        cdef size_t num_nodes = 128

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(as_cu(self._h_graph), nodes_vec.data(), &num_nodes))

        if num_nodes == 0:
            return set()

        if num_nodes > 128:
            nodes_vec.resize(num_nodes)
            with nogil:
                HANDLE_RETURN(cydriver.cuGraphGetNodes(as_cu(self._h_graph), nodes_vec.data(), &num_nodes))

        return {GraphNode._create(self._h_graph, nodes_vec[i]) for i in range(num_nodes)}

    def edges(self) -> set:
        """Return all edges in the graph as (from_node, to_node) pairs.

        Returns
        -------
        set of tuple
            Each element is a (from_node, to_node) pair representing
            a dependency edge in the graph.
        """
        cdef vector[cydriver.CUgraphNode] from_nodes
        cdef vector[cydriver.CUgraphNode] to_nodes
        from_nodes.resize(128)
        to_nodes.resize(128)
        cdef size_t num_edges = 128

        with nogil:
            IF CUDA_CORE_BUILD_MAJOR >= 13:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(
                    as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), NULL, &num_edges))
            ELSE:
                HANDLE_RETURN(cydriver.cuGraphGetEdges(
                    as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), &num_edges))

        if num_edges == 0:
            return set()

        if num_edges > 128:
            from_nodes.resize(num_edges)
            to_nodes.resize(num_edges)
            with nogil:
                IF CUDA_CORE_BUILD_MAJOR >= 13:
                    HANDLE_RETURN(cydriver.cuGraphGetEdges(
                        as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), NULL, &num_edges))
                ELSE:
                    HANDLE_RETURN(cydriver.cuGraphGetEdges(
                        as_cu(self._h_graph), from_nodes.data(), to_nodes.data(), &num_edges))

        return {
            (GraphNode._create(self._h_graph, from_nodes[i]),
             GraphNode._create(self._h_graph, to_nodes[i]))
            for i in range(num_edges)
        }

    @property
    def handle(self) -> driver.CUgraph:
        """Return the underlying driver CUgraph handle."""
        return as_py(self._h_graph)
