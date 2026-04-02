# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mutable-set proxy for graph node predecessors and successors."""

from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cuda.bindings cimport cydriver
from cuda.core._graph._graph_def._graph_node cimport GraphNode
from cuda.core._resource_handles cimport (
    GraphHandle,
    GraphNodeHandle,
    as_cu,
    graph_node_get_graph,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from collections.abc import MutableSet


# ---- Python MutableSet wrapper ----------------------------------------------

class AdjacencySetProxy(MutableSet):
    """Mutable set proxy for a node's predecessors or successors.  Mutations
    write through to the underlying CUDA graph."""

    __slots__ = ("_core",)

    def __init__(self, node, bint is_fwd):
        self._core = _AdjacencySetCore(node, is_fwd)

    # Used by operators such as &|^ to create non-proxy views when needed.
    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    # --- abstract methods required by MutableSet ---

    def __contains__(self, x):
        if not isinstance(x, GraphNode):
            return False
        return x in (<_AdjacencySetCore>self._core).query()

    def __iter__(self):
        return iter((<_AdjacencySetCore>self._core).query())

    def __len__(self):
        return (<_AdjacencySetCore>self._core).count()

    def add(self, value):
        if not isinstance(value, GraphNode):
            raise TypeError(
                f"expected GraphNode, got {type(value).__name__}")
        if value in self:
            return
        (<_AdjacencySetCore>self._core).add_edge(<GraphNode>value)

    def discard(self, value):
        if not isinstance(value, GraphNode):
            return
        if value not in self:
            return
        (<_AdjacencySetCore>self._core).remove_edge(<GraphNode>value)

    # --- override for bulk efficiency ---

    def clear(self):
        """Remove all edges in a single driver call."""
        members = (<_AdjacencySetCore>self._core).query()
        if members:
            (<_AdjacencySetCore>self._core).remove_edges(members)

    def __isub__(self, it):
        """Remove edges to all nodes in *it* in a single driver call."""
        if it is self:
            self.clear()
        else:
            to_remove = [v for v in it if isinstance(v, GraphNode) and v in self]
            if to_remove:
                (<_AdjacencySetCore>self._core).remove_edges(to_remove)
        return self

    def update(self, *others):
        """Add edges to multiple nodes at once."""
        nodes = []
        for other in others:
            if isinstance(other, GraphNode):
                nodes.append(other)
            else:
                nodes.extend(other)
        if not nodes:
            return
        for n in nodes:
            if not isinstance(n, GraphNode):
                raise TypeError(
                    f"expected GraphNode, got {type(n).__name__}")
        new = [n for n in nodes if n not in self]
        if new:
            (<_AdjacencySetCore>self._core).add_edges(new)

    def __ior__(self, it):
        """Add edges to all nodes in *it* in a single driver call."""
        self.update(it)
        return self

    def __repr__(self):
        return "{" + ", ".join(repr(n) for n in self) + "}"


# ---- cdef core holding a function pointer ------------------------------------

# Signature shared by driver_get_preds and driver_get_succs.
ctypedef cydriver.CUresult (*_adj_fn_t)(
    cydriver.CUgraphNode, cydriver.CUgraphNode*, size_t*) noexcept nogil


cdef class _AdjacencySetCore:
    """Cythonized core implementing AdjacencySetProxy"""
    cdef:
        GraphNodeHandle _h_node
        GraphHandle _h_graph
        _adj_fn_t _query_fn
        bint _is_fwd

    def __init__(self, GraphNode node, bint is_fwd):
        self._h_node = node._h_node
        self._h_graph = graph_node_get_graph(node._h_node)
        self._is_fwd = is_fwd
        self._query_fn = driver_get_succs if is_fwd else driver_get_preds

    cdef inline void _resolve_edge(
            self, GraphNode other,
            cydriver.CUgraphNode* c_from,
            cydriver.CUgraphNode* c_to) noexcept:
        if self._is_fwd:
            c_from[0] = as_cu(self._h_node)
            c_to[0] = as_cu(other._h_node)
        else:
            c_from[0] = as_cu(other._h_node)
            c_to[0] = as_cu(self._h_node)

    cdef list query(self):
        cdef cydriver.CUgraphNode c_node = as_cu(self._h_node)
        if c_node == NULL:
            return []
        cdef size_t count = 0
        with nogil:
            HANDLE_RETURN(self._query_fn(c_node, NULL, &count))
        if count == 0:
            return []
        cdef vector[cydriver.CUgraphNode] nodes_vec
        nodes_vec.resize(count)
        with nogil:
            HANDLE_RETURN(self._query_fn(
                c_node, nodes_vec.data(), &count))
        return [GraphNode._create(self._h_graph, nodes_vec[i])
                for i in range(count)]

    cdef Py_ssize_t count(self):
        cdef cydriver.CUgraphNode c_node = as_cu(self._h_node)
        if c_node == NULL:
            return 0
        cdef size_t n = 0
        with nogil:
            HANDLE_RETURN(self._query_fn(c_node, NULL, &n))
        return <Py_ssize_t>n

    cdef void add_edge(self, GraphNode other):
        cdef cydriver.CUgraphNode c_from, c_to
        self._resolve_edge(other, &c_from, &c_to)
        with nogil:
            HANDLE_RETURN(driver_add_edges(as_cu(self._h_graph), &c_from, &c_to, 1))

    cdef void add_edges(self, list nodes):
        cdef size_t n = len(nodes)
        cdef vector[cydriver.CUgraphNode] from_vec
        cdef vector[cydriver.CUgraphNode] to_vec
        from_vec.resize(n)
        to_vec.resize(n)
        cdef size_t i
        for i in range(n):
            self._resolve_edge(<GraphNode>nodes[i], &from_vec[i], &to_vec[i])
        with nogil:
            HANDLE_RETURN(driver_add_edges(
                as_cu(self._h_graph), from_vec.data(), to_vec.data(), n))

    cdef void remove_edge(self, GraphNode other):
        cdef cydriver.CUgraphNode c_from, c_to
        self._resolve_edge(other, &c_from, &c_to)
        with nogil:
            HANDLE_RETURN(driver_remove_edges(as_cu(self._h_graph), &c_from, &c_to, 1))

    cdef void remove_edges(self, list nodes):
        cdef size_t n = len(nodes)
        cdef vector[cydriver.CUgraphNode] from_vec
        cdef vector[cydriver.CUgraphNode] to_vec
        from_vec.resize(n)
        to_vec.resize(n)
        cdef size_t i
        for i in range(n):
            self._resolve_edge(<GraphNode>nodes[i], &from_vec[i], &to_vec[i])
        with nogil:
            HANDLE_RETURN(driver_remove_edges(
                as_cu(self._h_graph), from_vec.data(), to_vec.data(), n))


# ---- driver wrappers: absorb CUDA version differences ----

cdef inline cydriver.CUresult driver_get_preds(
        cydriver.CUgraphNode node, cydriver.CUgraphNode* out,
        size_t* count) noexcept nogil:
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cydriver.cuGraphNodeGetDependencies(node, out, NULL, count)
    ELSE:
        return cydriver.cuGraphNodeGetDependencies(node, out, count)


cdef inline cydriver.CUresult driver_get_succs(
        cydriver.CUgraphNode node, cydriver.CUgraphNode* out,
        size_t* count) noexcept nogil:
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cydriver.cuGraphNodeGetDependentNodes(node, out, NULL, count)
    ELSE:
        return cydriver.cuGraphNodeGetDependentNodes(node, out, count)


cdef inline cydriver.CUresult driver_add_edges(
        cydriver.CUgraph graph, cydriver.CUgraphNode* from_arr,
        cydriver.CUgraphNode* to_arr, size_t count) noexcept nogil:
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cydriver.cuGraphAddDependencies(
            graph, from_arr, to_arr, NULL, count)
    ELSE:
        return cydriver.cuGraphAddDependencies(
            graph, from_arr, to_arr, count)


cdef inline cydriver.CUresult driver_remove_edges(
        cydriver.CUgraph graph, cydriver.CUgraphNode* from_arr,
        cydriver.CUgraphNode* to_arr, size_t count) noexcept nogil:
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cydriver.cuGraphRemoveDependencies(
            graph, from_arr, to_arr, NULL, count)
    ELSE:
        return cydriver.cuGraphRemoveDependencies(
            graph, from_arr, to_arr, count)
