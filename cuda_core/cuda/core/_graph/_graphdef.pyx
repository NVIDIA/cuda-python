# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Private module for explicit CUDA graph construction.

This module provides GraphDef and Node classes for building CUDA graphs
explicitly (as opposed to stream capture). Both approaches produce the
same public Graph type for execution.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core import Device

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libc.string cimport memset

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport (
    GraphHandle,
    create_graph_handle,
    as_cu,
    as_intptr,
)
from cuda.core._module cimport Kernel
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core._utils.cuda_utils import driver


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

    def __repr__(self):
        return f"<GraphDef handle=0x{as_intptr(self._h_graph):x}>"

    def __eq__(self, other):
        if not isinstance(other, GraphDef):
            return NotImplemented
        return as_intptr(self._h_graph) == as_intptr((<GraphDef>other)._h_graph)

    def __hash__(self):
        return hash(as_intptr(self._h_graph))

    @property
    def root(self):
        """Return the root Node for this graph.

        The root node has no dependencies. Operations added from the root
        will be entry points to the graph.
        """
        return Node._create(self._h_graph, NULL, 0)

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

    def debug_dot_print(self, path: str, options=None):
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

        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef bytes path_bytes = path.encode('utf-8')
        cdef const char* c_path = path_bytes
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphDebugDotPrint(graph, c_path, flags))

    def nodes(self):
        """Return all nodes in the graph.

        Returns
        -------
        tuple of Node
            All nodes in the graph (excluding the virtual root).
        """
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef size_t num_nodes = 0

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(graph, NULL, &num_nodes))

        if num_nodes == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] nodes_vec
        nodes_vec.resize(num_nodes)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetNodes(graph, nodes_vec.data(), &num_nodes))

        return tuple(Node._create(self._h_graph, nodes_vec[i], 0) for i in range(num_nodes))

    def edges(self):
        """Return all edges in the graph as (from_node, to_node) pairs.

        Returns
        -------
        tuple of tuple
            Each element is a (from_node, to_node) pair representing
            a dependency edge in the graph.
        """
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef size_t num_edges = 0

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetEdges(graph, NULL, NULL, NULL, &num_edges))

        if num_edges == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] from_nodes
        cdef vector[cydriver.CUgraphNode] to_nodes
        from_nodes.resize(num_edges)
        to_nodes.resize(num_edges)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphGetEdges(
                graph, from_nodes.data(), to_nodes.data(), NULL, &num_edges))

        return tuple(
            (Node._create(self._h_graph, from_nodes[i], 0),
             Node._create(self._h_graph, to_nodes[i], 0))
            for i in range(num_edges)
        )

    @property
    def handle(self):
        """Return the underlying CUgraph handle."""
        return driver.CUgraph(as_intptr(self._h_graph))


cdef class Node:
    """Represents a node (or potential node) in a CUDA graph.

    Nodes are created by calling methods on other Nodes. Each method
    returns a new Node that depends on the current node(s).

    The root node (obtained from GraphDef.root) has a NULL internal
    node handle, representing graph entry points.
    """

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node, cydriver.CUdeviceptr dptr):
        """Internal factory method to create a Node."""
        cdef Node n = Node.__new__(Node)
        n._h_graph = h_graph
        n._node = node
        n._dptr = dptr
        return n

    def __repr__(self):
        if self._node == NULL:
            return "<Node root>"
        if self._dptr != 0:
            return f"<Node handle=0x{<uintptr_t>self._node:x} dptr=0x{self._dptr:x}>"
        return f"<Node handle=0x{<uintptr_t>self._node:x}>"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        cdef Node o = <Node>other
        return (as_intptr(self._h_graph) == as_intptr(o._h_graph) and
                self._node == o._node)

    def __hash__(self):
        return hash((as_intptr(self._h_graph), <uintptr_t>self._node))

    @property
    def graph(self):
        """Return the GraphDef this node belongs to."""
        return GraphDef._from_handle(self._h_graph)

    @property
    def dptr(self):
        """Return the device pointer for allocation nodes.

        Returns 0 for non-allocation nodes.
        """
        return self._dptr

    @property
    def pred(self):
        """Return the predecessor nodes (dependencies) of this node.

        Returns
        -------
        tuple of Node
            The nodes that this node depends on.
        """
        if self._node == NULL:
            return ()

        cdef size_t num_deps = 0
        cdef cydriver.CUgraphNode node = self._node

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, NULL, NULL, &num_deps))

        if num_deps == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, deps.data(), NULL, &num_deps))

        return tuple(Node._create(self._h_graph, deps[i], 0) for i in range(num_deps))

    @property
    def succ(self):
        """Return the successor nodes (dependents) of this node.

        Returns
        -------
        tuple of Node
            The nodes that depend on this node.
        """
        if self._node == NULL:
            return ()

        cdef size_t num_deps = 0
        cdef cydriver.CUgraphNode node = self._node

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, NULL, NULL, &num_deps))

        if num_deps == 0:
            return ()

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, deps.data(), NULL, &num_deps))

        return tuple(Node._create(self._h_graph, deps[i], 0) for i in range(num_deps))

    def launch(self, config, kernel, *args):
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
        Node
            A new Node representing the kernel launch.
        """
        cdef LaunchConfig conf = config
        cdef Kernel ker = <Kernel>kernel
        cdef ParamHolder ker_args = ParamHolder(args)

        cdef cydriver.CUDA_KERNEL_NODE_PARAMS node_params
        cdef cydriver.CUgraphNode new_node = NULL
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
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
            HANDLE_RETURN(cydriver.cuGraphAddKernelNode(&new_node, graph, deps, num_deps, &node_params))

        return Node._create(self._h_graph, new_node, 0)

    def join(self, *nodes):
        """Create an empty node that depends on this node and all given nodes.

        This is used to synchronize multiple branches of execution.

        Parameters
        ----------
        *nodes : Node
            Additional nodes to depend on.

        Returns
        -------
        Node
            A new Node that depends on all input nodes.
        """
        cdef vector[cydriver.CUgraphNode] deps
        cdef cydriver.CUgraphNode new_node = NULL
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef Node other
        cdef cydriver.CUgraphNode* deps_ptr = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps.push_back(self._node)
        for other in nodes:
            if (<Node>other)._node != NULL:
                deps.push_back((<Node>other)._node)

        num_deps = deps.size()
        if num_deps > 0:
            deps_ptr = deps.data()

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEmptyNode(&new_node, graph, deps_ptr, num_deps))

        return Node._create(self._h_graph, new_node, 0)

    def alloc(self, size_t size, options: GraphAllocOptions | None = None):
        """Add a memory allocation node depending on this node.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.
        options : GraphAllocOptions, optional
            Allocation options. If None, allocates on the current device.

        Returns
        -------
        Node
            A new Node representing the allocation. Access the allocated
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
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
            num_deps = 1

        cdef vector[cydriver.CUmemAccessDesc] access_descs
        cdef int peer_id

        if options is not None and options.peer_access is not None:
            for peer_dev in options.peer_access:
                peer_id = getattr(peer_dev, 'device_id', peer_dev)
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

        memset(&alloc_params, 0, sizeof(alloc_params))
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
            alloc_params.poolProps.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED
            alloc_params.poolProps.location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            alloc_params.poolProps.location.id = device_id
        else:
            raise ValueError(f"Invalid memory_type: {memory_type!r}. "
                           "Must be 'device', 'host', or 'managed'.")

        if access_descs.size() > 0:
            alloc_params.accessDescs = access_descs.data()
            alloc_params.accessDescCount = access_descs.size()

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemAllocNode(&new_node, graph, deps, num_deps, &alloc_params))

        return Node._create(self._h_graph, new_node, alloc_params.dptr)

    def free(self, dptr):
        """Add a memory free node depending on this node.

        Parameters
        ----------
        dptr : int
            Device pointer to free (typically from Node.dptr of an alloc node).

        Returns
        -------
        Node
            A new Node representing the free operation.
        """
        cdef cydriver.CUgraphNode new_node = NULL
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0
        cdef cydriver.CUdeviceptr c_dptr = <cydriver.CUdeviceptr>dptr

        if self._node != NULL:
            deps = &self._node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemFreeNode(&new_node, graph, deps, num_deps, c_dptr))

        return Node._create(self._h_graph, new_node, 0)
