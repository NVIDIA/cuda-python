# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Private module for explicit CUDA graph construction.

This module provides GraphDef and a Node class hierarchy for building CUDA
graphs explicitly (as opposed to stream capture). Both approaches produce
the same public Graph type for execution.

Node hierarchy:
    Node (base — also used for the virtual root)
    ├── EmptyNode         (synchronization / join point)
    ├── KernelNode        (kernel launch)
    ├── AllocNode         (memory allocation, exposes dptr and bytesize)
    ├── FreeNode          (memory free, exposes dptr)
    ├── MemsetNode        (memory set, exposes dptr, value, element_size, etc.)
    ├── EventRecordNode   (record an event)
    └── EventWaitNode     (wait for an event)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core import Device

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libc.string cimport memset as c_memset

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport (
    GraphHandle,
    create_graph_handle,
    as_cu,
    as_intptr,
)
from cuda.core._event cimport Event
from cuda.core._module cimport Kernel
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, _parse_fill_value

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
        cdef Node n = Node.__new__(Node)
        n._h_graph = self._h_graph
        n._node = NULL
        return n

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

        return tuple(Node._create(self._h_graph, nodes_vec[i]) for i in range(num_nodes))

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
            (Node._create(self._h_graph, from_nodes[i]),
             Node._create(self._h_graph, to_nodes[i]))
            for i in range(num_edges)
        )

    @property
    def handle(self):
        """Return the underlying CUgraph handle."""
        return driver.CUgraph(as_intptr(self._h_graph))


cdef class Node:
    """Base class for all graph nodes.

    Nodes are created by calling methods on other Nodes. Each method
    returns a new Node subclass that depends on the current node(s).

    The root node (obtained from GraphDef.root) is a base Node with a
    NULL internal handle, representing graph entry points.
    """

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Factory: dispatch to the right subclass based on node type."""
        if node == NULL:
            n = Node.__new__(Node)
            (<Node>n)._h_graph = h_graph
            (<Node>n)._node = NULL
            return n

        cdef cydriver.CUgraphNodeType node_type
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetType(node, &node_type))

        if node_type == cydriver.CU_GRAPH_NODE_TYPE_EMPTY:
            return EmptyNode._create_impl(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_KERNEL:
            return KernelNode._create_from_driver(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEM_ALLOC:
            return AllocNode._create_from_driver(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEM_FREE:
            return FreeNode._create_from_driver(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_MEMSET:
            return MemsetNode._create_from_driver(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            return EventRecordNode._create_from_driver(h_graph, node)
        elif node_type == cydriver.CU_GRAPH_NODE_TYPE_WAIT_EVENT:
            return EventWaitNode._create_from_driver(h_graph, node)
        else:
            n = Node.__new__(Node)
            (<Node>n)._h_graph = h_graph
            (<Node>n)._node = node
            return n

    def __repr__(self):
        if self._node == NULL:
            return "<Node root>"
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
    def type(self):
        """Return the CUDA graph node type.

        Returns
        -------
        CUgraphNodeType or None
            The node type enum value, or None for the virtual root node.
        """
        if self._node == NULL:
            return None
        cdef cydriver.CUgraphNodeType node_type
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetType(self._node, &node_type))
        return driver.CUgraphNodeType(<int>node_type)

    @property
    def graph(self):
        """Return the GraphDef this node belongs to."""
        return GraphDef._from_handle(self._h_graph)

    @property
    def pred(self):
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

        if self._node == NULL:
            self._pred_cache = ()
            return self._pred_cache

        cdef size_t num_deps = 0
        cdef cydriver.CUgraphNode node = self._node

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, NULL, NULL, &num_deps))

        if num_deps == 0:
            self._pred_cache = ()
            return self._pred_cache

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependencies(node, deps.data(), NULL, &num_deps))

        self._pred_cache = tuple(Node._create(self._h_graph, deps[i]) for i in range(num_deps))
        return self._pred_cache

    @property
    def succ(self):
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

        if self._node == NULL:
            self._succ_cache = ()
            return self._succ_cache

        cdef size_t num_deps = 0
        cdef cydriver.CUgraphNode node = self._node

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, NULL, NULL, &num_deps))

        if num_deps == 0:
            self._succ_cache = ()
            return self._succ_cache

        cdef vector[cydriver.CUgraphNode] deps
        deps.resize(num_deps)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphNodeGetDependentNodes(node, deps.data(), NULL, &num_deps))

        self._succ_cache = tuple(Node._create(self._h_graph, deps[i]) for i in range(num_deps))
        return self._succ_cache

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
        KernelNode
            A new KernelNode representing the kernel launch.
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

        self._succ_cache = None
        return KernelNode._create_with_params(
            self._h_graph, new_node,
            conf.grid, conf.block, conf.shmem_size,
            node_params.kern)

    def join(self, *nodes):
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

        self._succ_cache = None
        for other in nodes:
            (<Node>other)._succ_cache = None
        return EmptyNode._create_impl(self._h_graph, new_node)

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
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
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

        self._succ_cache = None
        return AllocNode._create_with_params(
            self._h_graph, new_node, alloc_params.dptr, size,
            device_id, memory_type, tuple(peer_ids))

    def free(self, dptr):
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
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0
        cdef cydriver.CUdeviceptr c_dptr = <cydriver.CUdeviceptr>dptr

        if self._node != NULL:
            deps = &self._node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddMemFreeNode(&new_node, graph, deps, num_deps, c_dptr))

        self._succ_cache = None
        return FreeNode._create_with_params(self._h_graph, new_node, c_dptr)

    def memset(self, dst, value, size_t width, size_t height=1, size_t pitch=0):
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
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
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
                &new_node, graph, deps, num_deps,
                &memset_params, ctx))

        self._succ_cache = None
        return MemsetNode._create_with_params(
            self._h_graph, new_node, c_dst,
            val, elem_size, width, height, pitch)

    def record_event(self, event):
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
        cdef cydriver.CUevent c_event = as_cu(ev._h_event)
        cdef cydriver.CUgraphNode new_node = NULL
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEventRecordNode(
                &new_node, graph, deps, num_deps, c_event))

        self._succ_cache = None
        return EventRecordNode._create_with_params(self._h_graph, new_node, c_event)

    def wait_event(self, event):
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
        cdef cydriver.CUevent c_event = as_cu(ev._h_event)
        cdef cydriver.CUgraphNode new_node = NULL
        cdef cydriver.CUgraph graph = as_cu(self._h_graph)
        cdef cydriver.CUgraphNode* deps = NULL
        cdef size_t num_deps = 0

        if self._node != NULL:
            deps = &self._node
            num_deps = 1

        with nogil:
            HANDLE_RETURN(cydriver.cuGraphAddEventWaitNode(
                &new_node, graph, deps, num_deps, c_event))

        self._succ_cache = None
        return EventWaitNode._create_with_params(self._h_graph, new_node, c_event)


# =============================================================================
# Node subclasses
# =============================================================================


cdef class EmptyNode(Node):
    """A synchronization / join node with no operation."""

    @staticmethod
    cdef EmptyNode _create_impl(GraphHandle h_graph, cydriver.CUgraphNode node):
        cdef EmptyNode n = EmptyNode.__new__(EmptyNode)
        n._h_graph = h_graph
        n._node = node
        return n

    def __repr__(self):
        return f"<EmptyNode handle=0x{<uintptr_t>self._node:x}>"


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
    cdef KernelNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                        tuple grid, tuple block, unsigned int shmem_size,
                                        cydriver.CUkernel kern):
        """Create from known params (called by launch() builder)."""
        cdef KernelNode n = KernelNode.__new__(KernelNode)
        n._h_graph = h_graph
        n._node = node
        n._grid = grid
        n._block = block
        n._shmem_size = shmem_size
        n._kern = kern
        return n

    @staticmethod
    cdef KernelNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUDA_KERNEL_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphKernelNodeGetParams(node, &params))
        return KernelNode._create_with_params(
            h_graph, node,
            (params.gridDimX, params.gridDimY, params.gridDimZ),
            (params.blockDimX, params.blockDimY, params.blockDimZ),
            params.sharedMemBytes,
            params.kern)

    def __repr__(self):
        return f"<KernelNode handle=0x{<uintptr_t>self._node:x}>"

    @property
    def grid(self):
        """Grid dimensions as a 3-tuple (gridDimX, gridDimY, gridDimZ)."""
        return self._grid

    @property
    def block(self):
        """Block dimensions as a 3-tuple (blockDimX, blockDimY, blockDimZ)."""
        return self._block

    @property
    def shmem_size(self):
        """Dynamic shared memory size in bytes."""
        return self._shmem_size

    @property
    def kernel(self):
        """The Kernel object for this launch node."""
        return Kernel.from_handle(<uintptr_t>self._kern)

    @property
    def config(self):
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
    cdef AllocNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                       cydriver.CUdeviceptr dptr, size_t bytesize,
                                       int device_id, str memory_type, tuple peer_access):
        """Create from known params (called by alloc() builder)."""
        cdef AllocNode n = AllocNode.__new__(AllocNode)
        n._h_graph = h_graph
        n._node = node
        n._dptr = dptr
        n._bytesize = bytesize
        n._device_id = device_id
        n._memory_type = memory_type
        n._peer_access = peer_access
        return n

    @staticmethod
    cdef AllocNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemAllocNodeGetParams(node, &params))

        cdef str memory_type
        if params.poolProps.allocType == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED:
            if params.poolProps.location.type == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST:
                memory_type = "host"
            else:
                memory_type = "device"
        elif params.poolProps.allocType == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED:
            memory_type = "managed"
        else:
            memory_type = "device"

        cdef list peer_ids = []
        cdef size_t i
        for i in range(params.accessDescCount):
            peer_ids.append(<int>params.accessDescs[i].location.id)

        return AllocNode._create_with_params(
            h_graph, node, params.dptr, params.bytesize,
            <int>params.poolProps.location.id, memory_type, tuple(peer_ids))

    def __repr__(self):
        return (f"<AllocNode handle=0x{<uintptr_t>self._node:x} "
                f"dptr=0x{self._dptr:x} size={self._bytesize}>")

    @property
    def dptr(self):
        """The device pointer for the allocation."""
        return self._dptr

    @property
    def bytesize(self):
        """The number of bytes allocated."""
        return self._bytesize

    @property
    def device_id(self):
        """The device on which the allocation was made."""
        return self._device_id

    @property
    def memory_type(self):
        """The type of memory: ``"device"``, ``"host"``, or ``"managed"``."""
        return self._memory_type

    @property
    def peer_access(self):
        """Device IDs with read-write access to this allocation."""
        return self._peer_access

    @property
    def options(self):
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
    cdef FreeNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                      cydriver.CUdeviceptr dptr):
        """Create from known params (called by free() builder)."""
        cdef FreeNode n = FreeNode.__new__(FreeNode)
        n._h_graph = h_graph
        n._node = node
        n._dptr = dptr
        return n

    @staticmethod
    cdef FreeNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUdeviceptr dptr
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemFreeNodeGetParams(node, &dptr))
        return FreeNode._create_with_params(h_graph, node, dptr)

    def __repr__(self):
        return f"<FreeNode handle=0x{<uintptr_t>self._node:x} dptr=0x{self._dptr:x}>"

    @property
    def dptr(self):
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
    cdef MemsetNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                        cydriver.CUdeviceptr dptr, unsigned int value,
                                        unsigned int element_size, size_t width,
                                        size_t height, size_t pitch):
        """Create from known params (called by memset() builder)."""
        cdef MemsetNode n = MemsetNode.__new__(MemsetNode)
        n._h_graph = h_graph
        n._node = node
        n._dptr = dptr
        n._value = value
        n._element_size = element_size
        n._width = width
        n._height = height
        n._pitch = pitch
        return n

    @staticmethod
    cdef MemsetNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUDA_MEMSET_NODE_PARAMS params
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphMemsetNodeGetParams(node, &params))
        return MemsetNode._create_with_params(
            h_graph, node, params.dst, params.value,
            params.elementSize, params.width, params.height, params.pitch)

    def __repr__(self):
        return (f"<MemsetNode handle=0x{<uintptr_t>self._node:x} "
                f"dptr=0x{self._dptr:x}>")

    @property
    def dptr(self):
        """The destination device pointer."""
        return self._dptr

    @property
    def value(self):
        """The fill value."""
        return self._value

    @property
    def element_size(self):
        """Element size in bytes (1, 2, or 4)."""
        return self._element_size

    @property
    def width(self):
        """Width of the row in elements."""
        return self._width

    @property
    def height(self):
        """Number of rows."""
        return self._height

    @property
    def pitch(self):
        """Pitch in bytes (unused if height is 1)."""
        return self._pitch


cdef class EventRecordNode(Node):
    """An event record node.

    Properties
    ----------
    event : Event
        The event being recorded (non-owning wrapper).
    """

    @staticmethod
    cdef EventRecordNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                             cydriver.CUevent event):
        """Create from known params (called by record_event() builder)."""
        cdef EventRecordNode n = EventRecordNode.__new__(EventRecordNode)
        n._h_graph = h_graph
        n._node = node
        n._event = event
        return n

    @staticmethod
    cdef EventRecordNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUevent event
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphEventRecordNodeGetEvent(node, &event))
        return EventRecordNode._create_with_params(h_graph, node, event)

    def __repr__(self):
        return f"<EventRecordNode handle=0x{<uintptr_t>self._node:x}>"

    @property
    def event(self):
        """The event being recorded (non-owning wrapper)."""
        return Event._from_raw_handle(self._event)


cdef class EventWaitNode(Node):
    """An event wait node.

    Properties
    ----------
    event : Event
        The event being waited on (non-owning wrapper).
    """

    @staticmethod
    cdef EventWaitNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                           cydriver.CUevent event):
        """Create from known params (called by wait_event() builder)."""
        cdef EventWaitNode n = EventWaitNode.__new__(EventWaitNode)
        n._h_graph = h_graph
        n._node = node
        n._event = event
        return n

    @staticmethod
    cdef EventWaitNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Create by fetching params from the driver (called by _create factory)."""
        cdef cydriver.CUevent event
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphEventWaitNodeGetEvent(node, &event))
        return EventWaitNode._create_with_params(h_graph, node, event)

    def __repr__(self):
        return f"<EventWaitNode handle=0x{<uintptr_t>self._node:x}>"

    @property
    def event(self):
        """The event being waited on (non-owning wrapper)."""
        return Event._from_raw_handle(self._event)
