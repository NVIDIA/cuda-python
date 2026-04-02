# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GraphNode base class — factory, properties, and builder methods."""

from __future__ import annotations

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libc.string cimport memset as c_memset

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from cuda.core._event cimport Event
from cuda.core._kernel_arg_handler cimport ParamHolder
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._module cimport Kernel
from cuda.core._graph._graph_def._graph_def cimport Condition, GraphDef
from cuda.core._graph._graph_def._subclasses cimport (
    AllocNode,
    ChildGraphNode,
    ConditionalNode,
    EmptyNode,
    EventRecordNode,
    EventWaitNode,
    FreeNode,
    HostCallbackNode,
    IfElseNode,
    IfNode,
    KernelNode,
    MemcpyNode,
    MemsetNode,
    SwitchNode,
    WhileNode,
)
from cuda.core._resource_handles cimport (
    EventHandle,
    GraphHandle,
    KernelHandle,
    GraphNodeHandle,
    as_cu,
    as_intptr,
    as_py,
    create_event_handle_ref,
    create_graph_handle_ref,
    create_graph_node_handle,
    graph_node_get_graph,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, _parse_fill_value

from cuda.core._graph._utils cimport (
    _attach_host_callback_to_graph,
    _attach_user_object,
)

from cuda.core import Device
from cuda.core._graph._graph_def._adjacency_set import AdjacencySet
from cuda.core._utils.cuda_utils import driver, handle_return


cdef class GraphNode:
    """Base class for all graph nodes.

    Nodes are created by calling builder methods on GraphDef (for
    entry-point nodes with no dependencies) or on other Nodes (for
    nodes that depend on a predecessor).
    """

    @staticmethod
    cdef GraphNode _create(GraphHandle h_graph, cydriver.CUgraphNode node):
        """Factory: dispatch to the right subclass based on node type."""
        return GN_create(h_graph, node)

    def __repr__(self) -> str:
        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        if node == NULL:
            return "<GraphNode entry>"
        return f"<GraphNode handle=0x{<uintptr_t>node:x}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        cdef GraphNode o = <GraphNode>other
        cdef GraphHandle self_graph = graph_node_get_graph(self._h_node)
        cdef GraphHandle other_graph = graph_node_get_graph(o._h_node)
        return (as_intptr(self._h_node) == as_intptr(o._h_node)
                and as_intptr(self_graph) == as_intptr(other_graph))

    def __hash__(self) -> int:
        cdef GraphHandle g = graph_node_get_graph(self._h_node)
        return hash((as_intptr(self._h_node), as_intptr(g)))

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
    def graph(self) -> "GraphDef":
        """Return the GraphDef this node belongs to."""
        return GraphDef._from_handle(graph_node_get_graph(self._h_node))

    @property
    def handle(self) -> driver.CUgraphNode:
        """Return the underlying driver CUgraphNode handle.

        Returns None for the entry node.
        """
        return as_py(self._h_node)

    def discard(self):
        """Discard this node and remove all its edges from the parent graph."""
        cdef cydriver.CUgraphNode node = as_cu(self._h_node)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphDestroyNode(node))

    @property
    def pred(self):
        """A mutable set-like view of this node's predecessors."""
        return AdjacencySet(self, False)

    @pred.setter
    def pred(self, value):
        p = AdjacencySet(self, False)
        p.clear()
        p.update(value)

    @property
    def succ(self):
        """A mutable set-like view of this node's successors."""
        return AdjacencySet(self, True)

    @succ.setter
    def succ(self, value):
        s = AdjacencySet(self, True)
        s.clear()
        s.update(value)

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
        return GN_launch(self, config, <Kernel>kernel, ParamHolder(args))

    def join(self, *nodes: GraphNode) -> EmptyNode:
        """Create an empty node that depends on this node and all given nodes.

        This is used to synchronize multiple branches of execution.

        Parameters
        ----------
        *nodes : GraphNode
            Additional nodes to depend on.

        Returns
        -------
        EmptyNode
            A new EmptyNode that depends on all input nodes.
        """
        return GN_join(self, nodes)

    def alloc(self, size_t size, options=None) -> AllocNode:
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
        return GN_alloc(self, size, options)

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
        return GN_free(self, <cydriver.CUdeviceptr>dptr)

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
        return GN_memset(self, <cydriver.CUdeviceptr>dst, val, elem_size, width, height, pitch)

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
        return GN_memcpy(self, <cydriver.CUdeviceptr>dst, <cydriver.CUdeviceptr>src, size)

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
        return GN_embed(self, <GraphDef>child)

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
        return GN_record_event(self, <Event>event)

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
        return GN_wait_event(self, <Event>event)

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
        return GN_callback(self, fn, user_data)

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


cdef void _destroy_event_handle_copy(void* ptr) noexcept nogil:
    cdef EventHandle* p = <EventHandle*>ptr
    del p


cdef void _destroy_kernel_handle_copy(void* ptr) noexcept nogil:
    cdef KernelHandle* p = <KernelHandle*>ptr
    del p


cdef inline ConditionalNode _make_conditional_node(
        GraphNode pred,
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

    return n

cdef inline GraphNode GN_create(GraphHandle h_graph, cydriver.CUgraphNode node):
    if node == NULL:
        n = GraphNode.__new__(GraphNode)
        (<GraphNode>n)._h_node = create_graph_node_handle(node, h_graph)
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
        n = GraphNode.__new__(GraphNode)
        (<GraphNode>n)._h_node = h_node
        return n


cdef inline KernelNode GN_launch(GraphNode self, LaunchConfig conf, Kernel ker, ParamHolder ker_args):
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

    return KernelNode._create_with_params(
        create_graph_node_handle(new_node, h_graph),
        conf.grid, conf.block, conf.shmem_size,
        ker._h_kernel)


cdef inline EmptyNode GN_join(GraphNode self, tuple nodes):
    cdef vector[cydriver.CUgraphNode] deps
    cdef cydriver.CUgraphNode new_node = NULL
    cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
    cdef GraphNode other
    cdef cydriver.CUgraphNode* deps_ptr = NULL
    cdef size_t num_deps = 0
    cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)

    if pred_node != NULL:
        deps.push_back(pred_node)
    for other in nodes:
        if as_cu((<GraphNode>other)._h_node) != NULL:
            deps.push_back(as_cu((<GraphNode>other)._h_node))

    num_deps = deps.size()
    if num_deps > 0:
        deps_ptr = deps.data()

    with nogil:
        HANDLE_RETURN(cydriver.cuGraphAddEmptyNode(
            &new_node, as_cu(h_graph), deps_ptr, num_deps))

    return EmptyNode._create_impl(create_graph_node_handle(new_node, h_graph))


cdef inline AllocNode GN_alloc(GraphNode self, size_t size, object options):
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

    return AllocNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), alloc_params.dptr, size,
        device_id, memory_type, tuple(peer_ids))


cdef inline FreeNode GN_free(GraphNode self, cydriver.CUdeviceptr c_dptr):
    cdef cydriver.CUgraphNode new_node = NULL
    cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
    cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
    cdef cydriver.CUgraphNode* deps = NULL
    cdef size_t num_deps = 0

    if pred_node != NULL:
        deps = &pred_node
        num_deps = 1

    with nogil:
        HANDLE_RETURN(cydriver.cuGraphAddMemFreeNode(
            &new_node, as_cu(h_graph), deps, num_deps, c_dptr))

    return FreeNode._create_with_params(create_graph_node_handle(new_node, h_graph), c_dptr)


cdef inline MemsetNode GN_memset(
        GraphNode self, cydriver.CUdeviceptr c_dst,
        unsigned int val, unsigned int elem_size,
        size_t width, size_t height, size_t pitch):
    cdef cydriver.CUDA_MEMSET_NODE_PARAMS memset_params
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

    return MemsetNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), c_dst,
        val, elem_size, width, height, pitch)


cdef inline MemcpyNode GN_memcpy(
        GraphNode self, cydriver.CUdeviceptr c_dst,
        cydriver.CUdeviceptr c_src, size_t size):
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

    return MemcpyNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), c_dst, c_src, size,
        c_dst_type, c_src_type)


cdef inline ChildGraphNode GN_embed(GraphNode self, GraphDef child_def):
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

    return ChildGraphNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), h_embedded)


cdef inline EventRecordNode GN_record_event(GraphNode self, Event ev):
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

    return EventRecordNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), ev._h_event)


cdef inline EventWaitNode GN_wait_event(GraphNode self, Event ev):
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

    return EventWaitNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), ev._h_event)


cdef inline HostCallbackNode GN_callback(GraphNode self, object fn, object user_data):
    import ctypes as ct

    cdef cydriver.CUDA_HOST_NODE_PARAMS node_params
    cdef cydriver.CUgraphNode new_node = NULL
    cdef GraphHandle h_graph = graph_node_get_graph(self._h_node)
    cdef cydriver.CUgraphNode pred_node = as_cu(self._h_node)
    cdef cydriver.CUgraphNode* deps = NULL
    cdef size_t num_deps = 0

    if pred_node != NULL:
        deps = &pred_node
        num_deps = 1

    _attach_host_callback_to_graph(
        as_cu(h_graph), fn, user_data,
        &node_params.fn, &node_params.userData)

    with nogil:
        HANDLE_RETURN(cydriver.cuGraphAddHostNode(
            &new_node, as_cu(h_graph), deps, num_deps, &node_params))

    cdef object callable_obj = fn if not isinstance(fn, ct._CFuncPtr) else None
    return HostCallbackNode._create_with_params(
        create_graph_node_handle(new_node, h_graph), callable_obj,
        node_params.fn, node_params.userData)
