# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GraphNode subclasses — EmptyNode through SwitchNode."""

from __future__ import annotations

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver

from cuda.core._event cimport Event
from cuda.core._launch_config cimport LaunchConfig
from cuda.core._module cimport Kernel
from cuda.core.graph._graph_def cimport Condition, GraphDef
from cuda.core.graph._graph_node cimport GraphNode
from cuda.core._resource_handles cimport (
    EventHandle,
    GraphHandle,
    KernelHandle,
    GraphNodeHandle,
    as_cu,
    as_intptr,
    create_event_handle_ref,
    create_graph_handle_ref,
    create_kernel_handle_ref,
    create_graph_node_handle,
    graph_node_get_graph,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core.graph._utils cimport _is_py_host_trampoline

from cuda.core._utils.cuda_utils import driver, handle_return


cdef bint _has_cuGraphNodeGetParams = False
cdef bint _version_checked = False

cdef bint _check_node_get_params():
    global _has_cuGraphNodeGetParams, _version_checked
    if not _version_checked:
        from cuda.core._utils.version import driver_version
        _has_cuGraphNodeGetParams = driver_version() >= (13, 2, 0)
        _version_checked = True
    return _has_cuGraphNodeGetParams


cdef class EmptyNode(GraphNode):
    """An empty (synchronization) node."""

    @staticmethod
    cdef EmptyNode _create_impl(GraphNodeHandle h_node):
        cdef EmptyNode n = EmptyNode.__new__(EmptyNode)
        n._h_node = h_node
        return n

    def __repr__(self) -> str:
        return f"<EmptyNode handle=0x{as_intptr(self._h_node):x}>"


cdef class KernelNode(GraphNode):
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
        return (f"<KernelNode handle=0x{as_intptr(self._h_node):x}"
                f" kernel=0x{as_intptr(self._h_kernel):x}>")

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


cdef class AllocNode(GraphNode):
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
        return (f"<AllocNode handle=0x{as_intptr(self._h_node):x}"
                f" dptr=0x{self._dptr:x} size={self._bytesize}>")

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
    def options(self):
        """A GraphAllocOptions reconstructed from this node's parameters."""
        from cuda.core.graph._graph_def import GraphAllocOptions
        return GraphAllocOptions(
            device=self._device_id,
            memory_type=self._memory_type,
            peer_access=list(self._peer_access) if self._peer_access else None,
        )


cdef class FreeNode(GraphNode):
    """A memory deallocation node.

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
        return f"<FreeNode handle=0x{as_intptr(self._h_node):x} dptr=0x{self._dptr:x}>"

    @property
    def dptr(self) -> int:
        """The device pointer being freed."""
        return self._dptr


cdef class MemsetNode(GraphNode):
    """A memset node.

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
        return (f"<MemsetNode handle=0x{as_intptr(self._h_node):x}"
                f" dptr=0x{self._dptr:x} value={self._value}>")

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


cdef class MemcpyNode(GraphNode):
    """A memcpy node.

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
        return (f"<MemcpyNode handle=0x{as_intptr(self._h_node):x}"
                f" dst=0x{self._dst:x}({dt}) src=0x{self._src:x}({st}) size={self._size}>")

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


cdef class ChildGraphNode(GraphNode):
    """A child graph node.

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
        return (f"<ChildGraphNode handle=0x{as_intptr(self._h_node):x}"
                f" child=0x{as_intptr(self._h_child_graph):x}>")

    @property
    def child_graph(self) -> "GraphDef":
        """The embedded graph definition (non-owning wrapper)."""
        return GraphDef._from_handle(self._h_child_graph)


cdef class EventRecordNode(GraphNode):
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
        return (f"<EventRecordNode handle=0x{as_intptr(self._h_node):x}"
                f" event=0x{as_intptr(self._h_event):x}>")

    @property
    def event(self) -> Event:
        """The event being recorded."""
        return Event._from_handle(self._h_event)


cdef class EventWaitNode(GraphNode):
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
        return (f"<EventWaitNode handle=0x{as_intptr(self._h_node):x}"
                f" event=0x{as_intptr(self._h_event):x}>")

    @property
    def event(self) -> Event:
        """The event being waited on."""
        return Event._from_handle(self._h_event)


cdef class HostCallbackNode(GraphNode):
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
        if _is_py_host_trampoline(params.fn):
            callable_obj = <object>params.userData

        return HostCallbackNode._create_with_params(
            h_node, callable_obj, params.fn, params.userData)

    def __repr__(self) -> str:
        if self._callable is not None:
            name = getattr(self._callable, '__name__', '?')
            return (f"<HostCallbackNode handle=0x{as_intptr(self._h_node):x}"
                    f" callback={name}>")
        return (f"<HostCallbackNode handle=0x{as_intptr(self._h_node):x}"
                f" cfunc=0x{<uintptr_t>self._fn:x}>")

    @property
    def callback_fn(self):
        """The Python callable, or None for ctypes function pointer callbacks."""
        return self._callable


cdef class ConditionalNode(GraphNode):
    """Base class for conditional nodes.

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
        return f"<ConditionalNode handle=0x{as_intptr(self._h_node):x}>"

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
    """An if-conditional node."""

    def __repr__(self) -> str:
        return (f"<IfNode handle=0x{as_intptr(self._h_node):x}"
                f" condition=0x{<unsigned long long>self._condition._c_handle:x}>")

    @property
    def then(self) -> "GraphDef":
        """The 'then' branch graph."""
        return self._branches[0]


cdef class IfElseNode(ConditionalNode):
    """An if-else conditional node."""

    def __repr__(self) -> str:
        return (f"<IfElseNode handle=0x{as_intptr(self._h_node):x}"
                f" condition=0x{<unsigned long long>self._condition._c_handle:x}>")

    @property
    def then(self) -> "GraphDef":
        """The 'then' branch graph (executed when condition is non-zero)."""
        return self._branches[0]

    @property
    def else_(self) -> "GraphDef":
        """The 'else' branch graph (executed when condition is zero)."""
        return self._branches[1]


cdef class WhileNode(ConditionalNode):
    """A while-loop conditional node."""

    def __repr__(self) -> str:
        return (f"<WhileNode handle=0x{as_intptr(self._h_node):x}"
                f" condition=0x{<unsigned long long>self._condition._c_handle:x}>")

    @property
    def body(self) -> "GraphDef":
        """The loop body graph."""
        return self._branches[0]


cdef class SwitchNode(ConditionalNode):
    """A switch conditional node."""

    def __repr__(self) -> str:
        return (f"<SwitchNode handle=0x{as_intptr(self._h_node):x}"
                f" condition=0x{<unsigned long long>self._condition._c_handle:x}>")
