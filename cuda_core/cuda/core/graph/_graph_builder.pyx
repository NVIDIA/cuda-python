# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from libc.stdint cimport intptr_t

from cuda.bindings cimport cydriver

from cuda.core.graph._graph_definition cimport GraphCondition
from cuda.core.graph._utils cimport _attach_host_callback_to_graph
from cuda.core._resource_handles cimport (
    GraphExecHandle, GraphHandle, StreamHandle, as_cu, as_py,
    create_graph_exec_handle, create_graph_handle, create_graph_handle_ref,
)
from cuda.core._stream cimport Stream
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.version cimport cy_binding_version, cy_driver_version

from cuda.core._utils.cuda_utils import (
    CUDAError,
    driver,
    handle_return,
)

__all__ = ['Graph', 'GraphBuilder', 'GraphCompleteOptions', 'GraphDebugPrintOptions']


@dataclass
class GraphDebugPrintOptions:
    """Options for debug_dot_print().

    Attributes
    ----------
    verbose : bool
        Output all debug data as if every debug flag is enabled (Default to False)
    runtime_types : bool
        Use CUDA Runtime structures for output (Default to False)
    kernel_node_params : bool
        Adds kernel parameter values to output (Default to False)
    memcpy_node_params : bool
        Adds memcpy parameter values to output (Default to False)
    memset_node_params : bool
        Adds memset parameter values to output (Default to False)
    host_node_params : bool
        Adds host parameter values to output (Default to False)
    event_node_params : bool
        Adds event parameter values to output (Default to False)
    ext_semas_signal_node_params : bool
        Adds external semaphore signal parameter values to output (Default to False)
    ext_semas_wait_node_params : bool
        Adds external semaphore wait parameter values to output (Default to False)
    kernel_node_attributes : bool
        Adds kernel node attributes to output (Default to False)
    handles : bool
        Adds node handles and every kernel function handle to output (Default to False)
    mem_alloc_node_params : bool
        Adds memory alloc parameter values to output (Default to False)
    mem_free_node_params : bool
        Adds memory free parameter values to output (Default to False)
    batch_mem_op_node_params : bool
        Adds batch mem op parameter values to output (Default to False)
    extra_topo_info : bool
        Adds edge numbering information (Default to False)
    conditional_node_params : bool
        Adds conditional node parameter values to output (Default to False)

    """

    verbose: bool = False
    runtime_types: bool = False
    kernel_node_params: bool = False
    memcpy_node_params: bool = False
    memset_node_params: bool = False
    host_node_params: bool = False
    event_node_params: bool = False
    ext_semas_signal_node_params: bool = False
    ext_semas_wait_node_params: bool = False
    kernel_node_attributes: bool = False
    handles: bool = False
    mem_alloc_node_params: bool = False
    mem_free_node_params: bool = False
    batch_mem_op_node_params: bool = False
    extra_topo_info: bool = False
    conditional_node_params: bool = False

    def _to_flags(self) -> int:
        """Convert options to CUDA driver API flags (internal use)."""
        flags = 0
        if self.verbose:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
        if self.runtime_types:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES
        if self.kernel_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
        if self.memcpy_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS
        if self.memset_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS
        if self.host_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS
        if self.event_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS
        if self.ext_semas_signal_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS
        if self.ext_semas_wait_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS
        if self.kernel_node_attributes:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES
        if self.handles:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES
        if self.mem_alloc_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS
        if self.mem_free_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS
        if self.batch_mem_op_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS
        if self.extra_topo_info:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO
        if self.conditional_node_params:
            flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS
        return flags


@dataclass
class GraphCompleteOptions:
    """Options for graph instantiation.

    Attributes
    ----------
    auto_free_on_launch : bool, optional
        Automatically free memory allocated in a graph before relaunching. (Default to False)
    upload_stream : Stream, optional
        Stream to use to automatically upload the graph after completion. (Default to None)
    device_launch : bool, optional
        Configure the graph to be launchable from the device. This flag can only
        be used on platforms which support unified addressing. This flag cannot be
        used in conjunction with auto_free_on_launch. (Default to False)
    use_node_priority : bool, optional
        Run the graph using the per-node priority attributes rather than the
        priority of the stream it is launched into. (Default to False)

    """

    auto_free_on_launch: bool = False
    upload_stream: Stream | None = None
    device_launch: bool = False
    use_node_priority: bool = False


def _instantiate_graph(h_graph, options: GraphCompleteOptions | None = None) -> Graph:
    cdef cydriver.CUgraphExec c_exec
    params = driver.CUDA_GRAPH_INSTANTIATE_PARAMS()
    if options:
        flags = 0
        if options.auto_free_on_launch:
            flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
        if options.upload_stream:
            flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD
            params.hUploadStream = options.upload_stream.handle
        if options.device_launch:
            flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
        if options.use_node_priority:
            flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
        params.flags = flags

    py_exec = handle_return(driver.cuGraphInstantiateWithParams(h_graph, params))
    c_exec = <cydriver.CUgraphExec><intptr_t>int(py_exec)
    graph = Graph._init(c_exec)
    if params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_ERROR:
        raise RuntimeError(
            "Instantiation failed for an unexpected reason which is described in the return value of the function."
        )
    elif params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE:
        raise RuntimeError("Instantiation failed due to invalid structure, such as cycles.")
    elif params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED:
        raise RuntimeError(
            "Instantiation for device launch failed because the graph contained an unsupported operation."
        )
    elif params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED:
        raise RuntimeError("Instantiation for device launch failed due to the nodes belonging to different contexts.")
    elif (
        cy_binding_version() >= (12, 8, 0)
        and params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED
    ):
        raise RuntimeError("One or more conditional handles are not associated with conditional builders.")
    elif params.result_out != driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_SUCCESS:
        raise RuntimeError(f"Graph instantiation failed with unexpected error code: {params.result_out}")
    return graph


# Distinguishes the three kinds of GraphBuilder, which differ in how they
# begin/end stream capture and whether they own the resulting CUgraph.
# Each kind progresses through _CaptureState as follows:
#
#   PRIMARY:          NOT_STARTED -> CAPTURING -> ENDED
#   FORKED:           CAPTURING (never transitions; joined and closed)
#   CONDITIONAL_BODY: NOT_STARTED -> CAPTURING -> ENDED
#
cdef enum _BuilderKind:
    # PRIMARY: The top-level builder created by Device or Stream. Owns the
    # captured CUgraph via an owning GraphHandle. Progresses through all three
    # capture states; responsible for ending capture if destroyed early.
    PRIMARY = 0
    # FORKED: Created by split(). Captures on a private stream forked from the
    # primary. Starts in CAPTURING state and never transitions; the user joins
    # it back to the primary via join(), which closes the builder.  Must NOT
    # call cuStreamEndCapture (the driver requires all forked streams to be
    # joined first).
    FORKED = 1
    # CONDITIONAL_BODY: Created by if_then/if_else/switch/while_loop. Captures
    # into a non-owned body graph via cuStreamBeginCaptureToGraph. The body
    # graph's lifetime is tied to a parent graph. Progresses through all three
    # capture states like PRIMARY.
    CONDITIONAL_BODY = 2


# Tracks the capture lifecycle of a GraphBuilder.
cdef enum _CaptureState:
    CAPTURE_NOT_STARTED = 0
    CAPTURING = 1
    CAPTURE_ENDED = 2


cdef class GraphBuilder:
    """A graph under construction by stream capture.

    A graph groups a set of CUDA kernels and other CUDA operations together and executes
    them with a specified dependency tree. It speeds up the workflow by combining the
    driver activities associated with CUDA kernel launches and CUDA API calls.

    Directly creating a :obj:`~graph.GraphBuilder` is not supported due
    to ambiguity. New graph builders should instead be created through a
    :obj:`~_device.Device`, or a :obj:`~_stream.stream` object.

    """

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Graph object can be ambiguous. Please either "
            "call Device.create_graph_builder() or stream.create_graph_builder()"
        )

    def __dealloc__(self):
        # Note: _stream could be set to None by cyclic-GC tp_clear before
        # __dealloc__, but _h_stream is guaranteed to be valid.
        if self._h_stream and self._state == CAPTURING and self._kind != FORKED:
            with nogil:
                cydriver.cuStreamEndCapture(as_cu(self._h_stream), NULL)

    @staticmethod
    cdef GraphBuilder _init(Stream stream):
        cdef GraphBuilder self = GraphBuilder.__new__(GraphBuilder)
        # _h_graph set by begin_building
        self._h_stream = stream._h_stream
        self._kind = PRIMARY
        self._state = CAPTURE_NOT_STARTED
        self._stream = stream
        return self

    def close(self):
        """Destroy the graph builder."""
        if self._h_stream and self._state == CAPTURING and self._kind != FORKED:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamEndCapture(as_cu(self._h_stream), NULL))
        self._h_graph.reset()
        self._h_stream.reset()
        self._state = CAPTURE_ENDED
        self._stream = None

    @property
    def stream(self) -> Stream:
        """Returns the stream associated with the graph builder."""
        return self._stream

    @property
    def is_join_required(self) -> bool:
        """Returns True if this graph builder must be joined before building is ended."""
        return self._kind == FORKED

    def begin_building(self, mode="relaxed") -> GraphBuilder:
        """Begins the building process.

        Build `mode` for controlling interaction with other API calls must be one of the following:

        - `global` : Prohibit potentially unsafe operations across all streams in the process.
        - `thread_local` : Prohibit potentially unsafe operations in streams created by the current thread.
        - `relaxed` : The local thread is not prohibited from potentially unsafe operations.

        Parameters
        ----------
        mode : str, optional
            Build mode to control the interaction with other API calls that are porentially unsafe.
            Default set to use relaxed.

        """
        if self._state != CAPTURE_NOT_STARTED:
            if self._state == CAPTURING:
                raise RuntimeError("Graph builder is already building.")
            else:
                raise RuntimeError("Cannot resume building after building has ended.")
        cdef cydriver.CUstreamCaptureMode c_mode
        if mode == "global":
            c_mode = cydriver.CU_STREAM_CAPTURE_MODE_GLOBAL
        elif mode == "thread_local":
            c_mode = cydriver.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        elif mode == "relaxed":
            c_mode = cydriver.CU_STREAM_CAPTURE_MODE_RELAXED
        else:
            raise ValueError(f"Unsupported build mode: {mode}")

        cdef cydriver.CUstream c_stream = as_cu(self._h_stream)
        cdef cydriver.CUgraph c_graph
        if self._kind == CONDITIONAL_BODY:
            c_graph = as_cu(self._h_graph)
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamBeginCaptureToGraph(
                    c_stream, c_graph, NULL, NULL, 0, c_mode))
        else:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamBeginCapture(c_stream, c_mode))
                _get_capture_info(c_stream, NULL, &c_graph)
            self._h_graph = create_graph_handle(c_graph)
        self._state = CAPTURING
        return self

    @property
    def is_building(self) -> bool:
        """Returns True if the graph builder is currently building."""
        cdef cydriver.CUstream c_stream = as_cu(self._h_stream)
        cdef cydriver.CUstreamCaptureStatus status
        with nogil:
            _get_capture_info(c_stream, &status, NULL)
        if status == cydriver.CU_STREAM_CAPTURE_STATUS_NONE:
            return False
        elif status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            return True
        elif status == cydriver.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
            raise RuntimeError(
                "Build process encountered an error and has been invalidated. Build process must now be ended."
            )
        else:
            raise NotImplementedError(f"Unsupported capture status type received: {status}")

    def end_building(self) -> GraphBuilder:
        """Ends the building process."""
        if not self.is_building:
            raise RuntimeError("Graph builder is not building.")
        cdef cydriver.CUstream c_stream = as_cu(self._h_stream)
        with nogil:
            HANDLE_RETURN(cydriver.cuStreamEndCapture(c_stream, NULL))

        # TODO: Resolving https://github.com/NVIDIA/cuda-python/issues/617 would allow us to
        #       resume the build process after the first call to end_building()
        self._state = CAPTURE_ENDED
        return self

    def complete(self, options: GraphCompleteOptions | None = None) -> Graph:
        """Completes the graph builder and returns the built :obj:`~graph.Graph` object.

        Parameters
        ----------
        options : :obj:`~graph.GraphCompleteOptions`, optional
            Customizable dataclass for the graph builder completion options.

        Returns
        -------
        graph : :obj:`~graph.Graph`
            The newly built graph.

        """
        if self._state != CAPTURE_ENDED:
            raise RuntimeError("Graph has not finished building.")

        return _instantiate_graph(as_py(self._h_graph), options)

    def debug_dot_print(self, path, options: GraphDebugPrintOptions | None = None):
        """Generates a DOT debug file for the graph builder.

        Parameters
        ----------
        path : str
            File path to use for writting debug DOT output
        options : :obj:`~graph.GraphDebugPrintOptions`, optional
            Customizable dataclass for the debug print options.

        """
        if self._state != CAPTURE_ENDED:
            raise RuntimeError("Graph has not finished building.")
        cdef unsigned int c_flags = options._to_flags() if options else 0
        cdef cydriver.CUgraph c_graph = as_cu(self._h_graph)
        cdef bytes b_path = path.encode() if isinstance(path, str) else path
        cdef const char* c_path = b_path
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphDebugDotPrint(c_graph, c_path, c_flags))

    def split(self, count: int) -> tuple[GraphBuilder, ...]:
        """Splits the original graph builder into multiple graph builders.

        The new builders inherit work dependencies from the original builder.
        The original builder is reused for the split and is returned first in the tuple.

        Parameters
        ----------
        count : int
            The number of graph builders to split the graph builder into.

        Returns
        -------
        graph_builders : tuple[:obj:`~graph.GraphBuilder`, ...]
            A tuple of split graph builders. The first graph builder in the tuple
            is always the original graph builder.

        """
        if count < 2:
            raise ValueError(f"Invalid split count: expecting >= 2, got {count}")

        event = self._stream.record()
        result = [self]
        for i in range(count - 1):
            stream = self._stream.device.create_stream()
            stream.wait(event)
            result.append(_init_forked(stream))
        event.close()
        return tuple(result)

    @staticmethod
    def join(*graph_builders) -> GraphBuilder:
        """Joins multiple graph builders into a single graph builder.

        The returned builder inherits work dependencies from the provided builders.

        Parameters
        ----------
        *graph_builders : :obj:`~graph.GraphBuilder`
            The graph builders to join.

        Returns
        -------
        graph_builder : :obj:`~graph.GraphBuilder`
            The newly joined graph builder.

        """
        if any(not isinstance(builder, GraphBuilder) for builder in graph_builders):
            raise TypeError("All arguments must be GraphBuilder instances")
        if len(graph_builders) < 2:
            raise ValueError("Must join with at least two graph builders")

        # Discover the root builder others should join
        root_idx = 0
        for i, builder in enumerate(graph_builders):
            if not builder.is_join_required:
                root_idx = i
                break

        # Join all onto the root builder
        root_bdr = graph_builders[root_idx]
        for idx, builder in enumerate(graph_builders):
            if idx == root_idx:
                continue
            root_bdr.stream.wait(builder.stream)
            builder.close()

        return root_bdr

    def __cuda_stream__(self) -> tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return self.stream.__cuda_stream__()

    def _get_conditional_context(self) -> driver.CUcontext:
        return self._stream.context.handle

    def create_condition(self, default_value=None) -> GraphCondition:
        """Create a condition variable for use with conditional nodes.

        The returned :class:`GraphCondition` object is passed to conditional-node
        builder methods (:meth:`if_then`, :meth:`if_else`, :meth:`while_loop`,
        :meth:`switch`). Its value is controlled at runtime by device code via
        ``cudaGraphSetConditional``.

        Parameters
        ----------
        default_value : int, optional
            The default value to assign to the condition. If None, no
            default is assigned.

        Returns
        -------
        GraphCondition
            A condition variable for controlling conditional execution.
        """
        if cy_driver_version() < (12, 3, 0):
            raise RuntimeError(f"Driver version {'.'.join(map(str, cy_driver_version()))} does not support conditional handles")
        if cy_binding_version() < (12, 3, 0):
            raise RuntimeError(f"Binding version {'.'.join(map(str, cy_binding_version()))} does not support conditional handles")
        if default_value is not None:
            flags = driver.CU_GRAPH_COND_ASSIGN_DEFAULT
        else:
            default_value = 0
            flags = 0

        status, _, graph, *_, _ = handle_return(driver.cuStreamGetCaptureInfo(self._stream.handle))
        if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            raise RuntimeError("Cannot create a condition when graph is not being built")

        raw_handle = handle_return(
            driver.cuGraphConditionalHandleCreate(graph, self._get_conditional_context(), default_value, flags)
        )
        return GraphCondition._from_handle(<cydriver.CUgraphConditionalHandle><intptr_t>int(raw_handle))

    def if_then(self, condition: GraphCondition) -> GraphBuilder:
        """Adds an if condition branch and returns a new graph builder for it.

        The resulting if graph will only execute the branch if the
        condition evaluates to true at runtime.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        condition : :class:`~graph.GraphCondition`
            The condition variable from :meth:`create_condition` controlling
            whether the branch executes.

        Returns
        -------
        graph_builder : :obj:`~graph.GraphBuilder`
            The newly created conditional graph builder.

        """
        if cy_driver_version() < (12, 3, 0):
            raise RuntimeError(f"Driver version {'.'.join(map(str, cy_driver_version()))} does not support conditional if")
        if cy_binding_version() < (12, 3, 0):
            raise RuntimeError(f"Binding version {'.'.join(map(str, cy_binding_version()))} does not support conditional if")
        if not isinstance(condition, GraphCondition):
            raise TypeError(
                f"condition must be a GraphCondition object (from "
                f"GraphBuilder.create_condition()), got {type(condition).__name__}")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = condition.handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 1
        node_params.conditional.ctx = self._get_conditional_context()
        return _cond_with_params(self, node_params)[0]

    def if_else(self, condition: GraphCondition) -> tuple[GraphBuilder, GraphBuilder]:
        """Adds an if-else condition branch and returns new graph builders for both branches.

        The resulting if graph will execute the branch if the condition
        evaluates to true at runtime, otherwise the else branch will execute.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        condition : :class:`~graph.GraphCondition`
            The condition variable from :meth:`create_condition` controlling
            which branch executes.

        Returns
        -------
        graph_builders : tuple[:obj:`~graph.GraphBuilder`, :obj:`~graph.GraphBuilder`]
            A tuple of two new graph builders, one for the if branch and one for the else branch.

        """
        if cy_driver_version() < (12, 8, 0):
            raise RuntimeError(f"Driver version {'.'.join(map(str, cy_driver_version()))} does not support conditional if-else")
        if cy_binding_version() < (12, 8, 0):
            raise RuntimeError(f"Binding version {'.'.join(map(str, cy_binding_version()))} does not support conditional if-else")
        if not isinstance(condition, GraphCondition):
            raise TypeError(
                f"condition must be a GraphCondition object (from "
                f"GraphBuilder.create_condition()), got {type(condition).__name__}")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = condition.handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 2
        node_params.conditional.ctx = self._get_conditional_context()
        return _cond_with_params(self, node_params)

    def switch(self, condition: GraphCondition, count: int) -> tuple[GraphBuilder, ...]:
        """Adds a switch condition branch and returns new graph builders for all cases.

        The resulting switch graph will execute the branch whose case index
        matches the value of the condition at runtime. If no match is found, no
        branch will be executed.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        condition : :class:`~graph.GraphCondition`
            The condition variable from :meth:`create_condition` selecting
            which case executes.
        count : int
            The number of cases to add to the switch conditional.

        Returns
        -------
        graph_builders : tuple[:obj:`~graph.GraphBuilder`, ...]
            A tuple of new graph builders, one for each branch.

        """
        if cy_driver_version() < (12, 8, 0):
            raise RuntimeError(f"Driver version {'.'.join(map(str, cy_driver_version()))} does not support conditional switch")
        if cy_binding_version() < (12, 8, 0):
            raise RuntimeError(f"Binding version {'.'.join(map(str, cy_binding_version()))} does not support conditional switch")
        if not isinstance(condition, GraphCondition):
            raise TypeError(
                f"condition must be a GraphCondition object (from "
                f"GraphBuilder.create_condition()), got {type(condition).__name__}")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = condition.handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_SWITCH
        node_params.conditional.size = count
        node_params.conditional.ctx = self._get_conditional_context()
        return _cond_with_params(self, node_params)

    def while_loop(self, condition: GraphCondition) -> GraphBuilder:
        """Adds a while loop and returns a new graph builder for it.

        The resulting while loop graph will execute the branch repeatedly at runtime
        until the condition evaluates to false.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        condition : :class:`~graph.GraphCondition`
            The condition variable from :meth:`create_condition` controlling
            loop continuation.

        Returns
        -------
        graph_builder : :obj:`~graph.GraphBuilder`
            The newly created while loop graph builder.

        """
        if cy_driver_version() < (12, 3, 0):
            raise RuntimeError(f"Driver version {'.'.join(map(str, cy_driver_version()))} does not support conditional while loop")
        if cy_binding_version() < (12, 3, 0):
            raise RuntimeError(f"Binding version {'.'.join(map(str, cy_binding_version()))} does not support conditional while loop")
        if not isinstance(condition, GraphCondition):
            raise TypeError(
                f"condition must be a GraphCondition object (from "
                f"GraphBuilder.create_condition()), got {type(condition).__name__}")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = condition.handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        node_params.conditional.size = 1
        node_params.conditional.ctx = self._get_conditional_context()
        return _cond_with_params(self, node_params)[0]

    def embed(self, GraphBuilder child):
        """Embed a previously-built :obj:`~graph.GraphBuilder` as a child node.

        Parameters
        ----------
        child : :obj:`~graph.GraphBuilder`
            The child graph builder. Must have finished building.
        """
        if child._state != CAPTURE_ENDED:
            raise ValueError("Child graph has not finished building.")

        if not self.is_building:
            raise ValueError("Parent graph is not being built.")

        stream_handle = self._stream.handle
        _, _, graph_out, *deps_info_out, num_dependencies_out = handle_return(
            driver.cuStreamGetCaptureInfo(stream_handle)
        )

        # See https://github.com/NVIDIA/cuda-python/pull/879#issuecomment-3211054159
        # for rationale
        deps_info_trimmed = deps_info_out[:num_dependencies_out]
        deps_info_update = [
            [
                handle_return(
                    driver.cuGraphAddChildGraphNode(
                        graph_out, *deps_info_trimmed, num_dependencies_out, as_py(child._h_graph)
                    )
                )
            ]
        ] + [None] * (len(deps_info_out) - 1)
        handle_return(
            driver.cuStreamUpdateCaptureDependencies(
                stream_handle,
                *deps_info_update,  # dependencies, edgeData
                1,
                driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
            )
        )

    def callback(self, fn, *, user_data=None):
        """Add a host callback to the graph during stream capture.

        The callback runs on the host CPU when the graph reaches this point
        in execution. Two modes are supported:

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
        """
        cdef Stream stream = self._stream
        cdef cydriver.CUstream c_stream = as_cu(stream._h_stream)
        cdef cydriver.CUstreamCaptureStatus capture_status
        cdef cydriver.CUgraph c_graph = NULL

        with nogil:
            _get_capture_info(c_stream, &capture_status, &c_graph)

        if capture_status != cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            raise RuntimeError("Cannot add callback when graph is not being built")

        cdef cydriver.CUhostFn c_fn
        cdef void* c_user_data = NULL
        _attach_host_callback_to_graph(c_graph, fn, user_data, &c_fn, &c_user_data)

        with nogil:
            HANDLE_RETURN(cydriver.cuLaunchHostFunc(c_stream, c_fn, c_user_data))


cdef inline GraphBuilder _init_forked(Stream stream):
    cdef GraphBuilder gb = GraphBuilder.__new__(GraphBuilder)
    # _h_graph not used for FORKED builders. Captures to primary graph.
    gb._h_stream = stream._h_stream
    gb._kind = FORKED
    gb._state = CAPTURING
    gb._stream = stream
    return gb


cdef inline GraphBuilder _init_conditional(Stream stream, cydriver.CUgraph cond_graph, GraphBuilder parent):
    cdef GraphBuilder gb = GraphBuilder.__new__(GraphBuilder)
    gb._h_graph = create_graph_handle_ref(cond_graph, parent._h_graph)
    gb._h_stream = stream._h_stream
    gb._kind = CONDITIONAL_BODY
    gb._state = CAPTURE_NOT_STARTED
    gb._stream = stream
    return gb


cdef inline int _get_capture_info(
        cydriver.CUstream stream,
        cydriver.CUstreamCaptureStatus* status,
        cydriver.CUgraph* graph) except?-1 nogil:
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(
            stream, status, NULL, graph, NULL, NULL, NULL))
    ELSE:
        return HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(
            stream, status, NULL, graph, NULL, NULL))


cdef inline tuple _cond_with_params(GraphBuilder gb, node_params):
    status, _, graph, *deps_info, num_dependencies = handle_return(
        driver.cuStreamGetCaptureInfo(gb._stream.handle)
    )
    if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise RuntimeError("Cannot add conditional node when not actively capturing")

    deps_info_update = [
        [handle_return(driver.cuGraphAddNode(graph, *deps_info, num_dependencies, node_params))]
    ] + [None] * (len(deps_info) - 1)

    handle_return(
        driver.cuStreamUpdateCaptureDependencies(
            gb._stream.handle,
            *deps_info_update,  # dependencies, edgeData
            1,  # numDependencies
            driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
        )
    )

    return tuple(
        _init_conditional(
            gb._stream.device.create_stream(),
            <cydriver.CUgraph><intptr_t>int(node_params.conditional.phGraph_out[i]),
            gb,
        )
        for i in range(node_params.conditional.size)
    )


cdef class Graph:
    """An executable graph.

    A graph groups a set of CUDA kernels and other CUDA operations together and executes
    them with a specified dependency tree. It speeds up the workflow by combining the
    driver activities associated with CUDA kernel launches and CUDA API calls.

    Graphs must be built using a :obj:`~graph.GraphBuilder` object.

    """

    def __init__(self):
        raise RuntimeError("directly constructing a Graph instance is not supported")

    @staticmethod
    cdef Graph _init(cydriver.CUgraphExec graph_exec):
        cdef Graph self = Graph.__new__(Graph)
        self._h_graph_exec = create_graph_exec_handle(graph_exec)
        return self

    def close(self):
        """Destroy the graph."""
        self._h_graph_exec.reset()

    @property
    def handle(self) -> driver.CUgraphExec:
        """Return the underlying ``CUgraphExec`` object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int()`` on the returned object.

        """
        return as_py(self._h_graph_exec)

    def update(self, source: "GraphBuilder | GraphDefinition") -> None:
        """Update the graph using a new graph definition.

        The topology of the provided source must be identical to this graph.

        Parameters
        ----------
        source : :obj:`~graph.GraphBuilder` or :obj:`~graph.GraphDefinition`
            The graph definition to update from. A GraphBuilder must have
            finished building.

        """
        from cuda.core.graph import GraphDefinition

        cdef cydriver.CUgraph cu_graph
        cdef cydriver.CUgraphExec cu_exec = as_cu(self._h_graph_exec)

        if isinstance(source, GraphBuilder):
            if (<GraphBuilder>source)._state != CAPTURE_ENDED:
                raise ValueError("Graph has not finished building.")
            cu_graph = as_cu((<GraphBuilder>source)._h_graph)
        elif isinstance(source, GraphDefinition):
            cu_graph = <cydriver.CUgraph><intptr_t>int(source.handle)
        else:
            raise TypeError(
                f"expected GraphBuilder or GraphDefinition, got {type(source).__name__}")

        cdef cydriver.CUgraphExecUpdateResultInfo result_info
        cdef cydriver.CUresult err
        with nogil:
            err = cydriver.cuGraphExecUpdate(cu_exec, cu_graph, &result_info)
        if err == cydriver.CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            reason = driver.CUgraphExecUpdateResult(result_info.result)
            msg = f"Graph update failed: {reason.__doc__.strip()} ({reason.name})"
            raise CUDAError(msg)
        HANDLE_RETURN(err)

    def upload(self, stream: Stream):
        """Uploads the graph in a stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream`
            The stream in which to upload the graph

        """
        cdef cydriver.CUgraphExec c_exec = as_cu(self._h_graph_exec)
        cdef cydriver.CUstream c_stream = <cydriver.CUstream><intptr_t>int(stream.handle)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphUpload(c_exec, c_stream))

    def launch(self, stream: Stream):
        """Launches the graph in a stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream`
            The stream in which to launch the graph

        """
        cdef cydriver.CUgraphExec c_exec = as_cu(self._h_graph_exec)
        cdef cydriver.CUstream c_stream = <cydriver.CUstream><intptr_t>int(stream.handle)
        with nogil:
            HANDLE_RETURN(cydriver.cuGraphLaunch(c_exec, c_stream))
