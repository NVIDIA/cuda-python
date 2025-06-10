# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils.cuda_utils import (
    driver,
    get_binding_version,
    handle_return,
)

_inited = False
_driver_ver = None


def _lazy_init():
    global _inited
    if _inited:
        return

    global _py_major_minor, _driver_ver
    # binding availability depends on cuda-python version
    _py_major_minor = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    _inited = True


@dataclass
class GraphDebugPrintOptions:
    """Customizable options for :obj:`_graph.GraphBuilder.debug_dot_print()`

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


@dataclass
class GraphCompleteOptions:
    """Customizable options for :obj:`_graph.GraphBuilder.complete()`

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
    upload_stream: Optional[Stream] = None
    device_launch: bool = False
    use_node_priority: bool = False


class GraphBuilder:
    """Represents a graph under construction.

    A graph groups a set of CUDA kernels and other CUDA operations together and executes
    them with a specified dependency tree. It speeds up the workflow by combining the
    driver activities associated with CUDA kernel launches and CUDA API calls.

    Directly creating a :obj:`~_graph.GraphBuilder` is not supported due
    to ambiguity. New graph builders should instead be created through a
    :obj:`~_device.Device`, or a :obj:`~_stream.stream` object.

    """

    class _MembersNeededForFinalize:
        __slots__ = ("stream", "is_stream_owner", "graph", "conditional_graph", "is_join_required")

        def __init__(self, graph_builder_obj, stream_obj, is_stream_owner, conditional_graph, is_join_required):
            self.stream = stream_obj
            self.is_stream_owner = is_stream_owner
            self.graph = None
            self.conditional_graph = conditional_graph
            self.is_join_required = is_join_required
            weakref.finalize(graph_builder_obj, self.close)

        def close(self):
            if self.stream:
                if not self.is_join_required:
                    capture_status = handle_return(driver.cuStreamGetCaptureInfo(self.stream.handle))[0]
                    if capture_status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
                        # Note how this condition only occures for the primary graph builder
                        # This is because calling cuStreamEndCapture streams that were split off of the primary
                        # would error out with CUDA_ERROR_STREAM_CAPTURE_UNJOINED.
                        # Therefore, it is currently a requirement that users join all split graph builders
                        # before a graph builder can be clearly destroyed.
                        handle_return(driver.cuStreamEndCapture(self.stream.handle))
                if self.is_stream_owner:
                    self.stream.close()
            self.stream = None
            if self.graph:
                handle_return(driver.cuGraphDestroy(self.graph))
            self.graph = None
            self.conditional_graph = None

    __slots__ = ("__weakref__", "_mnff", "_building_ended")

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Graph object can be ambiguous. Please either "
            "call Device.create_graph_builder() or stream.create_graph_builder()"
        )

    @classmethod
    def _init(cls, stream, is_stream_owner, conditional_graph=None, is_join_required=False):
        self = cls.__new__(cls)
        _lazy_init()
        self._mnff = GraphBuilder._MembersNeededForFinalize(
            self, stream, is_stream_owner, conditional_graph, is_join_required
        )

        self._building_ended = False
        return self

    @property
    def stream(self) -> Stream:
        """Returns the stream associated with the graph builder."""
        return self._mnff.stream

    @property
    def is_join_required(self) -> bool:
        """Returns True if this graph builder must be joined before building is ended."""
        return self._mnff.is_join_required

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
        if self._building_ended:
            raise RuntimeError("Cannot resume building after building has ended.")
        if mode not in ("global", "thread_local", "relaxed"):
            raise ValueError(f"Unsupported build mode: {mode}")
        if mode == "global":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL
        elif mode == "thread_local":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        elif mode == "relaxed":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        else:
            raise ValueError(f"Unsupported build mode: {mode}")

        if self._mnff.conditional_graph:
            handle_return(
                driver.cuStreamBeginCaptureToGraph(
                    self._mnff.stream.handle,
                    self._mnff.conditional_graph,
                    None,  # dependencies
                    None,  # dependencyData
                    0,  # numDependencies
                    capture_mode,
                )
            )
        else:
            handle_return(driver.cuStreamBeginCapture(self._mnff.stream.handle, capture_mode))
        return self

    @property
    def is_building(self) -> bool:
        """Returns True if the graph builder is currently building."""
        capture_status = handle_return(driver.cuStreamGetCaptureInfo(self._mnff.stream.handle))[0]
        if capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            return False
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            return True
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
            raise RuntimeError(
                "Build process encountered an error and has been invalidated. Build process must now be ended."
            )
        else:
            raise NotImplementedError(f"Unsupported capture status type received: {capture_status}")

    def end_building(self) -> GraphBuilder:
        """Ends the building process."""
        if not self.is_building:
            raise RuntimeError("Graph builder is not building.")
        if self._mnff.conditional_graph:
            self._mnff.conditional_graph = handle_return(driver.cuStreamEndCapture(self.stream.handle))
        else:
            self._mnff.graph = handle_return(driver.cuStreamEndCapture(self.stream.handle))

        # TODO: Resolving https://github.com/NVIDIA/cuda-python/issues/617 would allow us to
        #       resume the build process after the first call to end_building()
        self._building_ended = True
        return self

    def complete(self, options: Optional[GraphCompleteOptions] = None) -> Graph:
        """Completes the graph builder and returns the built :obj:`~_graph.Graph` object.

        Parameters
        ----------
        options : :obj:`~_graph.GraphCompleteOptions`, optional
            Customizable dataclass for the graph builder completion options.

        Returns
        -------
        graph : :obj:`~_graph.Graph`
            The newly built graph.

        """
        if not self._building_ended:
            raise RuntimeError("Graph has not finished building.")

        if (_driver_ver < 12000) or (_py_major_minor < (12, 0)):
            flags = 0
            if options:
                if options.auto_free_on_launch:
                    flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                if options.use_node_priority:
                    flags |= driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
            return Graph._init(handle_return(driver.cuGraphInstantiateWithFlags(self._mnff.graph, flags)))

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

        graph = Graph._init(handle_return(driver.cuGraphInstantiateWithParams(self._mnff.graph, params)))
        if params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_ERROR:
            # NOTE: Should never get here since the handle_return should have caught this case
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
            raise RuntimeError(
                "Instantiation for device launch failed due to the nodes belonging to different contexts."
            )
        elif params.result_out == driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED:
            raise RuntimeError("One or more conditional handles are not associated with conditional builders.")
        elif params.result_out != driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_SUCCESS:
            raise RuntimeError(f"Graph instantiation failed with unexpected error code: {params.result_out}")
        return graph

    def debug_dot_print(self, path, options: Optional[GraphDebugPrintOptions] = None):
        """Generates a DOT debug file for the graph builder.

        Parameters
        ----------
        path : str
            File path to use for writting debug DOT output
        options : :obj:`~_graph.GraphDebugPrintOptions`, optional
            Customizable dataclass for the debug print options.

        """
        if not self._building_ended:
            raise RuntimeError("Graph has not finished building.")
        flags = 0
        if options:
            if options.verbose:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
            if options.runtime_types:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES
            if options.kernel_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
            if options.memcpy_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS
            if options.memset_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS
            if options.host_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS
            if options.event_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS
            if options.ext_semas_signal_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS
            if options.ext_semas_wait_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS
            if options.kernel_node_attributes:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES
            if options.handles:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES
            if options.mem_alloc_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS
            if options.mem_free_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS
            if options.batch_mem_op_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS
            if options.extra_topo_info:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO
            if options.conditional_node_params:
                flags |= driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS

        handle_return(driver.cuGraphDebugDotPrint(self._mnff.graph, path, flags))

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
        graph_builders : tuple[:obj:`~_graph.GraphBuilder`, ...]
            A tuple of split graph builders. The first graph builder in the tuple
            is always the original graph builder.

        """
        if count < 2:
            raise ValueError(f"Invalid split count: expecting >= 2, got {count}")

        event = self._mnff.stream.record()
        result = [self]
        for i in range(count - 1):
            stream = self._mnff.stream.device.create_stream()
            stream.wait(event)
            result.append(
                GraphBuilder._init(stream=stream, is_stream_owner=True, conditional_graph=None, is_join_required=True)
            )
        event.close()
        return result

    @staticmethod
    def join(*graph_builders) -> GraphBuilder:
        """Joins multiple graph builders into a single graph builder.

        The returned builder inherits work dependencies from the provided builders.

        Parameters
        ----------
        *graph_builders : :obj:`~_graph.GraphBuilder`
            The graph builders to join.

        Returns
        -------
        graph_builder : :obj:`~_graph.GraphBuilder`
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
        return self._mnff.stream.context._handle

    def create_conditional_handle(self, default_value=None) -> driver.CUgraphConditionalHandle:
        """Creates a conditional handle for the graph builder.

        Parameters
        ----------
        default_value : int, optional
            The default value to assign to the conditional handle.

        Returns
        -------
        handle : driver.CUgraphConditionalHandle
            The newly created conditional handle.

        """
        if _driver_ver < 12030:
            raise RuntimeError(f"Driver version {_driver_ver} does not support conditional handles")
        if _py_major_minor < (12, 3):
            raise RuntimeError(f"Binding version {_py_major_minor} does not support conditional handles")
        if default_value is not None:
            flags = driver.CU_GRAPH_COND_ASSIGN_DEFAULT
        else:
            default_value = 0
            flags = 0

        status, _, graph, _, _ = handle_return(driver.cuStreamGetCaptureInfo(self._mnff.stream.handle))
        if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            raise RuntimeError("Cannot create a conditional handle when graph is not being built")

        return handle_return(
            driver.cuGraphConditionalHandleCreate(graph, self._get_conditional_context(), default_value, flags)
        )

    def _cond_with_params(self, node_params) -> GraphBuilder:
        # Get current capture info to ensure we're in a valid state
        status, _, graph, dependencies, num_dependencies = handle_return(
            driver.cuStreamGetCaptureInfo(self._mnff.stream.handle)
        )
        if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            raise RuntimeError("Cannot add conditional node when not actively capturing")

        # Add the conditional node to the graph
        node = handle_return(driver.cuGraphAddNode(graph, dependencies, num_dependencies, node_params))

        # Update the stream's capture dependencies
        handle_return(
            driver.cuStreamUpdateCaptureDependencies(
                self._mnff.stream.handle,
                [node],  # dependencies
                1,  # numDependencies
                driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
            )
        )

        # Create new graph builders for each condition
        return tuple(
            [
                GraphBuilder._init(
                    stream=self._mnff.stream.device.create_stream(),
                    is_stream_owner=True,
                    conditional_graph=node_params.conditional.phGraph_out[i],
                    is_join_required=False,
                )
                for i in range(node_params.conditional.size)
            ]
        )

    def if_cond(self, handle: driver.CUgraphConditionalHandle) -> GraphBuilder:
        """Adds an if condition branch and returns a new graph builder for it.

        The resulting if graph will only execute the branch if the conditional
        handle evaluates to true at runtime.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        handle : driver.CUgraphConditionalHandle
            The handle to use for the if conditional.

        Returns
        -------
        graph_builder : :obj:`~_graph.GraphBuilder`
            The newly created conditional graph builder.

        """
        if _driver_ver < 12030:
            raise RuntimeError(f"Driver version {_driver_ver} does not support conditional if")
        if _py_major_minor < (12, 3):
            raise RuntimeError(f"Binding version {_py_major_minor} does not support conditional if")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 1
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)[0]

    def if_else(self, handle: driver.CUgraphConditionalHandle) -> tuple[GraphBuilder, GraphBuilder]:
        """Adds an if-else condition branch and returns new graph builders for both branches.

        The resulting if graph will execute the branch if the conditional handle
        evaluates to true at runtime, otherwise the else branch will execute.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        handle : driver.CUgraphConditionalHandle
            The handle to use for the if-else conditional.

        Returns
        -------
        graph_builders : tuple[:obj:`~_graph.GraphBuilder`, :obj:`~_graph.GraphBuilder`]
            A tuple of two new graph builders, one for the if branch and one for the else branch.

        """
        if _driver_ver < 12080:
            raise RuntimeError(f"Driver version {_driver_ver} does not support conditional if-else")
        if _py_major_minor < (12, 8):
            raise RuntimeError(f"Binding version {_py_major_minor} does not support conditional if-else")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 2
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)

    def switch(self, handle: driver.CUgraphConditionalHandle, count: int) -> tuple[GraphBuilder, ...]:
        """Adds a switch condition branch and returns new graph builders for all cases.

        The resulting switch graph will execute the branch that matches the
        case index of the conditional handle at runtime. If no match is found, no branch
        will be executed.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        handle : driver.CUgraphConditionalHandle
            The handle to use for the switch conditional.
        count : int
            The number of cases to add to the switch conditional.

        Returns
        -------
        graph_builders : tuple[:obj:`~_graph.GraphBuilder`, ...]
            A tuple of new graph builders, one for each branch.

        """
        if _driver_ver < 12080:
            raise RuntimeError(f"Driver version {_driver_ver} does not support conditional switch")
        if _py_major_minor < (12, 8):
            raise RuntimeError(f"Binding version {_py_major_minor} does not support conditional switch")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_SWITCH
        node_params.conditional.size = count
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)

    def while_loop(self, handle: driver.CUgraphConditionalHandle) -> GraphBuilder:
        """Adds a while loop and returns a new graph builder for it.

        The resulting while loop graph will execute the branch repeatedly at runtime
        until the conditional handle evaluates to false.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        handle : driver.CUgraphConditionalHandle
            The handle to use for the while loop.

        Returns
        -------
        graph_builder : :obj:`~_graph.GraphBuilder`
            The newly created while loop graph builder.

        """
        if _driver_ver < 12030:
            raise RuntimeError(f"Driver version {_driver_ver} does not support conditional while loop")
        if _py_major_minor < (12, 3):
            raise RuntimeError(f"Binding version {_py_major_minor} does not support conditional while loop")
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        node_params.conditional.size = 1
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)[0]

    def close(self):
        """Destroy the graph builder.

        Closes the associated stream if we own it. Borrowed stream
        object will instead have their references released.

        """
        self._mnff.close()

    def add_child(self, child_graph: GraphBuilder):
        """Adds the child :obj:`~_graph.GraphBuilder` builder into self.

        The child graph builder will be added as a child node to the parent graph builder.

        Parameters
        ----------
        child_graph : :obj:`~_graph.GraphBuilder`
            The child graph builder. Must have finished building.
        """
        if (_driver_ver < 12000) or (_py_major_minor < (12, 0)):
            raise NotImplementedError(
                f"Launching child graphs is not implemented for versions older than CUDA 12."
                f"Found driver version is {_driver_ver} and binding version is {_py_major_minor}"
            )

        if not child_graph._building_ended:
            raise ValueError("Child graph has not finished building.")

        if not self.is_building:
            raise ValueError("Parent graph is not being built.")

        stream_handle = self._mnff.stream.handle
        _, _, graph_out, dependencies_out, num_dependencies_out = handle_return(
            driver.cuStreamGetCaptureInfo(stream_handle)
        )

        child_node = handle_return(
            driver.cuGraphAddChildGraphNode(graph_out, dependencies_out, num_dependencies_out, child_graph._mnff.graph)
        )
        handle_return(
            driver.cuStreamUpdateCaptureDependencies(
                stream_handle,
                [child_node],
                1,
                driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
            )
        )


class Graph:
    """Represents an executable graph.

    A graph groups a set of CUDA kernels and other CUDA operations together and executes
    them with a specified dependency tree. It speeds up the workflow by combining the
    driver activities associated with CUDA kernel launches and CUDA API calls.

    Graphs must be built using a :obj:`~_graph.GraphBuilder` object.

    """

    class _MembersNeededForFinalize:
        __slots__ = "graph"

        def __init__(self, graph_obj, graph):
            self.graph = graph
            weakref.finalize(graph_obj, self.close)

        def close(self):
            if self.graph:
                handle_return(driver.cuGraphExecDestroy(self.graph))
                self.graph = None

    __slots__ = ("__weakref__", "_mnff")

    def __init__(self):
        raise RuntimeError("directly constructing a Graph instance is not supported")

    @classmethod
    def _init(cls, graph):
        self = cls.__new__(cls)
        self._mnff = Graph._MembersNeededForFinalize(self, graph)
        return self

    def close(self):
        """Destroy the graph."""
        self._mnff.close()

    def update(self, builder: GraphBuilder):
        """Update the graph using new build configuration from the builder.

        The topology of the provided builder must be identical to this graph.

        Parameters
        ----------
        builder : :obj:`~_graph.GraphBuilder`
            The builder to update the graph with.

        """
        if not builder._building_ended:
            raise ValueError("Graph has not finished building.")

        # Update the graph with the new nodes from the builder
        exec_update_result = handle_return(driver.cuGraphExecUpdate(self._mnff.graph, builder._mnff.graph))
        if exec_update_result.result != driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_SUCCESS:
            raise RuntimeError(f"Failed to update graph: {exec_update_result.result()}")

    def upload(self, stream: Stream):
        """Uploads the graph in a stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream`
            The stream in which to upload the graph

        """
        handle_return(driver.cuGraphUpload(self._mnff.graph, stream.handle))

    def launch(self, stream: Stream):
        """Launches the graph in a stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream`
            The stream in which to launch the graph

        """
        handle_return(driver.cuGraphLaunch(self._mnff.graph, stream.handle))
