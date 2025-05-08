# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils.cuda_utils import (
    driver,
    handle_return,
)


@dataclass
class DebugPrintOptions:
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
class CompleteOptions:
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
        __slots__ = ("stream", "is_stream_owner", "graph", "is_conditional", "is_join_required")

        def __init__(self, graph_builder_obj, stream_obj, is_stream_owner, graph, is_conditional, is_join_required):
            self.stream = stream_obj
            self.is_stream_owner = is_stream_owner
            self.graph = graph
            self.is_conditional = is_conditional
            self.is_join_required = is_join_required
            weakref.finalize(graph_builder_obj, self.close)

        def close(self):
            if self.graph and not self.is_conditional:
                handle_return(driver.cuGraphDestroy(self.graph))
            self.graph = None
            if self.is_stream_owner and self.stream:
                self.stream.close()
            self.stream = None

    __slots__ = ("__weakref__", "_mnff", "_building_ended")

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Graph object can be ambiguous. Please either "
            "call Device.create_graph_builder() or stream.creating_graph_builder()"
        )

    @classmethod
    def _init(cls, stream, is_stream_owner, graph=None, is_conditional=False, is_join_required=False):
        self = cls.__new__(cls)
        self._mnff = GraphBuilder._MembersNeededForFinalize(
            self, stream, is_stream_owner, graph, is_conditional, is_join_required
        )

        if not self._mnff.graph:
            self._mnff.graph = handle_return(driver.cuGraphCreate(0))
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

    def begin_building(self) -> GraphBuilder:
        """Begins the building process."""
        if self._building_ended:
            raise RuntimeError("Cannot resume building after building has ended.")
        handle_return(
            driver.cuStreamBeginCaptureToGraph(
                self._mnff.stream.handle,
                self._mnff.graph,
                None,  # dependencies
                None,  # dependencyData
                0,  # numDependencies
                driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        )
        return self

    @property
    def is_building(self) -> bool:
        """Returns True if the graph builder is currently building."""
        capture_status, _, _, _, _ = handle_return(driver.cuStreamGetCaptureInfo(self._mnff.stream.handle))
        if capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            return False
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            return True
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
            self.end_building()
            raise RuntimeError(
                "Build process encountered an error and has been invalidated. Build process has been ended."
            )
        else:
            raise NotImplementedError(f"Unsupported capture status type received: {capture_status}")

    def end_building(self) -> GraphBuilder:
        """Ends the building process."""
        if not self.is_building:
            raise RuntimeError("Graph builder is not building.")
        self._mnff.graph = handle_return(driver.cuStreamEndCapture(self.stream.handle))

        # TODO: Resolving https://github.com/NVIDIA/cuda-python/issues/617 would allow us to
        #       resume the build process after the first call to end_building()
        self._building_ended = True
        return self

    def complete(self, options: Optional[CompleteOptions] = None) -> Graph:
        """Completes the graph builder and returns the built :obj:`~_graph.Graph` object.

        Parameters
        ----------
        options : :obj:`~_graph.CompleteOptions`, optional
            Customizable dataclass for the graph builder completion options.

        Returns
        -------
        graph : :obj:`~_graph.Graph`
            The newly built graph.

        """
        if not self._building_ended:
            raise RuntimeError("Graph has not finished building.")

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

    def debug_dot_print(self, path, options: Optional[DebugPrintOptions] = None):
        """Generates a DOT debug file for the graph builder.

        Parameters
        ----------
        path : str
            File path to use for writting debug DOT output
        options : :obj:`~_graph.DebugPrintOptions`, optional
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

    def split(self, count) -> Tuple[GraphBuilder, ...]:
        """Splits the original graph builder into multiple graph builders.

        The new builders inherit work dependencies from the original builder.
        The original builder is reused for the split and is returned first in the tuple.

        Parameters
        ----------
        count : int
            The number of graph builders to split the graph builder into.

        Returns
        -------
        graph_builders : Tuple[:obj:`~_graph.GraphBuilder`, ...]
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
                GraphBuilder._init(
                    stream=stream, is_stream_owner=True, graph=None, is_conditional=False, is_join_required=True
                )
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
        if not all(isinstance(builder, GraphBuilder) for builder in graph_builders):
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
        for idx, builder in enumerate(graph_builders):
            if idx == root_idx:
                continue
            graph_builders[root_idx].stream.wait(builder.stream)
            builder.close()

        return graph_builders[root_idx]

    def __cuda_stream__(self) -> Tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return self.stream.__cuda_stream__()

    def _get_conditional_context(self):
        driver_ver = handle_return(driver.cuDriverGetVersion())
        if driver_ver < 12050:
            # Pre 12.5 drivers don't allow querying the stream context during capture.
            return handle_return(driver.cuCtxGetCurrent())
        else:
            return handle_return(driver.cuStreamGetCtx(self._mnff.stream.handle))

    def create_conditional_handle(self, default_value=None) -> int:
        """Creates a conditional handle for the graph builder.

        Parameters
        ----------
        default_value : int, optional
            The default value to assign to the conditional handle.

        Returns
        -------
        handle : int
            The newly created conditional handle.

        """
        if default_value is not None:
            flags = driver.CU_GRAPH_COND_ASSIGN_DEFAULT
        else:
            default_value = 0
            flags = 0
        return int(
            handle_return(
                driver.cuGraphConditionalHandleCreate(
                    self._mnff.graph, self._get_conditional_context(), default_value, flags
                )
            )
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
                    graph=node_params.conditional.phGraph_out[i],
                    is_conditional=True,
                    is_join_required=False,
                )
                for i in range(node_params.conditional.size)
            ]
        )

    def if_cond(self, handle: int) -> GraphBuilder:
        """Adds an if condition branch and returns a new graph builder for it.

        The resulting if graph will only execute the branch if the conditional
        handle evaluates to true at runtime.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        handle : int
            The handle to use for the if conditional.

        Returns
        -------
        graph_builder : :obj:`~_graph.GraphBuilder`
            The newly created conditional graph builder.

        """
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 1
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)[0]

    def if_else(self, handle: int) -> Tuple[GraphBuilder, GraphBuilder]:
        """Adds an if-else condition branch and returns new graph builders for both branches.

        The resulting if graph will execute the branch if the conditional handle
        evaluates to true at runtime, otherwise the else branch will execute.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        handle : int
            The handle to use for the if-else conditional.

        Returns
        -------
        graph_builders : Tuple[:obj:`~_graph.GraphBuilder`, :obj:`~_graph.GraphBuilder`]
            A tuple of two new graph builders, one for the if branch and one for the else branch.

        """
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        node_params.conditional.size = 2
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)

    def switch(self, handle: int, count: int) -> Tuple[GraphBuilder, ...]:
        """Adds a switch condition branch and returns new graph builders for all cases.

        The resulting switch graph will execute the branch that matches the
        case index of the conditional handle at runtime. If no match is found, no branch
        will be executed.

        The new builders inherit work dependencies from the original builder.

        Parameters
        ----------
        handle : int
            The handle to use for the switch conditional.
        count : int
            The number of cases to add to the switch conditional.

        Returns
        -------
        graph_builders : Tuple[:obj:`~_graph.GraphBuilder`, ...]
            A tuple of new graph builders, one for each branch.

        """
        node_params = driver.CUgraphNodeParams()
        node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        node_params.conditional.handle = handle
        node_params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_SWITCH
        node_params.conditional.size = count
        node_params.conditional.ctx = self._get_conditional_context()
        return self._cond_with_params(node_params)

    def while_loop(self, handle: int) -> GraphBuilder:
        """Adds a while loop and returns a new graph builder for it.

        The resulting while loop graph will execute the branch repeatedly at runtime
        until the conditional handle evaluates to false.

        The new builder inherits work dependencies from the original builder.

        Parameters
        ----------
        handle : int
            The handle to use for the while loop.

        Returns
        -------
        graph_builder : :obj:`~_graph.GraphBuilder`
            The newly created while loop graph builder.

        """
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


def launch_graph(parent_graph: GraphBuilder, child_graph: GraphBuilder):
    """Adds the child :obj:`~_graph.GraphBuilder` builder into parent graph builder.

    The child graph builder will be added as a child node to the parent graph builder.

    Parameters
    ----------
    parent_graph : :obj:`~_graph.GraphBuilder`
        The parent graph builder. Must be in building state.
    child_graph : :obj:`~_graph.GraphBuilder`
        The child graph builder. Must have finished building.

    """

    if not child_graph._building_ended:
        raise ValueError("Child graph has not finished building.")

    if not parent_graph.is_building:
        raise ValueError("Parent graph is being built.")

    status, _, graph_out, dependencies_out, num_dependencies_out = handle_return(
        driver.cuStreamGetCaptureInfo(parent_graph.stream.handle)
    )
    if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise ValueError("Parent graph is not in an active capture state")

    child_node = handle_return(
        driver.cuGraphAddChildGraphNode(graph_out, dependencies_out, num_dependencies_out, child_graph._mnff.graph)
    )
    handle_return(
        driver.cuStreamUpdateCaptureDependencies(
            parent_graph.stream.handle,
            [child_node],
            1,
            driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
        )
    )
