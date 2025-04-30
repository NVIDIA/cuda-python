# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import weakref
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils import driver, handle_return, precondition


@dataclass
class DebugPrintOptions:
    """ """

    VERBOSE: bool = False
    RUNTIME_TYPES: bool = False
    KERNEL_NODE_PARAMS: bool = False
    MEMCPY_NODE_PARAMS: bool = False
    MEMSET_NODE_PARAMS: bool = False
    HOST_NODE_PARAMS: bool = False
    EVENT_NODE_PARAMS: bool = False
    EXT_SEMAS_SIGNAL_NODE_PARAMS: bool = False
    EXT_SEMAS_WAIT_NODE_PARAMS: bool = False
    KERNEL_NODE_ATTRIBUTES: bool = False
    HANDLES: bool = False
    MEM_ALLOC_NODE_PARAMS: bool = False
    MEM_FREE_NODE_PARAMS: bool = False
    BATCH_MEM_OP_NODE_PARAMS: bool = False
    EXTRA_TOPO_INFO: bool = False
    CONDITIONAL_NODE_PARAMS: bool = False


@dataclass
class CompleteOptions:
    """Customizable options for :obj:`_graph.GraphBuilder.complete()`

    Attributes
    ----------
    auto_free_on_launch : bool, optional
        Automatically free memory allocated in a graph before relaunching. (Default to False)
    upload : bool, optional
        Automatically upload the graph after instantiation. (Default to False)
    device_launch : bool, optional
        Configure the graph to be launchable from the device. This flag can only
        be used on platforms which support unified addressing. This flag cannot be
        used in conjunction with auto_free_on_launch. (Default to False)
    use_node_priority : bool, optional
        Run the graph using the per-node priority attributes rather than the
        priority of the stream it is launched into. (Default to False)

    """

    auto_free_on_launch: bool = False
    upload: bool = False
    device_launch: bool = False
    use_node_priority: bool = False


class GraphBuilder:
    """TBD

    Directly creating a :obj:`~_graph.GraphBuilder` is not supported due
    to ambiguity. New graph builders should instead be created through a
    :obj:`~_device.Device`, or a :obj:`~_stream.stream` object
    """

    class _MembersNeededForFinalize:
        __slots__ = ("stream", "graph")

        def __init__(self, graph_builder_obj, stream_obj):
            self.stream = stream_obj
            self.graph = None
            weakref.finalize(graph_builder_obj, self.close)

        def close(self):
            # FIXME: Are the stream and graph builder racing for the weakref callback?
            #        If so, maybe we need to enforce that all capture is completed
            status = handle_return(driver.cuStreamGetCaptureInfo(self.stream.handle))[0]
            if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
                # Callback routine needs to end capture for error free handling
                handle_return(driver.cuStreamEndCapture(self.stream.handle))
            if self.graph:
                handle_return(driver.cuGraphDestroy(self.graph))
            self.graph = None

    __slots__ = ("__weakref__", "_mnff", "_is_primary", "_capturing")

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Graph object can be ambiguous. Please either "
            "call Device.create_graph() or stream.creating_graph()"
        )

    @staticmethod
    def _init(stream, can_destroy_stream, _is_primary=True):
        self = GraphBuilder.__new__(GraphBuilder)
        # TODO: I need to know if we own this stream object.
        #       If from Device(), then we can destroy it on close
        #       If from Stream, then we can't
        self._capturing = False
        self._is_primary = _is_primary
        self._can_destroy_stream = can_destroy_stream
        self._mnff = GraphBuilder._MembersNeededForFinalize(self, stream)
        return self

    def _check_capture_stream_provided(self, *args, **kwargs):
        if self._mnff.stream == None:
            raise RuntimeError("Tried to use a stream capture operation on a graph builder without a stream")

    @property
    def stream(self) -> Stream:
        return self._mnff.stream

    @property
    def is_primary(self) -> bool:
        return self._is_primary

    @precondition(_check_capture_stream_provided)
    def begin_building(self, mode="global") -> GraphBuilder:
        # Supports "global", "local" or "relaxed"
        if mode == "global":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL
        elif mode == "local":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        elif mode == "relaxed":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        else:
            raise ValueError(f"Only 'global', 'local' or 'relaxed' capture mode are supported, got {capture_mode}")

        handle_return(driver.cuStreamBeginCapture(self._mnff.stream.handle, capture_mode))
        self._capturing = True
        return self

    @precondition(_check_capture_stream_provided)
    def is_capture_active(self) -> bool:
        result = handle_return(driver.cuStreamGetCaptureInfo(self._mnff.stream.handle))

        capture_status = result[0]
        if capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            return False
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            return True
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
            raise RuntimeError(
                "Stream is part of a capture sequence that has been invalidated, but "
                "not terminated. The capture sequence must be terminated with self.end_capture()."
            )
        else:
            raise NotImplementedError(f"Unsupported capture stuse type received: {capture_status}")

    @precondition(_check_capture_stream_provided)
    def end_building(self) -> GraphBuilder:
        if not self._capturing:
            raise RuntimeError("Stream is not capturing. Was self.begin_capture() called?")
        self._mnff.graph = handle_return(driver.cuStreamEndCapture(self.stream.handle))
        self._capturing = False
        return self

    def complete(self, options: Optional[CompleteOptions] = None) -> Graph:
        flags = 0
        if options:
            if options.auto_free_on_launch:
                flags |= driver.CU_GRAPH_COMPLETE_FLAG_AUTO_FREE_ON_LAUNCH
            if options.upload:
                flags |= driver.CU_GRAPH_COMPLETE_FLAG_UPLOAD
            if options.device_launch:
                flags |= driver.CU_GRAPH_COMPLETE_FLAG_DEVICE_LAUNCH
            if options.use_node_priority:
                flags |= driver.CU_GRAPH_COMPLETE_FLAG_USE_NODE_PRIORITY

        return Graph._init(handle_return(driver.cuGraphInstantiate(self._mnff.graph, flags)))

    def debug_dot_print(self, path, options: Optional[DebugPrintOptions] = None):
        if self._mnff.graph == None:
            raise RuntimeError("Graph needs to be built before generating a DOT debug file")

        flags = 0
        if options:
            if options.VERBOSE:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_VERBOSE
            if options.RUNTIME_TYPES:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_RUNTIME_TYPES
            if options.KERNEL_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_KERNEL_NODE_PARAMS
            if options.MEMCPY_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_MEMCPY_NODE_PARAMS
            if options.MEMSET_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_MEMSET_NODE_PARAMS
            if options.HOST_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_HOST_NODE_PARAMS
            if options.EVENT_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_EVENT_NODE_PARAMS
            if options.EXT_SEMAS_SIGNAL_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_EXT_SEMAS_SIGNAL_NODE_PARAMS
            if options.EXT_SEMAS_WAIT_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_EXT_SEMAS_WAIT_NODE_PARAMS
            if options.KERNEL_NODE_ATTRIBUTES:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_KERNEL_NODE_ATTRIBUTES
            if options.HANDLES:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_HANDLES
            if options.MEM_ALLOC_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_MEM_ALLOC_NODE_PARAMS
            if options.MEM_FREE_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_MEM_FREE_NODE_PARAMS
            if options.BATCH_MEM_OP_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_BATCH_MEM_OP_NODE_PARAMS
            if options.EXTRA_TOPO_INFO:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_EXTRA_TOPO_INFO
            if options.CONDITIONAL_NODE_PARAMS:
                flags |= driver.CU_GRAPH_DEBUG_DOT_PRINT_CONDITIONAL_NODE_PARAMS

        handle_return(driver.cuGraphDebugDotPrint(self._mnff.graph, path, flags))

    def split(self, count) -> Tuple[GraphBuilder, ...]:
        if count <= 1:
            raise ValueError(f"Invalid split count: expecting >= 2, got {count}")

        event = self._mnff.stream.record()
        result = [self]
        for i in range(count-1):
            stream = self._mnff.stream.device.create_stream()
            stream.wait(event)
            result.append(GraphBuilder._init(stream=stream, is_primary=False))
        event.close()
        return result

    @staticmethod
    def join(*graph_builders):
        if not all(isinstance(builder, GraphBuilder) for builder in graph_builders):
            raise TypeError("All arguments must be GraphBuilder instances")

        if len(graph_builders) < 1:
            raise ValueError("Must join with at least two graph builders")

        # Discover which builder should join
        join_idx = 0
        for i, builder in enumerate(graph_builders):
            if builder.is_primary:
                join_idx = i
                break

        # Join builder waits on all builders
        for i, builder in enumerate(graph_builders):
            if i == join_idx:
                continue
            builder.stream.wait(builder.stream)
            builder.close()

        return graph_builders[join_idx]

    def __cuda_stream__(self) -> Tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return self.stream.__cuda_stream__

    def create_conditional_handle(self, default_value=None):
        pass

    def if_cond(self, handle):
        pass

    def if_else(self, handle):
        pass

    def switch(self, handle, count):
        pass

    def close(self):
        if self._mnff.capturing:
            # Explicitly trying to close a graph builder who is still capturing is not allowed
            raise RuntimeError("Trying to close a graph builder who is still capturing. Did you forget to call end_capture()?")
        self._mnff.close()


class Graph:
    """ """

    def __init__(self):
        raise RuntimeError("directly constructing a Graph instance is not supported")

    @staticmethod
    def _init(graph):
        self = Graph.__new__(Graph)
        self._graph = graph
        return self
