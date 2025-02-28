# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

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
    def _init(stream, _is_primary=True):
        self = GraphBuilder.__new__(GraphBuilder)
        # TODO: I need to know if we own this stream object.
        #       If from Device(), then we can destroy it on close
        #       If from Stream, then we can't
        self._capturing = False
        self._is_primary = _is_primary
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
    def begin_capture(self, mode="global"):
        # Supports "global", "local" or "relaxed"
        # TODO; Test case for each mode and fail
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
                "not terminated. The capture sequence must be terminated with self.`()."
            )
        else:
            raise NotImplementedError(f"Unsupported capture stuse type received: {capture_status}")

    @precondition(_check_capture_stream_provided)
    def end_capture(self):
        if not self._capturing:
            raise RuntimeError("Stream is not capturing. Did you forget to call begin_capture()?")
        self._mnff.graph = handle_return(driver.cuStreamEndCapture(self.stream.handle))
        self._capturing = False

    def debug_dot_print(self, path, options: Optional[DebugPrintOptions] = None):
        # TODO: We should be able to print one while the capture is happening right? Just need to make sure driver version is new enough.
        if self._mnff.graph == None:
            raise RuntimeError("Graph needs to be built before generating a DOT debug file")

        # TODO: Apply each option to the value
        options_value = 0

        handle_return(driver.cuGraphDebugDotPrint(self._mnff.graph, path, options_value))

    def fork(self, count) -> Tuple[Graph, ...]:
        if count <= 1:
            raise ValueError(f"Invalid fork count: expecting >= 2, got {count}")

        # 1. Record an event on our stream
        event = self._mnff.stream.record()

        # TODO: Steps 2,3,4 can be combined under a single loop

        # 2. Create a streams for each of the new forks
        # TODO: Optimization where one of the fork stream is allowed to use
        # TODO: Should use the same stream options as initial stream??
        fork_stream = [self._mnff.stream.device.create_stream() for i in range(count)]

        # 3. Have each new stream wait on our singular event
        for stream in fork_stream:
            stream.wait(event)

        # 4. Discard the event
        # TODO: Is this actually allowed when using with a graph? Surely, since it just needs to create an edge for us... right?
        event.close()

        # 5. Create new graph builders for each new stream fork
        return [GraphBuilder._init(stream=stream, is_primary=False) for stream in fork_stream]

    def join(self, *graph_builders):
        if len(graph_builders) < 1:
            raise ValueError("Must specify which graphs should join but none were given")

        # Assert that none of the graph_builders are primary
        for graph in graph_builders:
            if graph.is_primary:
                raise ValueError("The primary graph builder should not be joined. Others builders should instead be joined onto it.")

        for graph in graph_builders:
            self._mnff.stream.wait(graph.stream)
            # TODO: Do we close them now or let weakref handle it during garbage collection?
            #       This is a perf question, is there a good default?
            graph.close()

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
