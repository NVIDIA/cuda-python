# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for updating individual graph node parameters."""

import ctypes
import threading
from dataclasses import dataclass
from typing import Callable

import pytest

from cuda.core._utils.cuda_utils import CUDAError
from cuda.core._utils.version import driver_version
from cuda.core.graph import GraphDefinition


@dataclass
class _DefinitionUpdateCase:
    graph_def: GraphDefinition
    node: object
    original: object
    replacement: object
    update: Callable[[object], None]
    assert_current: Callable[[object], None]
    assert_exec_uses: Callable[[object, object], None]
    invalid_update: Callable[[], None] | None
    invalid_exception: type[BaseException] | None
    invalid_argument_update: Callable[[], None] | None


def _assert_equal(actual, expected):
    assert actual == expected


def _event_record_case(device):
    """Keep the selected event pending to identify each exec's record target."""
    original = device.create_event()
    replacement = device.create_event()
    invalid_replacement = device.create_event()
    invalid_replacement.close()

    callback_started = threading.Event()
    callback_release = threading.Event()

    def blocking_callback():
        callback_started.set()
        callback_release.wait(timeout=30)

    graph_def = GraphDefinition()
    callback_node = graph_def.callback(blocking_callback)
    node = callback_node.record(original)

    def assert_exec_uses(graph, expected):
        callback_started.clear()
        callback_release.clear()
        stream = device.create_stream()
        graph.launch(stream)
        try:
            assert callback_started.wait(timeout=5)
            assert expected.is_done is False
            unexpected = replacement if expected is original else original
            assert unexpected.is_done is True
        finally:
            callback_release.set()
            stream.sync()

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=node.update,
        assert_current=lambda expected: _assert_equal(node.event, expected),
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(invalid_replacement),
        invalid_exception=CUDAError,
        invalid_argument_update=lambda: node.update(object()),
    )


def _event_wait_case(device):
    """Keep the selected event pending to identify each exec's wait target."""
    original = device.create_event()
    replacement = device.create_event()
    invalid_replacement = device.create_event()
    invalid_replacement.close()

    callback_called = threading.Event()
    graph_def = GraphDefinition()
    node = graph_def.wait(original)
    node.callback(callback_called.set)

    def assert_exec_uses(graph, expected):
        producer_started = threading.Event()
        producer_release = threading.Event()

        def blocking_callback():
            producer_started.set()
            producer_release.wait(timeout=30)

        producer_def = GraphDefinition()
        producer_def.callback(blocking_callback).record(expected)
        producer_graph = producer_def.instantiate()
        producer_stream = device.create_stream()
        consumer_stream = device.create_stream()

        callback_called.clear()
        producer_graph.launch(producer_stream)
        try:
            assert producer_started.wait(timeout=5)
            graph.launch(consumer_stream)
            assert not callback_called.wait(timeout=0.1)
        finally:
            producer_release.set()
            producer_stream.sync()
            consumer_stream.sync()
        assert callback_called.is_set()

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=node.update,
        assert_current=lambda expected: _assert_equal(node.event, expected),
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(invalid_replacement),
        invalid_exception=CUDAError,
        invalid_argument_update=lambda: node.update(object()),
    )


def _host_callback_case(device):
    """Use callbacks that report their identity to distinguish each exec."""
    called = []

    def original():
        called.append(original)

    def replacement():
        called.append(replacement)

    graph_def = GraphDefinition()
    node = graph_def.callback(original)

    def assert_exec_uses(graph, expected):
        called.clear()
        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()
        assert called == [expected]

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=node.update,
        assert_current=lambda expected: _assert_equal(node.callback, expected),
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(replacement, user_data=b"not valid for a Python callback"),
        invalid_exception=ValueError,
        invalid_argument_update=None,
    )


def _host_callback_ctypes_case(device):
    """Use ctypes callbacks and copied payloads to distinguish each exec."""
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    called = []

    def read_byte(data):
        return ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))[0]

    @callback_type
    def original_fn(data):
        called.append((original_fn, read_byte(data)))

    @callback_type
    def replacement_fn(data):
        called.append((replacement_fn, read_byte(data)))

    original = original_fn, bytes([0xA1])
    replacement = replacement_fn, bytes([0xB2])
    graph_def = GraphDefinition()
    node = graph_def.callback(original_fn, user_data=original[1])

    def update(value):
        fn, user_data = value
        node.update(fn, user_data=user_data)

    def assert_exec_uses(graph, expected):
        called.clear()
        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()
        assert called == [(expected[0], expected[1][0])]

    def invalid_update():
        node.update(lambda: None, user_data=b"not valid for a Python callback")

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=update,
        assert_current=lambda _expected: _assert_equal(node.callback, None),
        assert_exec_uses=assert_exec_uses,
        invalid_update=invalid_update,
        invalid_exception=ValueError,
        invalid_argument_update=None,
    )


@pytest.fixture(
    params=[
        pytest.param(_event_record_case, id="event-record"),
        pytest.param(_event_wait_case, id="event-wait"),
        pytest.param(_host_callback_case, id="host-callback-python"),
        pytest.param(_host_callback_ctypes_case, id="host-callback-ctypes"),
    ]
)
def definition_update_case(request, init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")
    return request.param(init_cuda)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_definition_node_update_changes_future_instantiations(
    definition_update_case,
):
    case = definition_update_case
    old_graph = case.graph_def.instantiate()

    case.update(case.replacement)
    case.assert_current(case.replacement)

    new_graph = case.graph_def.instantiate()
    case.assert_exec_uses(old_graph, case.original)
    case.assert_exec_uses(new_graph, case.replacement)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_failed_definition_node_update_preserves_state(
    definition_update_case,
):
    case = definition_update_case

    assert case.invalid_update is not None
    assert case.invalid_exception is not None
    with pytest.raises(case.invalid_exception):
        case.invalid_update()

    case.assert_current(case.original)
    graph = case.graph_def.instantiate()
    case.assert_exec_uses(graph, case.original)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_definition_node_update_rejects_wrong_type(
    definition_update_case,
):
    if definition_update_case.invalid_argument_update is None:
        pytest.skip("update method has no typed positional argument")
    with pytest.raises(TypeError):
        definition_update_case.invalid_argument_update()
