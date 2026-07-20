# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for updating individual graph node parameters."""

import threading
from dataclasses import dataclass
from typing import Callable

import pytest

from cuda.core import Device
from cuda.core._utils.cuda_utils import CUDAError
from cuda.core._utils.version import driver_version
from cuda.core.graph import GraphDefinition


@dataclass
class _DefinitionUpdateCase:
    graph_def: GraphDefinition
    node: object
    original: object
    replacement: object
    invalid_replacement: object
    update: Callable[[object], None]
    current: Callable[[], object]
    assert_exec_uses: Callable[[object, object], None]


def _event_record_case(device):
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
        invalid_replacement=invalid_replacement,
        update=node.update,
        current=lambda: node.event,
        assert_exec_uses=assert_exec_uses,
    )


@pytest.fixture(
    params=[
        pytest.param(_event_record_case, id="event-record"),
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
    assert case.current() == case.replacement

    new_graph = case.graph_def.instantiate()
    case.assert_exec_uses(old_graph, case.original)
    case.assert_exec_uses(new_graph, case.replacement)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_failed_definition_node_update_preserves_state(
    definition_update_case,
):
    case = definition_update_case

    with pytest.raises(CUDAError):
        case.update(case.invalid_replacement)

    assert case.current() == case.original
    graph = case.graph_def.instantiate()
    case.assert_exec_uses(graph, case.original)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_definition_node_update_rejects_wrong_type(
    definition_update_case,
):
    with pytest.raises(TypeError):
        definition_update_case.update(object())
