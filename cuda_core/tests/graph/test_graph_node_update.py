# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for updating individual graph node parameters."""

import ctypes
import threading
from dataclasses import dataclass
from typing import Callable

import pytest
from helpers.graph_kernels import compile_common_kernels

from cuda.core import LaunchConfig, LegacyPinnedMemoryResource
from cuda.core._utils.cuda_utils import CUDAError
from cuda.core._utils.version import driver_version
from cuda.core.graph import GraphDefinition, HostCallbackNode


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
    cleanup: Callable[[], None]


def _assert_equal(actual, expected):
    assert actual == expected


def _noop():
    pass


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
        cleanup=_noop,
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
        cleanup=_noop,
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
        cleanup=_noop,
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
        cleanup=_noop,
    )


def _memset_case(device, *, replace_dst):
    memory_resource = LegacyPinnedMemoryResource()
    original_buffer = memory_resource.allocate(4)
    replacement_buffer = memory_resource.allocate(4) if replace_dst else original_buffer
    original = {
        "dst": original_buffer,
        "value": 0x11,
        "element_size": 1,
        "width": 4,
        "height": 1,
        "pitch": 0,
    }
    replacement = {
        **original,
        "dst": replacement_buffer,
        "value": 0x22,
    }

    graph_def = GraphDefinition()
    node = graph_def.memset(original["dst"], original["value"], original["width"])

    def update(expected):
        if replace_dst:
            node.update(dst=expected["dst"], value=expected["value"])
        else:
            node.update(value=expected["value"])

    def assert_current(expected):
        assert node.dptr == int(expected["dst"].handle)
        assert node.value == expected["value"]
        assert node.element_size == expected["element_size"]
        assert node.width == expected["width"]
        assert node.height == expected["height"]
        assert node.pitch == expected["pitch"]

    def as_bytes(buffer):
        return (ctypes.c_uint8 * 4).from_address(int(buffer.handle))

    def assert_exec_uses(graph, expected):
        original_data = as_bytes(original_buffer)
        replacement_data = as_bytes(replacement_buffer)
        original_data[:] = [0] * 4
        replacement_data[:] = [0] * 4

        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()

        assert list(as_bytes(expected["dst"])) == [expected["value"]] * 4
        if replace_dst:
            unexpected = replacement_buffer if expected["dst"] is original_buffer else original_buffer
            assert list(as_bytes(unexpected)) == [0] * 4

    def cleanup():
        node.destroy()
        original_buffer.close()
        if replacement_buffer is not original_buffer:
            replacement_buffer.close()

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=update,
        assert_current=assert_current,
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(value=256),
        invalid_exception=OverflowError,
        invalid_argument_update=lambda: node.update(dst=object()),
        cleanup=cleanup,
    )


def _memset_value_case(device):
    """Change the fill value while preserving destination ownership."""
    return _memset_case(device, replace_dst=False)


def _memset_destination_case(device):
    """Replace the destination and its retained allocation owner."""
    return _memset_case(device, replace_dst=True)


def _memcpy_case(device, *, replace_operand):
    memory_resource = LegacyPinnedMemoryResource()
    original_src = memory_resource.allocate(4)
    original_dst = memory_resource.allocate(4)
    replacement_src = memory_resource.allocate(4) if replace_operand == "src" else original_src
    replacement_dst = memory_resource.allocate(4) if replace_operand == "dst" else original_dst
    original = {
        "dst": original_dst,
        "src": original_src,
        "size": 2 if replace_operand is None else 4,
    }
    replacement = {
        "dst": replacement_dst,
        "src": replacement_src,
        "size": 4,
    }

    graph_def = GraphDefinition()
    node = graph_def.memcpy(original["dst"], original["src"], original["size"])

    def update(expected):
        if replace_operand == "src":
            node.update(src=expected["src"])
        elif replace_operand == "dst":
            node.update(dst=expected["dst"])
        else:
            node.update(size=expected["size"])

    def assert_current(expected):
        assert node.dst == int(expected["dst"].handle)
        assert node.src == int(expected["src"].handle)
        assert node.size == expected["size"]

    def as_bytes(buffer):
        return (ctypes.c_uint8 * 4).from_address(int(buffer.handle))

    def assert_exec_uses(graph, expected):
        as_bytes(original_src)[:] = [0x11] * 4
        as_bytes(original_dst)[:] = [0] * 4
        if replacement_src is not original_src:
            as_bytes(replacement_src)[:] = [0x22] * 4
        if replacement_dst is not original_dst:
            as_bytes(replacement_dst)[:] = [0] * 4

        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()

        source_value = 0x11 if expected["src"] is original_src else 0x22
        expected_data = [source_value] * expected["size"]
        expected_data.extend([0] * (4 - expected["size"]))
        assert list(as_bytes(expected["dst"])) == expected_data
        if replacement_dst is not original_dst:
            unexpected_dst = replacement_dst if expected["dst"] is original_dst else original_dst
            assert list(as_bytes(unexpected_dst)) == [0] * 4

    def cleanup():
        node.destroy()
        original_src.close()
        original_dst.close()
        if replacement_src is not original_src:
            replacement_src.close()
        if replacement_dst is not original_dst:
            replacement_dst.close()

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=update,
        assert_current=assert_current,
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(size=-1),
        invalid_exception=OverflowError,
        invalid_argument_update=lambda: node.update(src=object()),
        cleanup=cleanup,
    )


def _memcpy_size_case(device):
    """Change the copy size while preserving both operand owners."""
    return _memcpy_case(device, replace_operand=None)


def _memcpy_source_case(device):
    """Replace the source while preserving destination ownership."""
    return _memcpy_case(device, replace_operand="src")


def _memcpy_destination_case(device):
    """Replace the destination while preserving source ownership."""
    return _memcpy_case(device, replace_operand="dst")


def _kernel_case(device, *, replace):
    module = compile_common_kernels()
    add_one = module.get_kernel("add_one")
    empty_kernel = module.get_kernel("empty_kernel")
    write_launch_dims = module.get_kernel("write_launch_dims")
    memory_resource = LegacyPinnedMemoryResource()
    original_buffer = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    replacement_buffer = memory_resource.allocate(ctypes.sizeof(ctypes.c_int)) if replace == "args" else original_buffer

    original_config = LaunchConfig(grid=1, block=1)
    replacement_config = LaunchConfig(grid=2, block=3) if replace == "config" else original_config
    original_kernel = write_launch_dims if replace == "config" else add_one
    replacement_kernel = empty_kernel if replace == "kernel" else original_kernel
    original_args = (original_buffer,)
    if replace == "kernel":
        replacement_args = ()
    elif replace == "args":
        replacement_args = (replacement_buffer,)
    else:
        replacement_args = original_args

    original = {
        "config": original_config,
        "kernel": original_kernel,
        "args": original_args,
        "output": original_buffer,
        "expected": 1001 if replace == "config" else 1,
    }
    replacement = {
        "config": replacement_config,
        "kernel": replacement_kernel,
        "args": replacement_args,
        "output": replacement_buffer,
        "expected": 2003 if replace == "config" else int(replace != "kernel"),
    }

    graph_def = GraphDefinition()
    node = graph_def.launch(original["config"], original["kernel"], *original["args"])

    def update(expected):
        if replace == "config":
            node.update(config=expected["config"])
        elif replace == "args":
            node.update(args=expected["args"])
        else:
            node.update(kernel=expected["kernel"], args=expected["args"])

    def assert_current(expected):
        assert node.config == expected["config"]
        assert int(node.kernel.handle) == int(expected["kernel"].handle)

    def as_int(buffer):
        return ctypes.c_int.from_address(int(buffer.handle))

    def assert_exec_uses(graph, expected):
        as_int(original_buffer).value = 0
        as_int(replacement_buffer).value = 0

        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()

        assert as_int(expected["output"]).value == expected["expected"]
        if replacement_buffer is not original_buffer:
            unexpected = replacement_buffer if expected["output"] is original_buffer else original_buffer
            assert as_int(unexpected).value == 0

    def invalid_update():
        if replace == "kernel":
            node.update(kernel=replacement_kernel)
        elif replace == "args":
            node.update(args=(object(),))
        else:
            node.update(config=object())

    invalid_exception = ValueError if replace == "kernel" else TypeError

    def cleanup():
        node.destroy()
        original_buffer.close()
        if replacement_buffer is not original_buffer:
            replacement_buffer.close()

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=update,
        assert_current=assert_current,
        assert_exec_uses=assert_exec_uses,
        invalid_update=invalid_update,
        invalid_exception=invalid_exception,
        invalid_argument_update=lambda: node.update(config=object()),
        cleanup=cleanup,
    )


def _kernel_config_case(device):
    """Replace launch dimensions while preserving the kernel and arguments."""
    return _kernel_case(device, replace="config")


def _kernel_args_case(device):
    """Replace arguments while preserving the kernel and configuration."""
    return _kernel_case(device, replace="args")


def _kernel_function_case(device):
    """Replace a kernel and explicitly supply its coupled arguments."""
    return _kernel_case(device, replace="kernel")


def _child_graph_case(device):
    """Replace the embedded clone while preserving existing executables."""
    called = []

    def original_callback():
        called.append(original_callback)

    def replacement_callback():
        called.append(replacement_callback)

    original_child = GraphDefinition()
    original_child.callback(original_callback)
    replacement_child = GraphDefinition()
    replacement_child.callback(replacement_callback)
    original = {
        "child": original_child,
        "callback": original_callback,
    }
    replacement = {
        "child": replacement_child,
        "callback": replacement_callback,
    }

    graph_def = GraphDefinition()
    node = graph_def.embed(original_child)
    invalid_child = node.child_graph

    def update(expected):
        node.update(expected["child"])

    def assert_current(expected):
        callback_node = next(
            child_node for child_node in node.child_graph.nodes() if isinstance(child_node, HostCallbackNode)
        )
        assert callback_node.callback is expected["callback"]

    def assert_exec_uses(graph, expected):
        called.clear()
        stream = device.create_stream()
        graph.launch(stream)
        stream.sync()
        assert called == [expected["callback"]]

    return _DefinitionUpdateCase(
        graph_def=graph_def,
        node=node,
        original=original,
        replacement=replacement,
        update=update,
        assert_current=assert_current,
        assert_exec_uses=assert_exec_uses,
        invalid_update=lambda: node.update(invalid_child),
        invalid_exception=CUDAError,
        invalid_argument_update=lambda: node.update(object()),
        cleanup=node.destroy,
    )


@pytest.fixture(
    params=[
        pytest.param(_event_record_case, id="event-record"),
        pytest.param(_event_wait_case, id="event-wait"),
        pytest.param(_host_callback_case, id="host-callback-python"),
        pytest.param(_host_callback_ctypes_case, id="host-callback-ctypes"),
        pytest.param(_memset_value_case, id="memset-value"),
        pytest.param(_memset_destination_case, id="memset-destination"),
        pytest.param(_memcpy_size_case, id="memcpy-size"),
        pytest.param(_memcpy_source_case, id="memcpy-source"),
        pytest.param(_memcpy_destination_case, id="memcpy-destination"),
        pytest.param(_kernel_config_case, id="kernel-config"),
        pytest.param(_kernel_args_case, id="kernel-args"),
        pytest.param(_kernel_function_case, id="kernel-function"),
        pytest.param(_child_graph_case, id="child-graph"),
    ]
)
def definition_update_case(request, init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")
    case = request.param(init_cuda)
    yield case
    case.cleanup()


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
