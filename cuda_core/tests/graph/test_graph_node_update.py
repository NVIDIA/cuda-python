# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for updating individual graph node parameters."""

import ctypes
import gc
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable

import pytest
from helpers.graph_kernels import compile_common_kernels

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource
from cuda.core._utils._weak_handles import weak_handle
from cuda.core._utils.cuda_utils import CUDAError, driver, handle_return
from cuda.core._utils.version import driver_version
from cuda.core.graph import (
    ChildGraphNode,
    EventRecordNode,
    EventWaitNode,
    ExecutableGraphNode,
    GraphDefinition,
    HostCallbackNode,
    KernelNode,
    MemcpyNode,
    MemsetNode,
)


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


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not satisfied within {timeout}s")
        gc.collect()
        time.sleep(0.02)


def _update_executable_case(graph, case):
    view = graph[case.node]
    replacement = case.replacement
    if isinstance(case.node, (EventRecordNode, EventWaitNode)):
        view.update(replacement)
    elif isinstance(case.node, HostCallbackNode):
        if isinstance(replacement, tuple):
            view.update(replacement[0], user_data=replacement[1])
        else:
            view.update(replacement)
    elif isinstance(case.node, MemsetNode):
        view.update(
            dst=replacement["dst"],
            value=replacement["value"],
            width=replacement["width"],
            height=replacement["height"],
            pitch=replacement["pitch"],
        )
    elif isinstance(case.node, MemcpyNode):
        view.update(
            dst=replacement["dst"],
            src=replacement["src"],
            size=replacement["size"],
        )
    elif isinstance(case.node, KernelNode):
        view.update(
            config=replacement["config"],
            kernel=replacement["kernel"],
            args=replacement["args"],
        )
    elif isinstance(case.node, ChildGraphNode):
        view.update(replacement["child"])
    else:  # pragma: no cover - fixture cases are exhaustive
        raise AssertionError(f"unsupported case: {type(case.node).__name__}")


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
        invalid_exception=RuntimeError,
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
        invalid_exception=RuntimeError,
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
        invalid_argument_update=lambda: node.update(object()),
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
    factory = request.param
    # pytest-run-parallel shares this fixture object across workers. Build the
    # case at call time on Device() so each worker gets its own graph/node.
    return lambda: factory(Device())


@pytest.mark.agent_authored(model="gpt-5.6")
def test_memcpy_update_rejects_unsupported_descriptor(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    memory_resource = LegacyPinnedMemoryResource()
    src = memory_resource.allocate(8)
    dst = memory_resource.allocate(8)
    graph_def = GraphDefinition()
    node = graph_def.memcpy(dst, src, 4)

    # cuda.core cannot construct this descriptor, but imported graphs can
    # contain one; use cuda.bindings to exercise that rejection path.
    params = driver.CUDA_MEMCPY3D()
    params.srcXInBytes = 1
    params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_HOST
    params.srcHost = int(src.handle)
    params.srcPitch = 4
    params.srcHeight = 2
    params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_HOST
    params.dstHost = int(dst.handle)
    params.dstPitch = 4
    params.dstHeight = 2
    params.WidthInBytes = 2
    params.Height = 2
    params.Depth = 1
    handle_return(driver.cuGraphMemcpyNodeSetParams(node.handle, params))

    with pytest.raises(NotImplementedError, match="multidimensional"):
        node.update(size=3)

    unchanged = handle_return(driver.cuGraphMemcpyNodeGetParams(node.handle))
    assert unchanged.srcXInBytes == 1
    assert unchanged.WidthInBytes == 2
    assert unchanged.Height == 2


@pytest.mark.agent_authored(model="gpt-5.6")
def test_kernel_update_rejects_unsupported_config(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    kernel = compile_common_kernels().get_kernel("empty_kernel")
    graph_def = GraphDefinition()
    node = graph_def.launch(LaunchConfig(grid=1, block=1), kernel)

    clustered = LaunchConfig(grid=1, block=1)
    clustered.cluster = (1, 1, 1)
    with pytest.raises(NotImplementedError, match="clustered or cooperative"):
        node.update(config=clustered)
    with pytest.raises(NotImplementedError, match="clustered or cooperative"):
        graph_def.launch(clustered, kernel)

    cooperative = LaunchConfig(grid=1, block=1)
    cooperative.is_cooperative = True
    with pytest.raises(NotImplementedError, match="clustered or cooperative"):
        node.update(config=cooperative)
    with pytest.raises(NotImplementedError, match="clustered or cooperative"):
        graph_def.launch(cooperative, kernel)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_partial_memory_updates_are_keyword_only(init_cuda):
    memory_resource = LegacyPinnedMemoryResource()
    src = memory_resource.allocate(4)
    dst = memory_resource.allocate(4)
    graph_def = GraphDefinition()
    memset_node = graph_def.memset(dst, 0, 4)
    memcpy_node = graph_def.memcpy(dst, src, 4)

    with pytest.raises(TypeError):
        memset_node.update(dst)
    with pytest.raises(TypeError):
        memcpy_node.update(dst)


@pytest.mark.parametrize(
    "device_operand",
    [
        pytest.param("src", id="device-to-host"),
        pytest.param("dst", id="host-to-device"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_memcpy_update_between_host_and_device(init_cuda, device_operand):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    memory_resource = LegacyPinnedMemoryResource()
    host_src = memory_resource.allocate(4)
    host_dst = memory_resource.allocate(4)
    host_src_bytes = (ctypes.c_uint8 * 4).from_address(int(host_src.handle))
    host_dst_bytes = (ctypes.c_uint8 * 4).from_address(int(host_dst.handle))
    host_src_bytes[:] = [0x5A] * 4
    host_dst_bytes[:] = [0] * 4

    stream = init_cuda.create_stream()
    device_buffer = init_cuda.memory_resource.allocate(4, stream=stream)
    device_buffer.fill(0, stream=stream)
    if device_operand == "src":
        device_buffer.copy_from(host_src, stream=stream)
    stream.sync()

    graph_def = GraphDefinition()
    node = graph_def.memcpy(host_dst, host_src, 4)
    if device_operand == "src":
        node.update(src=device_buffer)
    else:
        node.update(dst=device_buffer)

    graph = graph_def.instantiate()
    graph.launch(stream)
    if device_operand == "dst":
        device_buffer.copy_to(host_dst, stream=stream)
    stream.sync()

    assert list(host_dst_bytes) == [0x5A] * 4


@pytest.mark.agent_authored(model="gpt-5.6")
def test_definition_node_update_changes_future_instantiations(
    definition_update_case,
):
    case = definition_update_case()
    assert case.original != case.replacement
    old_graph = case.graph_def.instantiate()

    case.update(case.replacement)
    case.assert_current(case.replacement)

    new_graph = case.graph_def.instantiate()
    assert old_graph != new_graph
    case.assert_exec_uses(old_graph, case.original)
    case.assert_exec_uses(new_graph, case.replacement)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_destroyed_definition_node_rejects_update(
    definition_update_case,
):
    case = definition_update_case()
    case.node.destroy()

    assert not case.node.is_valid
    assert case.node not in case.graph_def.nodes()
    with pytest.raises(RuntimeError, match="GraphNode has been destroyed"):
        case.update(case.replacement)
    assert not case.node.is_valid
    assert case.node not in case.graph_def.nodes()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_failed_definition_node_update_preserves_state(
    definition_update_case,
):
    case = definition_update_case()

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
    case = definition_update_case()
    if case.invalid_argument_update is None:
        pytest.skip("update method has no typed positional argument")
    with pytest.raises(TypeError):
        case.invalid_argument_update()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_executable_node_update_changes_existing_exec(
    definition_update_case,
):
    case = definition_update_case()
    graph = case.graph_def.instantiate()

    _update_executable_case(graph, case)

    case.assert_current(case.original)
    case.assert_exec_uses(graph, case.replacement)


@pytest.mark.parametrize("node_kind", ["kernel", "memcpy", "memset"])
@pytest.mark.agent_authored(model="gpt-5.6")
def test_executable_node_enable_state(init_cuda, node_kind):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    graph_def = GraphDefinition()
    if node_kind == "kernel":
        kernel = compile_common_kernels().get_kernel("empty_kernel")
        node = graph_def.launch(LaunchConfig(grid=1, block=1), kernel)
    else:
        memory_resource = LegacyPinnedMemoryResource()
        src = memory_resource.allocate(4)
        dst = memory_resource.allocate(4)
        if node_kind == "memcpy":
            node = graph_def.memcpy(dst, src, 4)
        else:
            node = graph_def.memset(dst, 0, 4)

    view = graph_def.instantiate()[node]
    assert view.is_enabled
    view.disable()
    assert not view.is_enabled
    view.disable()
    assert not view.is_enabled
    view.enable()
    assert view.is_enabled
    view.enable()
    assert view.is_enabled


@pytest.mark.agent_authored(model="gpt-5.6")
def test_executable_node_view_rejects_unsupported_and_destroyed_nodes(
    init_cuda,
):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    kernel = compile_common_kernels().get_kernel("empty_kernel")
    graph_def = GraphDefinition()
    empty = graph_def.empty()
    kernel_node = graph_def.launch(LaunchConfig(grid=1, block=1), kernel)
    graph = graph_def.instantiate()

    with pytest.raises(TypeError, match="does not support executable updates"):
        graph[empty]
    with pytest.raises(TypeError):
        graph[object()]

    kernel_node.destroy()
    with pytest.raises(RuntimeError, match="GraphNode has been destroyed"):
        graph[kernel_node]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_executable_node_view_retains_source_only_while_live(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def replacement():
        called.append("replacement")

    source = GraphDefinition()
    node = source.callback(original)
    source_weak = weak_handle(source)
    graph = source.instantiate()
    view = graph[node]

    del source, node
    gc.collect()
    assert source_weak

    view.update(replacement)
    del view
    _wait_until(lambda: not source_weak)

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert called == ["replacement"]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_executable_attachment_accumulators_are_independent(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def first_replacement():
        called.append("first")

    def second_replacement():
        called.append("second")

    first_weak = weakref.ref(first_replacement)
    second_weak = weakref.ref(second_replacement)
    source = GraphDefinition()
    node = source.callback(original)
    first = source.instantiate()
    second = source.instantiate()

    first[node].update(first_replacement)
    second[node].update(second_replacement)
    del first_replacement, second_replacement, original, node, source
    gc.collect()
    assert first_weak() is not None
    assert second_weak() is not None

    stream = init_cuda.create_stream()
    first.launch(stream)
    second.launch(stream)
    stream.sync()
    assert called == ["first", "second"]

    del first
    _wait_until(lambda: first_weak() is None)
    assert second_weak() is not None

    del second
    _wait_until(lambda: second_weak() is None)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_rejected_executable_update_rolls_back_owners(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    kernel = compile_common_kernels().get_kernel("add_one")
    config = LaunchConfig(grid=1, block=1)
    memory_resource = LegacyPinnedMemoryResource()
    active = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    rejected = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    ctypes.c_int.from_address(int(active.handle)).value = 0
    rejected_weak = weak_handle(rejected)

    source = GraphDefinition()
    source.launch(config, kernel, active)
    graph = source.instantiate()
    unrelated = GraphDefinition()
    unrelated_node = unrelated.launch(config, kernel, active)

    with pytest.raises(CUDAError):
        graph[unrelated_node].update(config=config, kernel=kernel, args=(rejected,))

    del rejected
    _wait_until(lambda: not rejected_weak)

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert ctypes.c_int.from_address(int(active.handle)).value == 1


@pytest.mark.thread_unsafe(reason="deferred cleanup on main thread which would wait")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_whole_update_replaces_executable_attachment_accumulator(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def individual():
        called.append("individual")

    def whole():
        called.append("whole")

    individual_weak = weakref.ref(individual)
    source = GraphDefinition()
    node = source.callback(original)
    graph = source.instantiate()
    graph[node].update(individual)

    replacement = GraphDefinition()
    replacement.callback(whole)
    del individual
    graph.update(replacement)
    _wait_until(lambda: individual_weak() is None)

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert called == ["whole"]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_failed_whole_update_preserves_executable_accumulator(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def active():
        called.append("active")

    active_weak = weakref.ref(active)
    source = GraphDefinition()
    node = source.callback(original)
    graph = source.instantiate()
    graph[node].update(active)

    rejected = GraphDefinition()
    rejected.callback(lambda: called.append("rejected"))
    rejected.empty()
    with pytest.raises(CUDAError):
        graph.update(rejected)

    del active, original, node, source, rejected
    gc.collect()
    assert active_weak() is not None

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert called == ["active"]

    del graph
    _wait_until(lambda: active_weak() is None)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_inflight_launch_defers_replaced_executable_owners(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    callback_started = threading.Event()
    callback_release = threading.Event()

    def blocking_callback():
        callback_started.set()
        assert callback_release.wait(timeout=30)

    kernel = compile_common_kernels().get_kernel("add_one")
    config = LaunchConfig(grid=1, block=1)
    memory_resource = LegacyPinnedMemoryResource()
    original = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    inflight = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    future = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    inflight_weak = weak_handle(inflight)

    source = GraphDefinition()
    kernel_node = source.callback(blocking_callback).launch(config, kernel, original)
    graph = source.instantiate()
    graph[kernel_node].update(config=config, kernel=kernel, args=(inflight,))
    del inflight
    gc.collect()
    assert inflight_weak

    replacement = GraphDefinition()
    replacement.callback(lambda: None).launch(config, kernel, future)
    stream = init_cuda.create_stream()
    graph.launch(stream)
    assert callback_started.wait(timeout=5)

    try:
        graph.update(replacement)
        gc.collect()
        assert inflight_weak
    finally:
        callback_release.set()
        stream.sync()

    _wait_until(lambda: not inflight_weak)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_sequential_executable_updates_accumulate_owners(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def first():
        called.append("first")

    def second():
        called.append("second")

    first_weak = weakref.ref(first)
    second_weak = weakref.ref(second)
    source = GraphDefinition()
    node = source.callback(original)
    graph = source.instantiate()

    graph[node].update(first)
    graph[node].update(second)

    # CUDA cannot detach user objects from an executable graph, so the
    # superseded owner stays reachable for as long as the executable lives.
    del first, second, original
    gc.collect()
    assert first_weak() is not None
    assert second_weak() is not None

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert called == ["second"]

    del node, source, graph
    _wait_until(lambda: first_weak() is None and second_weak() is None)


@pytest.mark.thread_unsafe(reason="deferred cleanup on main thread which would wait")
@pytest.mark.agent_authored(model="claude-opus-5")
def test_child_graph_update_transfers_source_owners_to_executable(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    called = []

    def original():
        called.append("original")

    def replacement():
        called.append("replacement")

    original_child = GraphDefinition()
    original_child.callback(original)
    source = GraphDefinition()
    node = source.embed(original_child)
    graph = source.instantiate()

    replacement_child = GraphDefinition()
    replacement_child.callback(replacement)
    graph[node].update(replacement_child)

    replacement_weak = weakref.ref(replacement)
    child_weak = weak_handle(replacement_child)

    # A child-graph update is the one executable update that attaches no owner
    # of its own. It is safe because CUDA clones the replacement graph's user
    # object references into the executable, so the callback must outlive the
    # definition that supplied it.
    del replacement_child, replacement
    _wait_until(lambda: not child_weak)
    assert replacement_weak() is not None

    stream = init_cuda.create_stream()
    graph.launch(stream)
    stream.sync()
    assert called == ["replacement"]

    del graph
    _wait_until(lambda: replacement_weak() is None)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_closing_executable_during_launch_defers_owner_release(init_cuda):
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    callback_started = threading.Event()
    callback_release = threading.Event()

    def blocking_callback():
        callback_started.set()
        assert callback_release.wait(timeout=30)

    kernel = compile_common_kernels().get_kernel("add_one")
    config = LaunchConfig(grid=1, block=1)
    memory_resource = LegacyPinnedMemoryResource()
    original = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    inflight = memory_resource.allocate(ctypes.sizeof(ctypes.c_int))
    inflight_weak = weak_handle(inflight)

    source = GraphDefinition()
    kernel_node = source.callback(blocking_callback).launch(config, kernel, original)
    graph = source.instantiate()
    graph[kernel_node].update(config=config, kernel=kernel, args=(inflight,))
    del inflight
    gc.collect()
    assert inflight_weak

    stream = init_cuda.create_stream()
    graph.launch(stream)
    assert callback_started.wait(timeout=5)

    try:
        # The launch still writes through the buffer the update attached, so
        # closing the executable must not retire the accumulator yet.
        graph.close()
        gc.collect()
        assert inflight_weak
    finally:
        callback_release.set()
        stream.sync()

    _wait_until(lambda: not inflight_weak)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_memory_node_update_validates_owners_and_noops(init_cuda):
    """Memory-node updates validate owners, preserve no-ops, and update geometry."""
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    memory_resource = LegacyPinnedMemoryResource()
    with memory_resource.allocate(16) as src, memory_resource.allocate(16) as dst:
        graph_def = GraphDefinition()
        memset_node = graph_def.memset(dst, 0x11, 8)
        memcpy_node = graph_def.memcpy(dst, src, 8)

        with pytest.raises(ValueError, match=r"^dst_owner requires dst$"):
            memset_node.update(dst_owner=dst)
        memset_node.update()
        assert memset_node.value == 0x11
        assert memset_node.width == 8

        memset_node.update(width=4, height=2, pitch=8)
        assert memset_node.width == 4
        assert memset_node.height == 2
        assert memset_node.pitch == 8

        with pytest.raises(ValueError, match=r"^dst_owner requires dst$"):
            memcpy_node.update(dst_owner=dst)
        with pytest.raises(ValueError, match=r"^src_owner requires src$"):
            memcpy_node.update(src_owner=src)
        memcpy_node.update()
        assert memcpy_node.size == 8
        assert memcpy_node.dst == int(dst.handle)
        assert memcpy_node.src == int(src.handle)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_executable_graph_node_cannot_be_constructed_directly():
    """Executable-node views are factory-only and fail before any CUDA call."""
    with pytest.raises(RuntimeError, match=r"^directly constructing an executable graph node is not supported$"):
        ExecutableGraphNode()


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_executable_node_repr_reports_graph_and_node(init_cuda):
    """An executable-node view reprs its subclass name and both handles."""
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    kernel = compile_common_kernels().get_kernel("empty_kernel")
    graph_def = GraphDefinition()
    node = graph_def.launch(LaunchConfig(grid=1, block=1), kernel)
    graph = graph_def.instantiate()

    assert repr(graph[node]) == f"<ExecutableKernelNode graph=0x{int(graph.handle):x} node=0x{int(node.handle):x}>"


@pytest.mark.parametrize("config_kind", ["clustered", "cooperative"])
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_executable_kernel_update_rejects_unsupported_config(init_cuda, config_kind):
    """Executable kernel updates reject clustered and cooperative launches."""
    if driver_version() < (12, 2, 0):
        pytest.skip("individual graph node updates require CUDA 12.2+")

    kernel = compile_common_kernels().get_kernel("empty_kernel")
    graph_def = GraphDefinition()
    node = graph_def.launch(LaunchConfig(grid=1, block=1), kernel)
    view = graph_def.instantiate()[node]

    config = LaunchConfig(grid=1, block=1)
    if config_kind == "clustered":
        config.cluster = (1, 1, 1)
    else:
        config.is_cooperative = True
    with pytest.raises(
        NotImplementedError,
        match=r"^updating clustered or cooperative kernel nodes is not supported$",
    ):
        view.update(config=config, kernel=kernel, args=())


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_ctypes_host_callback_repr(init_cuda):
    """A ctypes host callback repr reports its node and function addresses."""
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    @callback_type
    def host_fn(_user_data):
        return None

    graph_def = GraphDefinition()
    node = graph_def.callback(host_fn)
    assert isinstance(node, HostCallbackNode)
    assert node.callback is None
    cfunc = ctypes.cast(host_fn, ctypes.c_void_p).value
    assert cfunc is not None
    assert repr(node) == f"<HostCallbackNode handle=0x{int(node.handle):x} cfunc=0x{cfunc:x}>"
