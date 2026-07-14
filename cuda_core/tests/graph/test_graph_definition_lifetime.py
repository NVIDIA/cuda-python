# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GraphDefinition resource lifetime management and RAII correctness."""

import ctypes
import gc
import subprocess
import sys
import textwrap
import threading
import time
import weakref

import pytest
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from conftest import xfail_on_graph_mempool_oom
from cuda_python_test_helpers import under_compute_sanitizer

# Resource finalization triggered by graph destruction is not synchronous. A
# CUDA user-object callback transfers the slot table to a pending-call
# cleanup queue, which releases each owner from Python's main thread. Release is
# deterministic at the reference-count level, so the predicate normally flips
# within milliseconds; this budget only bounds a slow/loaded runner. It stays a
# hard failure rather than a warning so a real leak still fails the suite.
# Compute-sanitizer slows everything down, hence the larger ceiling there.
_FINALIZE_TIMEOUT = 30.0 if under_compute_sanitizer() else 5.0


class _Sentinel:
    """Weak-referenceable stand-in for an owner attached to a graph slot.

    Bare ``object()`` instances do not support weak references, so tests that
    observe owner release through a :class:`weakref.ref` use this trivial
    subclass instead.
    """


class _ThreadRecordingCallback:
    def __init__(self, finalized_threads):
        self.finalized_threads = finalized_threads

    def __call__(self):
        pass

    def __del__(self):
        self.finalized_threads.append(threading.get_ident())


def _wait_until(predicate, timeout=None, interval=0.02):
    """Poll ``predicate()`` until true, or raise AssertionError on timeout.

    Each iteration drives ``gc.collect()`` and reaches bytecode boundaries so
    pending cleanup can run. Used for resource cleanup that lags graph
    destruction; see ``_FINALIZE_TIMEOUT``.
    """
    if timeout is None:
        timeout = _FINALIZE_TIMEOUT
    deadline = time.monotonic() + timeout
    while True:
        gc.collect()
        if predicate():
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0)  # yield the GIL to the driver's finalizer thread
        time.sleep(interval)
    # Final attempt after one more yield and collection.
    time.sleep(0)
    gc.collect()
    if predicate():
        return
    raise AssertionError(f"condition not satisfied within {timeout}s")


from cuda.core import Device, DeviceMemoryResource, EventOptions, Kernel, LaunchConfig
from cuda.core.graph import (
    ChildGraphNode,
    ConditionalNode,
    GraphDefinition,
    KernelNode,
)


def _skip_if_no_mempool():
    if not Device(0).properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")


# =============================================================================
# Conditional body graph lifetime
# =============================================================================


def _make_if(g, cond):
    node = g.if_then(cond)
    return [node.then]


def _make_if_else(g, cond):
    node = g.if_else(cond)
    return [node.then, node.else_]


def _make_while(g, cond):
    node = g.while_loop(cond)
    return [node.body]


def _make_switch(g, cond):
    node = g.switch(cond, 4)
    return list(node.branches)


_COND_BUILDERS = [
    pytest.param(_make_if, 1, id="if"),
    pytest.param(_make_if_else, 2, id="if_else"),
    pytest.param(_make_while, 1, id="while"),
    pytest.param(_make_switch, 4, id="switch"),
]


@pytest.mark.parametrize("builder, expected_count", _COND_BUILDERS)
def test_branches_survive_parent_deletion(init_cuda, builder, expected_count):
    """All branch graphs remain valid after parent GraphDefinition is deleted."""
    g = GraphDefinition()
    condition = try_create_condition(g)
    branches = builder(g, condition)
    assert len(branches) == expected_count

    del g, condition
    gc.collect()

    for branch in branches:
        assert branch.nodes() == set()


@pytest.mark.parametrize("builder, expected_count", _COND_BUILDERS)
def test_branches_usable_after_parent_deletion(init_cuda, builder, expected_count):
    """Nodes can be added to branch graphs after parent GraphDefinition is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    condition = try_create_condition(g)
    branches = builder(g, condition)

    del g, condition
    gc.collect()

    for branch in branches:
        branch.launch(config, kernel)
        assert len(branch.nodes()) == 1


def test_reconstructed_body_survives_parent_deletion(init_cuda):
    """Body graph obtained via nodes() reconstruction survives parent deletion."""
    g = GraphDefinition()
    condition = try_create_condition(g)
    g.while_loop(condition)

    all_nodes = g.nodes()
    cond_nodes = [n for n in all_nodes if isinstance(n, ConditionalNode)]
    assert len(cond_nodes) == 1

    branches = cond_nodes[0].branches
    if not branches:
        pytest.skip("Body reconstruction requires CUDA 13.2+")
    body = branches[0]

    del g, condition, all_nodes, cond_nodes, branches
    gc.collect()

    assert body.nodes() == set()


# =============================================================================
# Child graph (embed) lifetime
# =============================================================================


def test_child_graph_survives_parent_deletion(init_cuda):
    """Embedded child graph remains valid after parent GraphDefinition is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    child_def = GraphDefinition()
    child_def.launch(config, kernel)
    child_def.launch(config, kernel)

    g = GraphDefinition()
    node = g.embed(child_def)
    child_ref = node.child_graph

    del g, node, child_def
    gc.collect()

    assert len(child_ref.nodes()) == 2


def test_nested_child_graph_lifetime(init_cuda):
    """Grandchild graph keeps entire ancestor chain alive."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    inner = GraphDefinition()
    inner.launch(config, kernel)

    middle = GraphDefinition()
    middle.embed(inner)

    outer = GraphDefinition()
    outer_node = outer.embed(middle)

    middle_ref = outer_node.child_graph
    middle_nodes = middle_ref.nodes()
    child_node = next(n for n in middle_nodes if isinstance(n, ChildGraphNode))
    grandchild = child_node.child_graph

    del outer, outer_node, middle, inner, middle_ref, middle_nodes, child_node
    gc.collect()

    assert len(grandchild.nodes()) == 1


# =============================================================================
# Event lifetime — event nodes should keep the Event alive
# =============================================================================


def test_event_record_node_keeps_event_alive(init_cuda):
    """EventRecordNode should keep the Event alive after original is deleted."""
    _skip_if_no_mempool()
    dev = Device()
    g = GraphDefinition()
    with xfail_on_graph_mempool_oom(dev):
        alloc = g.allocate(1024)

    event = dev.create_event(EventOptions(timing_enabled=False))
    node = alloc.record(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_done is True


def test_event_wait_node_keeps_event_alive(init_cuda):
    """EventWaitNode should keep the Event alive after original is deleted."""
    _skip_if_no_mempool()
    dev = Device()
    g = GraphDefinition()
    with xfail_on_graph_mempool_oom(dev):
        alloc = g.allocate(1024)

    event = dev.create_event(EventOptions(timing_enabled=False))
    node = alloc.wait(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_done is True


def test_event_record_node_preserves_metadata(init_cuda):
    """Reconstructed EventRecordNode recovers full Event metadata via reverse lookup."""
    dev = Device()
    g = GraphDefinition()

    event = dev.create_event(EventOptions(timing_enabled=True, blocking_sync=True))
    node = g.record(event)

    reconstructed = node.event
    assert reconstructed.is_timing_enabled is True
    assert reconstructed.is_blocking_sync is True
    assert reconstructed.is_ipc_enabled is False
    assert reconstructed.device is not None


def test_event_wait_node_preserves_metadata(init_cuda):
    """Reconstructed EventWaitNode recovers full Event metadata via reverse lookup."""
    dev = Device()
    g = GraphDefinition()

    event = dev.create_event(EventOptions(timing_enabled=False))
    node = g.wait(event)

    reconstructed = node.event
    assert reconstructed.is_timing_enabled is False
    assert reconstructed.is_blocking_sync is False
    assert reconstructed.device is not None


def test_event_metadata_survives_gc(init_cuda):
    """Event metadata is preserved through reverse lookup even after original is GC'd."""
    dev = Device()
    g = GraphDefinition()

    event = dev.create_event(EventOptions(timing_enabled=True, blocking_sync=True))
    node = g.record(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_timing_enabled is True
    assert retrieved.is_blocking_sync is True
    assert retrieved.is_done is True


def test_event_survives_graph_instantiation_and_execution(init_cuda):
    """Graph with event nodes executes correctly after original Event is deleted."""
    dev = Device()
    g = GraphDefinition()

    event = dev.create_event(EventOptions(timing_enabled=False))
    rec = g.record(event)
    rec.wait(event)

    del event
    gc.collect()

    graph = g.instantiate()
    stream = dev.create_stream()
    graph.launch(stream)
    stream.sync()


def test_event_survives_graph_clone_and_execution(init_cuda):
    """Cloned graph with event nodes executes after original Event is deleted.

    This is the critical test for CUDA User Objects: a graph clone does
    not inherit Python-level references, so only user objects (which
    propagate through cuGraphClone) can keep the event alive.
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    dev = Device()
    g = GraphDefinition()

    event = dev.create_event(EventOptions(timing_enabled=False))
    rec = g.record(event)
    rec.wait(event)

    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del event, g, rec
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()


# =============================================================================
# Host callback lifetime — callbacks and user_data tied to graph
# =============================================================================


@pytest.mark.agent_authored(model="gpt-5.6")
def test_user_object_cleanup_is_coalesced_on_python_thread(init_cuda):
    """More than 32 CUDA callbacks drain through one main-thread pending call."""
    finalized_threads = []
    main_thread = threading.get_ident()
    graphs = []

    for _ in range(64):
        callback = _ThreadRecordingCallback(finalized_threads)
        graph = GraphDefinition()
        graph.callback(callback)
        graphs.append(graph)

    del callback, graph, graphs
    _wait_until(lambda: len(finalized_threads) == 64)
    assert set(finalized_threads) == {main_thread}


@pytest.mark.agent_authored(model="gpt-5.6")
def test_pending_call_scheduling_failure_retries_later(init_cuda):
    """A full CPython queue delays reclamation until a later safe retry."""
    pending_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    add_pending_call = ctypes.pythonapi.Py_AddPendingCall
    add_pending_call.argtypes = [pending_callback_type, ctypes.c_void_p]
    add_pending_call.restype = ctypes.c_int

    @pending_callback_type
    def noop_pending_call(_):
        return 0

    finalized_threads = []
    main_thread = threading.get_ident()
    first_callback = _ThreadRecordingCallback(finalized_threads)
    first_graph = GraphDefinition()
    first_graph.callback(first_callback)
    graph_holder = [first_graph]
    worker_done = threading.Event()
    queue_was_full = []

    del first_callback, first_graph

    def fill_queue_and_destroy():
        while add_pending_call(noop_pending_call, None) == 0:
            pass
        queue_was_full.append(True)
        graph_holder.clear()
        worker_done.set()

    worker = threading.Thread(target=fill_queue_and_destroy)
    worker.start()
    assert worker_done.wait(timeout=5)
    worker.join()
    assert queue_was_full == [True]
    assert finalized_threads == []

    # Preparing another graph attachment is a safe cuda-core entry point that
    # retries scheduling the first graph's still-intact queued payload.
    second_callback = _ThreadRecordingCallback(finalized_threads)
    second_graph = GraphDefinition()
    second_graph.callback(second_callback)
    del second_callback, second_graph

    _wait_until(lambda: len(finalized_threads) == 2)
    assert set(finalized_threads) == {main_thread}


@pytest.mark.agent_authored(model="gpt-5.6")
def test_pending_cleanup_is_safe_during_python_shutdown(init_cuda):
    """Outstanding graph attachments neither call Python nor hang at shutdown."""
    code = textwrap.dedent(
        """
        from cuda.core import Device
        from cuda.core.graph import GraphDefinition

        class Callback:
            def __call__(self):
                pass

        Device()
        graph = GraphDefinition()
        graph.callback(Callback())
        """
    )
    result = subprocess.run(  # noqa: S603 - controlled interpreter probe
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_python_callable_callback_survives_del(init_cuda):
    """Python callable is kept alive by the graph after Python ref is dropped."""
    called = [False]

    def my_callback():
        called[0] = True

    g = GraphDefinition()
    g.callback(my_callback)

    del my_callback
    gc.collect()

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert called[0]


def test_cfunc_callback_survives_del(init_cuda):
    """ctypes CFUNCTYPE wrapper is kept alive by the graph after Python ref is dropped."""
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    called = [False]

    @CALLBACK
    def raw_fn(data):
        called[0] = True

    g = GraphDefinition()
    g.callback(raw_fn)

    del raw_fn
    gc.collect()

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert called[0]


def test_cfunc_bytes_user_data_survives_del(init_cuda):
    """Bytes-backed user_data is kept alive by the graph after Python ref is dropped."""
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    result = [0]

    @CALLBACK
    def read_byte(data):
        result[0] = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))[0]

    payload = bytes([0xCD])
    g = GraphDefinition()
    g.callback(read_byte, user_data=payload)

    del payload
    gc.collect()

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert result[0] == 0xCD


# =============================================================================
# Kernel lifetime — kernel nodes should keep the Kernel/Module alive
# =============================================================================


def test_kernel_node_keeps_kernel_alive(init_cuda):
    """KernelNode should keep the Kernel alive after original is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    node = g.launch(config, kernel)

    del kernel, mod
    gc.collect()

    retrieved = node.kernel
    assert retrieved.attributes.max_threads_per_block > 0


def test_kernel_survives_graph_instantiation_and_execution(init_cuda):
    """Graph with kernel node executes correctly after Kernel/Module is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    g.launch(config, kernel)

    del kernel, mod
    gc.collect()

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()


def test_kernel_survives_graph_clone_and_execution(init_cuda):
    """Cloned graph with kernel node executes after Kernel/Module is deleted.

    Validates that CUDA User Objects keep the kernel's library alive
    through graph cloning (where Python-level references are lost).
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    dev = Device()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    g.launch(config, kernel)

    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del kernel, mod, g
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()


# =============================================================================
# Kernel handle recovery — from_handle and graph node reconstruction
# =============================================================================


def test_kernel_from_handle_recovers_library(init_cuda):
    """Kernel.from_handle on a cuda.core-created kernel recovers the library
    dependency, keeping it alive after the original objects are deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    handle = int(kernel.handle)

    reconstructed = Kernel.from_handle(handle)

    del kernel, mod
    gc.collect()

    assert reconstructed.attributes.max_threads_per_block > 0


def test_kernel_node_reconstruction_preserves_validity(init_cuda):
    """A KernelNode reconstructed via DAG traversal has a valid kernel,
    kept alive by user objects and existing node references."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    kernel_node = g.launch(config, kernel)
    # Chain a second node so we can reconstruct the kernel node via pred
    event = Device().create_event()
    successor = kernel_node.record(event)

    del kernel, mod
    gc.collect()

    # Reconstruct the kernel node through DAG traversal
    # successor.pred -> GraphNode._create -> KernelNode._create_from_driver
    # -> create_kernel_handle_ref -> handle recovery
    reconstructed = next(iter(successor.pred))
    assert isinstance(reconstructed, KernelNode)
    assert reconstructed.kernel.attributes.max_threads_per_block > 0

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()


# =============================================================================
# Kernel argument lifetime — kernel nodes should keep argument objects alive
# =============================================================================


def test_kernel_args_buffer_lifetime(init_cuda):
    """Buffer passed as a kernel arg is kept alive by the graph, the kernel
    executes against its memory after the original Python ref drops, and the
    Buffer is released once the graph is destroyed.

    Without the user-object attachment, the ParamHolder is destroyed when the
    kernel node is added, the Buffer is GC'd, and the graph is left with a
    stale device pointer.

    The final freeing assertion uses a bounded poll because CUgraphExec
    releases its user-object references via an asynchronous DPC, and on
    free-threaded Python the resulting Py_DECREF chain may need an extra
    GC pass to settle.
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    add_one = compile_common_kernels().get_kernel("add_one")
    buf = mr.allocate(ctypes.sizeof(ctypes.c_int), stream=dev.default_stream)
    buf.fill(0, stream=dev.default_stream)
    dev.default_stream.sync()
    buf_weak = weakref.ref(buf)
    dptr = int(buf.handle)

    g = GraphDefinition()
    g.launch(LaunchConfig(grid=1, block=1), add_one, buf)

    del buf
    gc.collect()
    assert buf_weak() is not None  # graph kept the Buffer alive

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_int * 1)(0)
    handle_return(driver.cuMemcpyDtoH(out, dptr, ctypes.sizeof(ctypes.c_int)))
    assert out[0] == 1

    del g
    _wait_until(lambda: buf_weak() is None)


def test_kernel_args_survive_graph_clone(init_cuda):
    """Cloned graph keeps Buffer alive via CUDA user objects.

    A graph clone does not inherit Python-level references, so only user
    objects (which propagate through cuGraphClone) can keep the args alive.
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    add_one = compile_common_kernels().get_kernel("add_one")
    buf = mr.allocate(ctypes.sizeof(ctypes.c_int), stream=dev.default_stream)
    buf.fill(0, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)

    g = GraphDefinition()
    g.launch(LaunchConfig(grid=1, block=1), add_one, buf)
    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del buf, g
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()

    out = (ctypes.c_int * 1)(0)
    handle_return(driver.cuMemcpyDtoH(out, dptr, ctypes.sizeof(ctypes.c_int)))
    assert out[0] == 1


# =============================================================================
# Memcpy/memset Buffer lifetime — operands passed as Buffer objects
# =============================================================================


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memset_buffer_lifetime(init_cuda):
    """Memset retains the Buffer allocation after the wrapper is collected."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)

    g = GraphDefinition()
    g.memset(buf, 0xAB, 4)

    del buf
    gc.collect()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dptr, 4))
    assert list(out) == [0xAB] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memset_buffer_survives_close(init_cuda):
    """Memset retains the allocation when the Buffer wrapper is closed."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)

    g = GraphDefinition()
    g.memset(buf, 0xAB, 4)
    buf.close()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dptr, 4))
    assert list(out) == [0xAB] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffer_lifetime(init_cuda):
    """Memcpy retains operand allocations after the Buffer wrappers are collected."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    src.fill(0xCD, stream=dev.default_stream)
    dev.default_stream.sync()
    dst_dptr = int(dst.handle)

    g = GraphDefinition()
    g.memcpy(dst, src, 4)

    del src, dst
    gc.collect()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dst_dptr, 4))
    assert list(out) == [0xCD] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffer_survives_close(init_cuda):
    """Memcpy retains allocations when Buffer wrappers are closed."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    src.fill(0xCD, stream=dev.default_stream)
    dev.default_stream.sync()
    dst_dptr = int(dst.handle)

    g = GraphDefinition()
    g.memcpy(dst, src, 4)
    src.close()
    dst.close()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dst_dptr, 4))
    assert list(out) == [0xCD] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffer_allocations_released_after_graph_destroyed(init_cuda):
    """Destroying the graph frees both memcpy operand allocations.

    Each operand's device-pointer handle is observed via a weak handle
    (see ``cuda.core._utils._weak_handles``), so release is checked at the
    reference-count level rather than through a driver side effect. With both
    Buffer wrappers closed, the graph's slots are the only remaining owners;
    destroying the graph releases them and the weak handles expire.
    """
    from cuda.core._utils._weak_handles import weak_handle

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()

    g = GraphDefinition()
    g.memcpy(dst, src, 4)

    # Observe the allocations, then drop the wrappers' strong references; the
    # graph slots remain the sole owners.
    src_weak = weak_handle(src)
    dst_weak = weak_handle(dst)
    src.close()
    dst.close()
    assert src_weak and dst_weak  # graph slots still retain both allocations

    del g
    _wait_until(lambda: not src_weak and not dst_weak)


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffers_survive_graph_clone(init_cuda):
    """Cloned graph keeps memcpy operand allocations alive via CUDA user objects."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    src.fill(0xCD, stream=dev.default_stream)
    dev.default_stream.sync()
    dst_dptr = int(dst.handle)

    g = GraphDefinition()
    g.memcpy(dst, src, 4)
    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del src, dst, g
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dst_dptr, 4))
    assert list(out) == [0xCD] * 4


# =============================================================================
# Explicit dst_owner / src_owner for raw pointer operands
# =============================================================================


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memset_raw_ptr_with_dst_owner(init_cuda):
    """Raw dst plus Buffer dst_owner retains the allocation after close."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)

    g = GraphDefinition()
    g.memset(dptr, 0xAB, 4, dst_owner=buf)
    buf.close()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dptr, 4))
    assert list(out) == [0xAB] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_slot_owners_released_after_graph_destroyed(init_cuda):
    """Destroying the graph releases every owner held in its slot table.

    Raw-pointer operands with explicit sentinel owners make release observable
    in pure Python: the slot table holds a strong Python reference to each owner
    (via ``make_opaque_py``), and graph destruction frees the table -- dropping
    those references. This exercises the same teardown that releases a Buffer
    operand's device-pointer handle (slot 0 for ``dst``, slot 1 for ``src``).
    """
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(8, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)

    dst_owner = _Sentinel()
    src_owner = _Sentinel()
    dst_weak = weakref.ref(dst_owner)
    src_weak = weakref.ref(src_owner)

    g = GraphDefinition()
    # Non-overlapping 4-byte copy within an 8-byte allocation.
    g.memcpy(dptr, dptr + 4, 4, dst_owner=dst_owner, src_owner=src_owner)

    del dst_owner, src_owner
    gc.collect()
    assert dst_weak() is not None and src_weak() is not None  # graph retains owners

    del g
    _wait_until(lambda: dst_weak() is None and src_weak() is None)

    buf.close()


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_raw_ptrs_with_owners(init_cuda):
    """Raw src/dst plus Buffer owners retain allocations after close."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    src.fill(0xCD, stream=dev.default_stream)
    dev.default_stream.sync()
    src_dptr = int(src.handle)
    dst_dptr = int(dst.handle)

    g = GraphDefinition()
    g.memcpy(dst_dptr, src_dptr, 4, dst_owner=dst, src_owner=src)
    src.close()
    dst.close()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dst_dptr, 4))
    assert list(out) == [0xCD] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_mixed_buffer_and_raw_owner(init_cuda):
    """Buffer dst and raw src with src_owner retain allocations after close."""
    from cuda.core._utils.cuda_utils import driver, handle_return

    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    src = mr.allocate(4, stream=dev.default_stream)
    dst = mr.allocate(4, stream=dev.default_stream)
    src.fill(0xCD, stream=dev.default_stream)
    dev.default_stream.sync()
    src_dptr = int(src.handle)
    dst_dptr = int(dst.handle)

    g = GraphDefinition()
    g.memcpy(dst, src_dptr, 4, src_owner=src)
    src.close()
    dst.close()

    stream = dev.create_stream()
    g.instantiate().launch(stream)
    stream.sync()

    out = (ctypes.c_uint8 * 4)(0)
    handle_return(driver.cuMemcpyDtoH(out, dst_dptr, 4))
    assert list(out) == [0xCD] * 4


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memset_closed_buffer_rejected(init_cuda):
    """Memset rejects a Buffer with no active allocation."""
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    buf.close()

    g = GraphDefinition()
    with pytest.raises(ValueError, match="dst Buffer has no active allocation"):
        g.memset(buf, 0xAB, 4)


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memset_closed_buffer_dst_owner_rejected(init_cuda):
    """Memset rejects a closed Buffer passed as dst_owner."""
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)
    buf.close()

    g = GraphDefinition()
    with pytest.raises(ValueError, match="dst_owner Buffer has no active allocation"):
        g.memset(dptr, 0xAB, 4, dst_owner=buf)


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_closed_buffer_src_owner_rejected(init_cuda):
    """Memcpy rejects a closed Buffer passed as src_owner."""
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()
    dptr = int(buf.handle)
    buf.close()

    g = GraphDefinition()
    with pytest.raises(ValueError, match="src_owner Buffer has no active allocation"):
        g.memcpy(dptr, dptr, 4, src_owner=buf)


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffer_and_dst_owner_rejected(init_cuda):
    """dst_owner cannot be combined with a Buffer dst operand."""
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()

    g = GraphDefinition()
    with pytest.raises(ValueError, match="dst_owner cannot be used when dst is a Buffer"):
        g.memcpy(buf, buf, 4, dst_owner=object())


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_memcpy_buffer_and_src_owner_rejected(init_cuda):
    """src_owner cannot be combined with a Buffer src operand."""
    _skip_if_no_mempool()
    dev = Device()
    mr = DeviceMemoryResource(dev)
    buf = mr.allocate(4, stream=dev.default_stream)
    dev.default_stream.sync()

    g = GraphDefinition()
    with pytest.raises(ValueError, match="src_owner cannot be used when src is a Buffer"):
        g.memcpy(buf, buf, 4, src_owner=object())
