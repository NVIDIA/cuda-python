# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the error handling policy (docs/source/error_handling.rst).

Ordinary calls raise and leave the caller's context untouched; failures that
cannot be raised are reported as CUDAWarning; a failure to restore the caller's
context is raised (or reported) with an explanation rather than swallowed or
turned into a process abort; and a secondary failure that occurs while an
exception is being raised is attached to that exception as a note. Restoration
failures are injected with the handle layer's test hook, which leaves the target
context current exactly as a real ``cuCtxSetCurrent`` failure would, so every
test here restores the context stack itself.
"""

import ctypes
import sys
from contextlib import contextmanager

import pytest
from helpers.constants import POOL_SIZE
from helpers.contexts import assert_no_cuda_warning, current_context_handle

import cuda.core
from cuda.core import (
    CUDAWarning,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    LegacyPinnedMemoryResource,
)
from cuda.core._memory._synchronous_memory_resource import _SynchronousMemoryResource
from cuda.core._resource_handles import (
    _note_or_report_cuda_error_for_testing,
    _set_context_restore_fault_for_testing,
)
from cuda.core._stream import default_stream
from cuda.core._utils.cuda_utils import CUDAError, driver, handle_return
from cuda.core._utils.version import binding_version, driver_version
from cuda.core.graph import GraphDefinition

INVALID_CONTEXT = int(driver.CUresult.CUDA_ERROR_INVALID_CONTEXT)
INVALID_VALUE = int(driver.CUresult.CUDA_ERROR_INVALID_VALUE)
DEINITIALIZED = int(driver.CUresult.CUDA_ERROR_DEINITIALIZED)

# PEP 678 exception notes; on 3.10 the same information lands in the message.
HAS_NOTES = sys.version_info >= (3, 11)

thread_unsafe_context_fault = pytest.mark.thread_unsafe(
    reason="injects a thread-local restoration fault and mutates the CUDA context stack"
)
thread_unsafe_warning_capture = pytest.mark.thread_unsafe(reason="warning capture is process-global")


def error_text(exc):
    """The message plus any notes, wherever the detail lives on this interpreter."""
    return "\n".join([str(exc), *getattr(exc, "__notes__", [])])


@contextmanager
def no_context_with_restore_fault(status=INVALID_CONTEXT):
    """Pop the current context and make the next restoration fail with ``status``.

    On exit, drop whatever the failed restoration left current, clear an unused
    fault, and push the popped context back.
    """
    previous = handle_return(driver.cuCtxPopCurrent())
    assert current_context_handle() == 0
    _set_context_restore_fault_for_testing(status)
    try:
        yield
    finally:
        _set_context_restore_fault_for_testing(0)
        handle_return(driver.cuCtxSetCurrent(driver.CUcontext(0)))
        handle_return(driver.cuCtxPushCurrent(previous))


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cudawarning_is_public_and_shown_by_default():
    assert "CUDAWarning" in cuda.core.__all__
    assert issubclass(CUDAWarning, RuntimeWarning)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_create_stream_raises_when_context_cannot_be_restored(init_cuda):
    """Creation is undone and the error explains the context state; no abort, no warning."""
    dev = init_cuda
    with no_context_with_restore_fault():
        with assert_no_cuda_warning(), pytest.raises(CUDAError) as excinfo:
            dev.create_stream()
        text = error_text(excinfo.value)
        assert "could not be restored" in text
        assert "CUDA_ERROR_INVALID_CONTEXT" in text
        assert "Device.set_current()" in text
        if HAS_NOTES:
            # The explanation is a note, separable from the driver error message.
            assert "could not be restored" not in str(excinfo.value)
            assert any("could not be restored" in note for note in excinfo.value.__notes__)
        # As documented, a failed restoration leaves the device's context current.
        assert current_context_handle() == int(dev.context.handle)
    assert current_context_handle() == int(dev.context.handle)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_sync_raises_when_context_cannot_be_restored(init_cuda):
    """A context-scoped call without a created resource raises the same explanation."""
    dev = init_cuda
    with no_context_with_restore_fault():
        with pytest.raises(CUDAError) as excinfo:
            dev.sync()
        assert "could not be restored" in error_text(excinfo.value)
        assert current_context_handle() == int(dev.context.handle)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_failed_call_raises_its_own_error_with_the_restore_failure_attached(init_cuda):
    """When the call and the restoration both fail, the call's error is raised and the
    restoration failure is attached to it; nothing is reported out of band."""
    dev = init_cuda
    mr = _SynchronousMemoryResource(dev.device_id)
    with no_context_with_restore_fault():
        with assert_no_cuda_warning(), pytest.raises(CUDAError) as excinfo:
            mr.allocate(1 << 62)
        message = str(excinfo.value)
        text = error_text(excinfo.value)
        # The allocation failure is the primary error, not the restoration failure.
        assert not message.startswith("CUDA_ERROR_INVALID_CONTEXT")
        assert "could not be restored after this failure" in text
        assert "cuCtxSetCurrent: CUDA_ERROR_INVALID_CONTEXT" in text
        assert "Device.set_current()" in text
        assert current_context_handle() == int(dev.context.handle)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_unused_restore_fault_does_not_fire_without_a_context_switch(init_cuda):
    """The hook only affects restorations; a call in the current context never restores."""
    dev = init_cuda
    _set_context_restore_fault_for_testing(INVALID_CONTEXT)
    try:
        stream = dev.create_stream()
        stream.close()
    finally:
        _set_context_restore_fault_for_testing(0)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cleanup_reports_restore_failure_as_warning(mempool_device):
    """A restoration failure inside a destructor cannot raise, so it is reported."""
    dev = mempool_device
    mr = DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))
    # A default-stream deallocation records the allocating context, so freeing
    # with no context current switches to it and must switch back.
    buf = mr.allocate(256, stream=default_stream())
    with no_context_with_restore_fault():
        with pytest.warns(CUDAWarning, match="restoring the caller's context") as records:
            buf.close()
        assert any("CUDA_ERROR_INVALID_CONTEXT" in str(record.message) for record in records)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_escalated_cudawarning_from_cleanup_is_not_a_crash(mempool_device):
    """With CUDAWarning promoted to an error, a destructor-path report cannot be raised.

    It is delivered through sys.unraisablehook instead; the process continues and
    the resource release still runs. pytest surfaces the hook as a warning, so the
    hook is replaced here to keep the test's own outcome deterministic.
    """
    import sys
    import warnings

    dev = mempool_device
    mr = DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))
    buf = mr.allocate(256, stream=default_stream())
    unraisable = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with no_context_with_restore_fault(), warnings.catch_warnings():
            warnings.simplefilter("error", CUDAWarning)
            buf.close()
    finally:
        sys.unraisablehook = previous_hook
    assert len(unraisable) == 1
    assert issubclass(unraisable[0].exc_type, CUDAWarning)
    assert "restoring the caller's context" in str(unraisable[0].exc_value)


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_set_current_with_context_works_without_a_current_context(init_cuda):
    """set_current(ctx) binds in one driver call; no previous context means None."""
    dev = init_cuda
    ctx = dev.context
    previous = handle_return(driver.cuCtxPopCurrent())
    try:
        assert current_context_handle() == 0
        assert dev.set_current(ctx) is None
        assert current_context_handle() == int(ctx.handle)
    finally:
        # Leave exactly one context on the stack, as the fixture expects.
        handle_return(driver.cuCtxSetCurrent(driver.CUcontext(0)))
        handle_return(driver.cuCtxPushCurrent(previous))


@thread_unsafe_context_fault
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_memset_update_keeps_new_owners_alive_when_context_cannot_be_restored(device_x2):
    """The node's new parameters stay valid: the attachment is published before the
    restoration failure is raised, so the updated graph instantiates and runs."""
    if driver_version() < (13, 2, 0) or binding_version() < (13, 2, 0):
        pytest.skip("node contexts are only recorded by cuGraphNodeGetParams on CUDA 13.2+")
    node_dev, other_dev = device_x2
    node_dev.set_current()
    memory_resource = LegacyPinnedMemoryResource()
    dst = memory_resource.allocate(4)
    replacement = memory_resource.allocate(4)
    graph_def = GraphDefinition()
    node = graph_def.memset(dst, 0x11, 4)

    # Updating from another device's context switches to the node's context and
    # must switch back; make that restoration fail.
    other_dev.set_current()
    _set_context_restore_fault_for_testing(INVALID_CONTEXT)
    try:
        with pytest.raises(CUDAError) as excinfo:
            node.update(dst=replacement, value=0x22)
        assert "could not be restored" in error_text(excinfo.value)
    finally:
        _set_context_restore_fault_for_testing(0)
        node_dev.set_current()

    def as_bytes(buffer):
        return (ctypes.c_uint8 * 4).from_address(int(buffer.handle))

    as_bytes(dst)[:] = [0] * 4
    as_bytes(replacement)[:] = [0] * 4
    graph = graph_def.instantiate()
    stream = node_dev.create_stream()
    graph.launch(stream)
    stream.sync()
    # The driver applied the update, and the replacement buffer it references
    # is still retained by the graph rather than dangling.
    assert list(as_bytes(replacement)) == [0x22] * 4
    assert list(as_bytes(dst)) == [0] * 4
    graph.close()
    stream.close()


@thread_unsafe_warning_capture
@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rollback_failure_is_attached_to_the_propagating_exception():
    """A failed rollback inside an except block becomes a note on the exception being
    handled (Python 3.11+); on 3.10, or with no exception being handled, it is reported
    as a CUDAWarning. CUDA_ERROR_DEINITIALIZED is neither attached nor reported."""
    with pytest.raises(RuntimeError) as excinfo:
        try:
            raise RuntimeError("primary failure")
        except RuntimeError:
            if HAS_NOTES:
                with assert_no_cuda_warning():
                    _note_or_report_cuda_error_for_testing(INVALID_VALUE)
            else:
                with pytest.warns(CUDAWarning, match="cuTestOperation failed while testing"):
                    _note_or_report_cuda_error_for_testing(INVALID_VALUE)
            with assert_no_cuda_warning():
                _note_or_report_cuda_error_for_testing(DEINITIALIZED)
            raise
    exc = excinfo.value
    assert str(exc) == "primary failure"
    if HAS_NOTES:
        assert len(exc.__notes__) == 1
        assert "cuTestOperation failed while testing: CUDA_ERROR_INVALID_VALUE" in exc.__notes__[0]
    else:
        assert not hasattr(exc, "__notes__")
    # With no exception being handled there is nothing to attach to.
    with pytest.warns(CUDAWarning, match="cuTestOperation failed while testing"):
        _note_or_report_cuda_error_for_testing(INVALID_VALUE)
