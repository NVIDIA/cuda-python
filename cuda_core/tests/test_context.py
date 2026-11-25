# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import time

import pytest
from helpers.child_processes import child_timeout_sec, run_child
from helpers.contexts import assert_no_cuda_warning, primary_context_active
from helpers.misc import run_in_threads

import cuda.core
from cuda.bindings import driver
from cuda.core import Device, Stream
from cuda.core._utils.cuda_utils import handle_return


def test_context_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Context objects cannot be instantiated directly\."):
        cuda.core._context.Context()  # Ensure back door is locked.


# ============================================================================
# Context Equality Tests
# ============================================================================


def test_context_equality_same_context(init_cuda):
    """Contexts from same device should be equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"


def test_context_equality_reflexive(init_cuda):
    """Context should equal itself (reflexive property)."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    assert context == context, "Context should equal itself"


def test_context_type_safety(init_cuda):
    """Comparing Context with wrong type should return False."""
    device = Device()
    context = device.create_stream().context

    assert (context == "not a context") is False
    assert (context == 123) is False
    assert (context is None) is False


# ============================================================================
# Context Hash Tests
# ============================================================================


def test_context_hash_consistency(init_cuda):
    """Hash of same Context object should be consistent."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    hash1 = hash(context)
    hash2 = hash(context)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_context_hash_equality(init_cuda):
    """Contexts from same device should hash equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"
    assert hash(ctx1) == hash(ctx2), "Same context should hash equal"


def test_context_dict_key(init_cuda):
    """Contexts should be usable as dictionary keys."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    ctx_cache = {context: "context_data"}
    assert ctx_cache[context] == "context_data"


# ============================================================================
# Primary-context lifetime (#3533)
# ============================================================================

# Each test runs in a fresh child process, so nothing else holds a retain of the
# primary context: the pytest process does, through fixtures, which would hide
# a released or leaked retain.


# Several threads retain the primary context concurrently, after which every
# thread exits and every Context is dropped. cuda.core keeps exactly one retain
# of the primary context for the rest of the process: the context stays active,
# and releasing one retain from outside cuda.core leaves no retain behind. This
# covers the cause of #3533: on releasing code, the primary context is inactive
# here.
def _concurrent_primary_retains_child(device_id):
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    contexts = []

    def worker():
        dev = Device(device_id)
        barrier.wait()
        dev.set_current()
        contexts.append(dev.context)

    with assert_no_cuda_warning():
        run_in_threads([worker] * num_threads, barrier=barrier, timeout=child_timeout_sec())
        assert len(contexts) == num_threads
        assert len({int(ctx.handle) for ctx in contexts}) == 1
        contexts.clear()
        gc.collect()
        # Thread.join() can return before a thread's C++ thread_local
        # destructors run, and they cannot be observed, so wait a fixed second.
        # A release at thread exit would make the next check fail.
        time.sleep(1.0)
        assert primary_context_active(device_id), "cuda.core released its primary-context retain"

        # This process has no other retain of the primary context, so one
        # release must leave it inactive: cuda.core held exactly one retain.
        handle_return(driver.cuDevicePrimaryCtxRelease(device_id))
        assert not primary_context_active(device_id), "cuda.core holds more than one primary-context retain"


@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_concurrent_set_current_keeps_one_primary_retain(init_cuda):
    """Concurrent Device.set_current() calls take one primary-context retain, which cuda.core never releases."""
    run_child(_concurrent_primary_retains_child, init_cuda.device_id)


# Another owner (here cuda.bindings, standing in for another library) retains the
# primary context and makes it current, and cuda.core has not retained it. An
# Event recorded on the legacy default stream and that stream's Context, both
# resolved for the current context, must keep it alive once that owner releases
# its retain. Both resolve through the same lookup; the first one pins the
# primary context, and the second finds it registered.
def _foreign_primary_context_child(device_id):
    handle_return(driver.cuInit(0))
    with assert_no_cuda_warning():
        assert not primary_context_active(device_id)
        primary = handle_return(driver.cuDevicePrimaryCtxRetain(device_id))
        handle_return(driver.cuCtxSetCurrent(primary))
        legacy = Stream.from_handle(int(driver.CU_STREAM_LEGACY))
        event = legacy.record()
        ctx = legacy.context
        handle_return(driver.cuCtxSetCurrent(None))
        handle_return(driver.cuDevicePrimaryCtxRelease(device_id))

        assert primary_context_active(device_id), (
            "primary context destroyed while a cuda.core object still refers to it"
        )
        event.sync()
        dev = Device(device_id)
        dev.set_current(ctx)
        dev.create_stream().sync()


@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_lookup_retains_foreign_primary_context(init_cuda):
    """A Context resolved for a primary context that another owner made current keeps it alive."""
    run_child(_foreign_primary_context_child, init_cuda.device_id)
