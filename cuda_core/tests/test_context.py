# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from helpers.child_processes import run_child
from helpers.contexts import assert_no_cuda_warning

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
# Primary-context lifetime across threads (#3533)
# ============================================================================

# Thread A retains the primary context, then thread B retains it too and exits.
# Thread A then resolves a Context through the context registry (``lookup``),
# and the main thread uses that object (``use``) once both threads are gone.
# Each test runs in a fresh child process, so nothing else holds a retain of
# the primary context: the pytest process does, through fixtures, which would
# hide the bug.
#
# Thread.join() can return before a thread's C++ thread_local destructors run,
# and they cannot be observed, so the child waits a fixed second after each
# join. With the fix, the test passes regardless of that timing. Without it,
# the test fails only if the destructors run within the wait, so a pass on
# unfixed code is not evidence that the bug is gone.


def _primary_active(device_id):
    _, active = handle_return(driver.cuDevicePrimaryCtxGetState(device_id))
    return bool(active)


def _lookup_foreign_stream_context(dev):
    return Stream.from_handle(int(driver.CU_STREAM_LEGACY)).context


def _use_context(dev, ctx):
    dev.set_current(ctx)
    dev.create_stream().sync()


def _lookup_default_stream_event(dev):
    return dev.default_stream.record()


def _use_event(dev, event):
    event.sync()


def _primary_context_across_threads_child(device_id, lookup, use):
    def retain():
        Device(device_id).set_current()

    # A failed primary-context release is reported as a CUDAWarning, which a
    # warning filter cannot turn into a failure here (see
    # assert_no_cuda_warning), so the block asserts that none was issued.
    with assert_no_cuda_warning():
        # Each executor runs its work on one thread, whose exit ends the ``with``
        # block, and result() re-raises any error from that thread.
        with ThreadPoolExecutor(max_workers=1) as thread_a:
            thread_a.submit(retain).result()
            # B must retain the primary context after A: if B registered first
            # and exited, A would register its own retain and hide the bug.
            with ThreadPoolExecutor(max_workers=1) as thread_b:
                thread_b.submit(retain).result()
            time.sleep(1.0)
            obj = thread_a.submit(lambda: lookup(Device(device_id))).result()
        time.sleep(1.0)

        assert _primary_active(device_id), "primary context destroyed while a cuda.core object still refers to it"
        use(Device(device_id), obj)


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.parametrize(
    ("lookup", "use"),
    [
        pytest.param(_lookup_foreign_stream_context, _use_context, id="foreign-stream-context"),
        pytest.param(_lookup_default_stream_event, _use_event, id="default-stream-event"),
    ],
)
def test_primary_context_survives_thread_exit(init_cuda, lookup, use):
    """A Context resolved through the registry keeps the primary context alive after other threads exit."""
    # The child asserts on the primary context's state and uses the object, so
    # it exits non-zero (failing the test) if the context was destroyed.
    run_child(_primary_context_across_threads_child, init_cuda.device_id, lookup, use)


# Several threads retain the primary context concurrently, so most of them lose
# the registration race in get_primary_context() and must release their extra
# retain. One Context must keep the context alive after every thread has exited,
# and dropping it must release the last retain: a leaked extra retain keeps the
# context active, and an extra release destroys it while the Context is held.
# The thread_local caches are destroyed after Thread.join() returns. They hold
# the same box as the Context, so they do not affect the first check, and the
# second check polls until they are gone.
def _concurrent_primary_retains_child(device_id):
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def worker(_):
        dev = Device(device_id)
        barrier.wait()
        dev.set_current()
        return dev.context

    # An extra release of the primary context fails and is reported as a
    # CUDAWarning, so the block asserts that none was issued.
    with assert_no_cuda_warning():
        # The workers block on the barrier, so each runs on its own thread.
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            contexts = list(pool.map(worker, range(num_threads)))

        held = contexts[0]
        contexts.clear()
        gc.collect()
        assert _primary_active(device_id), "primary context destroyed while a Context is held"

        del held
        gc.collect()
        deadline = time.monotonic() + 5.0
        while _primary_active(device_id) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _primary_active(device_id), "primary context still active after its last Context was dropped"


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_concurrent_set_current_shares_one_primary_retain(init_cuda):
    """Concurrent Device.set_current() calls leave exactly one primary-context retain, held by the Context."""
    run_child(_concurrent_primary_retains_child, init_cuda.device_id)
