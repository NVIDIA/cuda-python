# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda_python_test_helpers.subprocess_runner import run_python_snippet
from helpers.child_processes import child_timeout_sec

import cuda.core
from cuda.core import Device


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

# Thread A retains the primary context and waits until thread B has retained
# it too and exited. Thread A then resolves a Context through the context
# registry (``{lookup}``), and the main thread uses that object (``{use}``) once
# both threads are gone. The child process starts clean, so nothing else holds
# a retain of the primary context: the pytest process does, through fixtures,
# which would hide the bug.
#
# Thread.join() can return before a thread's C++ thread_local destructors run,
# and they cannot be observed, so the child waits a fixed second after each
# join. With the fix, the test passes regardless of that timing. Without it,
# the test fails only if the destructors run within the wait, so a pass on
# unfixed code is not evidence that the bug is gone.
_PRIMARY_CONTEXT_ACROSS_THREADS = """
import threading
import time
import warnings

from cuda.bindings import driver
from cuda.core import CUDAWarning, Device, Stream

warnings.simplefilter("error", CUDAWarning)
out = []
errors = []
b_exited = threading.Event()


def thread_a():
    try:
        dev = Device({device_id})
        dev.set_current()
        b_exited.wait()
        out.append({lookup})
    except BaseException as exc:
        errors.append(exc)


def thread_b():
    try:
        Device({device_id}).set_current()
    except BaseException as exc:
        errors.append(exc)


a = threading.Thread(target=thread_a)
a.start()
b = threading.Thread(target=thread_b)
b.start()
b.join()
time.sleep(1.0)
b_exited.set()
a.join()
time.sleep(1.0)
if errors:
    raise errors[0]

err, _, active = driver.cuDevicePrimaryCtxGetState({device_id})
assert err == driver.CUresult.CUDA_SUCCESS, err
assert active, "primary context destroyed while a cuda.core object still refers to it"
obj = out[0]
{use}
"""


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.parametrize(
    ("lookup", "use"),
    [
        pytest.param(
            "Stream.from_handle(int(driver.CU_STREAM_LEGACY)).context",
            "Device({device_id}).set_current(obj)\nDevice({device_id}).create_stream().sync()",
            id="foreign-stream-context",
        ),
        pytest.param("dev.default_stream.record()", "obj.sync()", id="default-stream-event"),
    ],
)
def test_primary_context_survives_thread_exit(init_cuda, lookup, use):
    """A Context resolved through the registry keeps the primary context alive after other threads exit."""
    device_id = init_cuda.device_id
    code = _PRIMARY_CONTEXT_ACROSS_THREADS.format(
        device_id=device_id, lookup=lookup, use=use.format(device_id=device_id)
    )
    result = run_python_snippet(code, timeout=child_timeout_sec())
    assert "CONTEXT_IS_DESTROYED" not in result.stderr


# Several threads retain the primary context concurrently, so most of them lose
# the registration race in get_primary_context() and must release their extra
# retain. One Context must keep the context alive after every thread has exited,
# and dropping it must release the last retain: a leaked extra retain keeps the
# context active, and an extra release destroys it while the Context is held.
# The thread_local caches are destroyed after Thread.join() returns, so the child
# waits a fixed second before checking (see _PRIMARY_CONTEXT_ACROSS_THREADS).
_CONCURRENT_PRIMARY_RETAINS = """
import gc
import threading
import time
import warnings

from cuda.bindings import driver
from cuda.core import CUDAWarning, Device

warnings.simplefilter("error", CUDAWarning)
NUM_THREADS = 8
barrier = threading.Barrier(NUM_THREADS)
contexts = []
errors = []


def worker():
    try:
        dev = Device({device_id})
        barrier.wait()
        dev.set_current()
        contexts.append(dev.context)
    except BaseException as exc:
        errors.append(exc)


def primary_active():
    err, _, active = driver.cuDevicePrimaryCtxGetState({device_id})
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return bool(active)


threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
time.sleep(1.0)
if errors:
    raise errors[0]
assert len(contexts) == NUM_THREADS

held = contexts[0]
contexts.clear()
gc.collect()
assert primary_active(), "primary context destroyed while a Context is held"

del held
gc.collect()
deadline = time.monotonic() + 5.0
while primary_active() and time.monotonic() < deadline:
    time.sleep(0.05)
assert not primary_active(), "primary context still active after its last Context was dropped"
"""


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_concurrent_set_current_shares_one_primary_retain(init_cuda):
    """Concurrent Device.set_current() calls leave exactly one primary-context retain, held by the Context."""
    code = _CONCURRENT_PRIMARY_RETAINS.format(device_id=init_cuda.device_id)
    run_python_snippet(code, timeout=child_timeout_sec())
