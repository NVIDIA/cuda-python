# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the deferred-release retry in ``get_device_mempool``.

Issue #2381: each memory pool reserves virtual address space scaling with device
memory, and the per-process budget is bounded (notably ~1 TB on Windows MCDM), so
pools awaiting teardown can starve the *default* pool's reservation. cuda.core then
reports ``CUDA_ERROR_OUT_OF_MEMORY`` on a device with ample free memory.

A pool's address space comes back only once the pool is destroyed, and
``cuMemPoolDestroy`` waits on the stream-ordered frees of the pool's outstanding
allocations. So when those frees are queued behind unfinished work, the address
space is recoverable but not yet recovered -- which is the window
``get_device_mempool`` closes by draining the context once and retrying.

This runs in a child process on purpose. The driver creates the default pool
lazily, and once it exists ``cuDeviceGetMemPool`` is a handle lookup that cannot
fail, so a process that has already touched it could never reproduce the failure.
"""

from __future__ import annotations

import multiprocessing as _mp
import queue
import traceback

import pytest
from helpers import IS_WSL
from helpers.child_processes import child_timeout_sec, kill_subprocesses

# Fill coarsely first, then top up finely. A single pool size cannot do both:
# large pools exhaust a big address space in a feasible number of steps but
# leave a gap the default pool still fits into, while small pools fill precisely
# but would need far too many to cover a 48-bit space.
POOL_BYTES_COARSE = 64 * 1024**3
POOL_BYTES_FINE = 1024**3
BLOCK_MS = 2000

# The cost that matters is the number of pools, not their size: every one has to
# be created, then destroyed again with its stream-ordered free queued behind the
# blocking kernel. This cap bounds both halves. A machine needing more than this
# to exhaust its address space skips instead, which is how large-address-space
# environments (WSL, for one) opt out without naming a platform.
MAX_POOLS_TOTAL = 512
# Ceiling for the address-space probe, reached by doubling from 1 GiB.
VA_PROBE_LIMIT = 256 * 1024**4
VA_ALIGNMENT = 2 * 1024**2

# Outcomes meaning this machine cannot host the probe, so there is nothing to
# assert. The parent skips on all of these.
UNSUPPORTED = "unsupported"
ADDRESS_SPACE_TOO_LARGE = "address-space-too-large"
NO_EXHAUSTION = "no-exhaustion"
BUDGET_EXPIRED = "budget-expired"
NOT_DEFERRED = "not-deferred"
SKIP_OUTCOMES = frozenset({UNSUPPORTED, ADDRESS_SPACE_TOO_LARGE, NO_EXHAUSTION, BUDGET_EXPIRED, NOT_DEFERRED})

# Outcomes meaning the probe actually ran. The parent asserts on these.
RECOVERED = "recovered"
NOT_RECOVERED = "not-recovered"
CRASHED = "crashed"

# Parent-side only: the child never reported anything, so its own budgets failed
# to keep it inside the timeout. Distinct from BUDGET_EXPIRED, where the child
# recognised it was out of time and bowed out cleanly -- that skips, this fails.
NO_RESULT = "no-result"


def _largest_va_reservation(driver):
    """Largest single VA range that can be reserved, found by doubling.

    A reservation costs address space but no physical memory, so this is cheap.
    It is a lower bound on the total space, which is all that is needed to decide
    whether exhausting it is affordable.
    """
    largest = 0
    size = 1024**3
    while size <= VA_PROBE_LIMIT:
        err, ptr = driver.cuMemAddressReserve(size, VA_ALIGNMENT, 0, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            break
        driver.cuMemAddressFree(ptr, size)
        largest = size
        size *= 2
    return largest


def _run_deferred_release():
    """Drive a default-pool lookup into deferred-release failure, then retry it.

    Returns an ``(outcome, detail)`` pair using the module-level outcome strings,
    so the parent can tell a genuine regression apart from a machine that cannot
    run this at all.
    """
    import gc
    import time

    from helpers.nanosleep_kernel import NanosleepKernel

    from cuda.bindings import driver
    from cuda.core import Device, DeviceMemoryResource, DeviceMemoryResourceOptions
    from cuda.core._utils.cuda_utils import CUDAError

    device = Device(0)
    device.set_current()
    if not device.properties.memory_pools_supported:
        return UNSUPPORTED, "Device does not support mempool operations"

    err, dev = driver.cuDeviceGet(0)
    assert err == driver.CUresult.CUDA_SUCCESS, err

    # Decide up front whether exhausting this address space is affordable, rather
    # than discovering it by running out of budget partway through.
    va_bytes = _largest_va_reservation(driver)
    if va_bytes // POOL_BYTES_COARSE > MAX_POOLS_TOTAL:
        return ADDRESS_SPACE_TOO_LARGE, (
            f"address space of at least {va_bytes / 1024**4:.1f} TiB needs over {MAX_POOLS_TOTAL} pools to exhaust"
        )

    def default_pool_available():
        err, _pool = driver.cuDeviceGetMemPool(dev)
        return err == driver.CUresult.CUDA_SUCCESS

    # Build everything before the address space is gone; afterwards even kernel
    # compilation could fail for unrelated reasons.
    sleeper = NanosleepKernel(device, sleep_duration_ms=BLOCK_MS)
    work_stream = device.create_stream()
    dealloc_stream = device.create_stream()

    pools = []
    buffers = []
    # Stay well inside the parent's timeout, in two stages. Reserving stops at
    # the first deadline; the deferred-release phase that follows is expensive in
    # its own right (destroying every pool, waiting out the blocking kernel, then
    # retrying), so if that cannot also fit we skip instead of letting the parent
    # time out. Capping only the reservation loop is not enough -- a large address
    # space produces thousands of pools whose teardown alone overruns the budget.
    budget = child_timeout_sec()
    reserve_deadline = time.monotonic() + budget / 3
    recovery_deadline = time.monotonic() + budget / 2

    def reserve_until_oom(pool_bytes):
        options = DeviceMemoryResourceOptions(max_size=pool_bytes)
        # The cap is shared across both passes: it bounds total teardown cost, so
        # splitting it per pass would let the two together exceed the budget.
        while len(pools) < MAX_POOLS_TOTAL:
            if time.monotonic() > reserve_deadline:
                return
            try:
                mr = DeviceMemoryResource(device, options=options)
                # An empty pool is torn down immediately, which is the one case
                # that cannot defer anything, so hold an allocation to force
                # teardown to wait. Exhaustion frequently surfaces here rather
                # than at construction, so both must share this guard -- an
                # unguarded allocate kills the whole child process.
                buffer = mr.allocate(1024, stream=dealloc_stream)
            except (CUDAError, RuntimeError):
                return
            pools.append(mr)
            buffers.append(buffer)

    reserve_until_oom(POOL_BYTES_COARSE)
    reserve_until_oom(POOL_BYTES_FINE)

    if default_pool_available():
        return NO_EXHAUSTION, f"{len(pools)} pools reserved, default pool still available"
    if time.monotonic() > recovery_deadline:
        return BUDGET_EXPIRED, f"exhausted with {len(pools)} pools, too slow to also test recovery"

    # Stall the deallocation stream behind a slow kernel on another stream, so
    # the frees below cannot retire until the context is drained.
    sleeper.launch(work_stream)
    dealloc_stream.wait(work_stream.record())

    buffers.clear()
    pools.clear()
    gc.collect()

    if default_pool_available():
        # Release outran us; there was no deferred window to recover from.
        return NOT_DEFERRED, "pool release completed before the retry could be tested"

    # The raw lookup just failed, so anything the retried lookup achieves below
    # is attributable to the retry itself -- no cross-build comparison needed.
    # Time it too: recovering requires draining the blocking kernel, so a call
    # that returns instantly would mean something other than the drain fixed it.
    started = time.perf_counter()
    try:
        DeviceMemoryResource(device)
    except (CUDAError, RuntimeError) as exc:
        return NOT_RECOVERED, repr(exc)
    return RECOVERED, time.perf_counter() - started


def _worker_deferred_release(result_queue):
    """Always report an outcome, so a crash surfaces as a diagnosis.

    Without this the parent would block until its timeout and raise a bare
    ``queue.Empty``, hiding whatever actually went wrong in the child.
    """
    try:
        result_queue.put(_run_deferred_release())
    except BaseException:
        result_queue.put((CRASHED, traceback.format_exc()))


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.thread_unsafe(reason="Reserves the device's entire virtual address space.")
# Must be pre-spawn: the budgets below are checked between operations, and under
# WSL control never returns to Python for them to run.
@pytest.mark.skipif(IS_WSL, reason="child blocks in the CUDA driver; in-child budgets cannot preempt it")
def test_default_mempool_lookup_recovers_from_deferred_release():
    ctx = _mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_deferred_release, args=(result_queue,))
    proc.start()
    try:
        outcome, detail = result_queue.get(timeout=child_timeout_sec())
    except queue.Empty:
        outcome, detail = NO_RESULT, f"child produced no result within {child_timeout_sec()}s"
    finally:
        proc.join(timeout=child_timeout_sec())
        survivors = kill_subprocesses(proc)

    if outcome in SKIP_OUTCOMES:
        pytest.skip(f"{outcome}: {detail}")

    assert outcome == RECOVERED, f"{outcome}: {detail}"
    assert not survivors, "child process did not exit"

    # Recovery had to wait out the blocking kernel. Returning much faster would
    # mean the lookup succeeded for some reason other than draining the context,
    # leaving this test passing without exercising the retry.
    assert detail >= (BLOCK_MS / 1000) / 4, f"recovered suspiciously fast ({detail:.3f}s)"
