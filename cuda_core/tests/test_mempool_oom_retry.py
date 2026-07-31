# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the deferred-release retry in ``get_device_mempool``.

Issue #2381: the driver reserves virtual address space per memory pool -- on
Windows MCDM roughly 2x device memory against a 40-bit (1 TiB) cap -- so pools
that are awaiting teardown can starve the *default* pool's reservation. cuda.core
then reports ``CUDA_ERROR_OUT_OF_MEMORY`` on a device with ample free memory.

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

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses

# Fill coarsely first, then top up finely. A single pool size cannot do both:
# large pools exhaust a big address space in a feasible number of steps but
# leave a gap the default pool still fits into, while small pools fill precisely
# but would need far too many to cover a 48-bit space.
POOL_BYTES_COARSE = 64 * 1024**3
POOL_BYTES_FINE = 1024**3
MAX_POOLS_PER_PASS = 4096
BLOCK_MS = 2000

NO_EXHAUSTION = "no-exhaustion"
NOT_DEFERRED = "not-deferred"
RECOVERED = "recovered"
NOT_RECOVERED = "not-recovered"


def _worker_deferred_release(result_queue):
    """Drive a default-pool lookup into deferred-release failure, then retry it.

    Reports one of the module-level outcome strings so the parent can tell a
    genuine regression apart from a machine whose address space is too large to
    exhaust.
    """
    import gc
    import time

    from helpers.nanosleep_kernel import NanosleepKernel

    from cuda.bindings import driver
    from cuda.core import Device, DeviceMemoryResource, DeviceMemoryResourceOptions
    from cuda.core._utils.cuda_utils import CUDAError

    def default_pool_available():
        err, _pool = driver.cuDeviceGetMemPool(dev)
        return err == driver.CUresult.CUDA_SUCCESS

    device = Device(0)
    device.set_current()
    err, dev = driver.cuDeviceGet(0)
    assert err == driver.CUresult.CUDA_SUCCESS, err

    # Build everything before the address space is gone; afterwards even kernel
    # compilation could fail for unrelated reasons.
    sleeper = NanosleepKernel(device, sleep_duration_ms=BLOCK_MS)
    work_stream = device.create_stream()
    dealloc_stream = device.create_stream()

    pools = []
    buffers = []

    def reserve_until_oom(pool_bytes):
        options = DeviceMemoryResourceOptions(max_size=pool_bytes)
        for _ in range(MAX_POOLS_PER_PASS):
            try:
                mr = DeviceMemoryResource(device, options=options)
            except (CUDAError, RuntimeError):
                return
            # An empty pool is torn down immediately, which is the one case that
            # cannot defer anything. Hold an allocation so teardown must wait.
            buffers.append(mr.allocate(1024, stream=dealloc_stream))
            pools.append(mr)

    reserve_until_oom(POOL_BYTES_COARSE)
    reserve_until_oom(POOL_BYTES_FINE)

    if default_pool_available():
        result_queue.put((NO_EXHAUSTION, len(pools)))
        return

    # Stall the deallocation stream behind a slow kernel on another stream, so
    # the frees below cannot retire until the context is drained.
    sleeper.launch(work_stream)
    dealloc_stream.wait(work_stream.record())

    buffers.clear()
    pools.clear()
    gc.collect()

    if default_pool_available():
        # Release outran us; there was no deferred window to recover from.
        result_queue.put((NOT_DEFERRED, len(pools)))
        return

    # The raw lookup just failed, so anything the retried lookup achieves below
    # is attributable to the retry itself -- no cross-build comparison needed.
    # Time it too: recovering requires draining the blocking kernel, so a call
    # that returns instantly would mean something other than the drain fixed it.
    started = time.perf_counter()
    try:
        DeviceMemoryResource(device)
    except (CUDAError, RuntimeError) as exc:
        result_queue.put((NOT_RECOVERED, repr(exc)))
        return
    result_queue.put((RECOVERED, time.perf_counter() - started))


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.thread_unsafe(reason="Reserves the device's entire virtual address space.")
def test_default_mempool_lookup_recovers_from_deferred_release():
    ctx = _mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_deferred_release, args=(result_queue,))
    proc.start()
    try:
        outcome, detail = result_queue.get(timeout=child_timeout_sec())
    finally:
        proc.join(timeout=child_timeout_sec())
        survivors = kill_subprocesses(proc)
    assert not survivors, "child process did not exit"

    if outcome == NO_EXHAUSTION:
        pytest.skip(f"could not exhaust the address space; reserved {detail} pools")
    if outcome == NOT_DEFERRED:
        pytest.skip("pool release completed too quickly to leave a deferred window")
    assert outcome == RECOVERED, f"default pool lookup did not recover: {detail}"
    # Recovery had to wait out the blocking kernel. Returning much faster would
    # mean the lookup succeeded for some reason other than draining the context,
    # leaving this test passing without exercising the retry.
    assert detail >= (BLOCK_MS / 1000) / 4, f"recovered suspiciously fast ({detail:.3f}s)"
