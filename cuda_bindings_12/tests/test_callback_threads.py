# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from cuda.bindings import driver

NUM_THREADS = 16
NUM_ITERATIONS = 500


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_async_notification_register_unregister_from_many_threads():
    (err,) = driver.cuInit(0)
    assert err == driver.CUresult.CUDA_SUCCESS
    err, device = driver.cuDeviceGet(0)
    assert err == driver.CUresult.CUDA_SUCCESS

    barrier = threading.Barrier(NUM_THREADS)
    failures = []

    def worker():
        try:
            barrier.wait()
            for _ in range(NUM_ITERATIONS):
                err, handle = driver.cuDeviceRegisterAsyncNotification(device, 0, 0)
                assert err == driver.CUresult.CUDA_SUCCESS
                (err,) = driver.cuDeviceUnregisterAsyncNotification(device, handle)
                assert err == driver.CUresult.CUDA_SUCCESS
        except BaseException as e:
            failures.append(e)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, failures
