# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
import numpy as np
try:
    from numba import cuda
    skip_tests = False
except ImportError:
    skip_tests = True

def launch_empty(kernel, stream):
    kernel[1,1, stream]()

def launch(kernel, stream, arg):
    kernel[1,1, stream](arg)

# Measure launch latency with no parmaeters
@pytest.mark.skipif(skip_tests, reason="Numba is not installed")
@pytest.mark.benchmark(group="numba", min_rounds=1000)
def test_launch_latency_empty_kernel(benchmark):
    stream = cuda.stream()

    @cuda.jit
    def empty_kernel():
        return

    benchmark(launch_empty, empty_kernel, stream)

    cuda.synchronize()

# Measure launch latency with a single parameter
@pytest.mark.skipif(skip_tests, reason="Numba is not installed")
@pytest.mark.benchmark(group="numba", min_rounds=1000)
def test_launch_latency_small_kernel(benchmark):
    stream = cuda.stream()

    arg = cuda.device_array(1, dtype=np.float32, stream=stream)

    @cuda.jit
    def small_kernel(array):
        array[0] = 0.0

    benchmark(launch, small_kernel, stream, arg)

    cuda.synchronize()
