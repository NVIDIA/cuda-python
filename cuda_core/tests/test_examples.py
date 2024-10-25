# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental import Device
from cuda.core.experimental import LaunchConfig, launch
from cuda.core.experimental import Program
import sys
import pytest

@pytest.fixture(scope='module')
def init_cuda():
    Device().set_current()

#saxpy example
def test_saxpy_example():
    import cupy as cp
    # compute out = a * x + y
    code = """
    template<typename T>
    __global__ void saxpy(const T a,
                        const T* x,
                        const T* y,
                        T* out,
                        size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
            out[tid] = a * x[tid] + y[tid];
        }
    }
    """


    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    # prepare program
    prog = Program(code, code_type="c++")
    mod = prog.compile(
        "cubin",
        options=("-std=c++11", "-arch=sm_" + "".join(f"{i}" for i in dev.compute_capability),),
        logs=sys.stdout,
        name_expressions=("saxpy<float>", "saxpy<double>"))

    # run in single precision
    ker = mod.get_kernel("saxpy<float>")
    dtype = cp.float32

    # prepare input/output
    size = cp.uint64(64)
    a = dtype(10)
    x = cp.random.random(size, dtype=dtype)
    y = cp.random.random(size, dtype=dtype)
    out = cp.empty_like(x)
    dev.sync()  # cupy runs on a different stream from s, so sync before accessing

    # prepare launch
    block = 32
    grid = int((size + block - 1) // block)
    config = LaunchConfig(grid=grid, block=block, stream=s)
    ker_args = (a, x.data.ptr, y.data.ptr, out.data.ptr, size)

    # launch kernel on stream s
    launch(ker, config, *ker_args)
    s.sync()

    # check result
    assert cp.allclose(out, a*x+y)

    # let's repeat again, this time allocates our own out buffer instead of cupy's
    # run in double precision
    ker = mod.get_kernel("saxpy<double>")
    dtype = cp.float64

    # prepare input
    size = cp.uint64(128)
    a = dtype(42)
    x = cp.random.random(size, dtype=dtype)
    y = cp.random.random(size, dtype=dtype)
    dev.sync()

    # prepare output
    buf = dev.allocate(size * 8,  # = dtype.itemsize
                    stream=s)

    # prepare launch
    block = 64
    grid = int((size + block - 1) // block)
    config = LaunchConfig(grid=grid, block=block, stream=s)
    ker_args = (a, x.data.ptr, y.data.ptr, buf, size)

    # launch kernel on stream s
    launch(ker, config, *ker_args)
    s.sync()

    # check result
    # we wrap output buffer as a cupy array for simplicity
    out = cp.ndarray(size, dtype=dtype,
                    memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(buf.handle), buf.size, buf), 0))
    assert cp.allclose(out, a*x+y)

    # clean up resources that we allocate
    # cupy cleans up automatically the rest
    buf.close(s)
    s.close()

def test_vector_add_example():
    import cupy as cp
    # compute c = a + b
    code = """
    template<typename T>
    __global__ void vector_add(const T* A,
                            const T* B,
                            T* C,
                            size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
            C[tid] = A[tid] + B[tid];
        }
    }
    """


    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    # prepare program
    prog = Program(code, code_type="c++")
    mod = prog.compile(
        "cubin",
        options=("-std=c++17", "-arch=sm_" + "".join(f"{i}" for i in dev.compute_capability),),
        name_expressions=("vector_add<float>",))

    # run in single precision
    ker = mod.get_kernel("vector_add<float>")
    dtype = cp.float32

    # prepare input/output
    size = 50000
    a = cp.random.random(size, dtype=dtype)
    b = cp.random.random(size, dtype=dtype)
    c = cp.empty_like(a)

    # cupy runs on a different stream from s, so sync before accessing
    dev.sync()

    # prepare launch
    block = 256
    grid = (size + block - 1) // block
    config = LaunchConfig(grid=grid, block=block, stream=s)

    # launch kernel on stream s
    launch(ker, config, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))
    s.sync()

    # check result
    assert cp.allclose(c, a+b)
