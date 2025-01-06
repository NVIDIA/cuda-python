# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.


import pytest
from conftest import can_load_generated_ptx

from cuda.core.experimental import Device, Program


@pytest.mark.xfail(not can_load_generated_ptx(), reason="PTX version too new")
def test_get_kernel():
    kernel = """extern "C" __global__ void ABC() { }"""
    object_code = Program(kernel, "c++").compile("ptx", options=("-rdc=true",))
    assert object_code._handle is None
    kernel = object_code.get_kernel("ABC")
    assert object_code._handle is not None
    assert kernel._handle is not None


def test_kernel_attributes():
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
    dev.create_stream()

    # prepare program
    prog = Program(code, code_type="c++")
    mod = prog.compile(
        "cubin",
        options=(
            "-std=c++11",
            "-arch=sm_" + "".join(f"{i}" for i in dev.compute_capability),
        ),
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )

    # run in single precision
    kernel = mod.get_kernel("saxpy<float>")
    print(kernel.max_threads_per_block)
