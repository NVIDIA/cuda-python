# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.


import pytest
from conftest import can_load_generated_ptx

try:
    from cuda.bindings import runtime
except ImportError:
    from cuda import cudart as runtime

from cuda.core.experimental import Program, ProgramOptions
from cuda.core.experimental._utils import handle_return


@pytest.fixture(scope="module")
def cuda_version():
    # WAR this is a workaround for the fact that checking the runtime version using the cuGetDriverVersion
    # doesnt actually return the driver version buyt rather the latest cuda whcih is supported by the installed drive3r.

    version = handle_return(runtime.cudaRuntimeGetVersion())
    major_version = version // 1000
    minor_version = (version % 1000) // 10
    return major_version, minor_version


@pytest.fixture(scope="function")
def get_saxpy_kernel(init_cuda):
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

    # prepare program
    prog = Program(code, code_type="c++")
    mod = prog.compile(
        "cubin",
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )

    # run in single precision
    return mod.get_kernel("saxpy<float>")


@pytest.mark.xfail(not can_load_generated_ptx(), reason="PTX version too new")
def test_get_kernel(init_cuda):
    kernel = """extern "C" __global__ void ABC() { }"""
    object_code = Program(kernel, "c++", options=ProgramOptions(relocatable_device_code=True)).compile("ptx")
    assert object_code._handle is None
    kernel = object_code.get_kernel("ABC")
    assert object_code._handle is not None
    assert kernel._handle is not None


@pytest.mark.parametrize(
    "attr, expected_type",
    [
        ("max_threads_per_block", int),
        ("shared_size_bytes", int),
        ("const_size_bytes", int),
        ("local_size_bytes", int),
        ("num_regs", int),
        ("ptx_version", int),
        ("binary_version", int),
        ("cache_mode_ca", bool),
        ("cluster_size_must_be_set", bool),
        ("max_dynamic_shared_size_bytes", int),
        ("preferred_shared_memory_carveout", int),
        ("required_cluster_width", int),
        ("required_cluster_height", int),
        ("required_cluster_depth", int),
        ("non_portable_cluster_size_allowed", bool),
        ("cluster_scheduling_policy_preference", int),
    ],
)
def test_read_only_kernel_attributes(get_saxpy_kernel, attr, expected_type, cuda_version):
    if cuda_version[0] < 12:
        pytest.skip("CUDA version is less than 12, and doesn't support kernel attribute access")

    kernel = get_saxpy_kernel

    # Access the attribute to ensure it can be read
    value = getattr(kernel.attributes, attr)
    assert value is not None
    assert isinstance(value, expected_type), f"Expected {attr} to be of type {expected_type}, but got {type(value)}"

    # Attempt to set the attribute and ensure it raises an exception
    with pytest.raises(AttributeError):
        setattr(kernel.attributes, attr, value)
