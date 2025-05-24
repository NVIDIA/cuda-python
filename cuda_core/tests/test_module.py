# Copyright 2024 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import warnings

import pytest
from conftest import skipif_testing_with_compute_sanitizer

import cuda.core.experimental
from cuda.core.experimental import ObjectCode, Program, ProgramOptions, system
from cuda.core.experimental._utils.cuda_utils import CUDAError, driver, get_binding_version, handle_return

SAXPY_KERNEL = r"""
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


@pytest.fixture(scope="module")
def cuda12_prerequisite_check():
    # binding availability depends on cuda-python version
    # and version of underlying CUDA toolkit
    _py_major_ver, _ = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    return _py_major_ver >= 12 and _driver_ver >= 12000


def test_kernel_attributes_init_disabled():
    with pytest.raises(RuntimeError, match=r"^KernelAttributes cannot be instantiated directly\."):
        cuda.core.experimental._module.KernelAttributes()  # Ensure back door is locked.


def test_kernel_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Kernel objects cannot be instantiated directly\."):
        cuda.core.experimental._module.Kernel()  # Ensure back door is locked.


def test_object_code_init_disabled():
    with pytest.raises(RuntimeError, match=r"^ObjectCode objects cannot be instantiated directly\."):
        ObjectCode()  # Reject at front door.


@pytest.fixture(scope="function")
def get_saxpy_kernel(init_cuda):
    # prepare program
    prog = Program(SAXPY_KERNEL, code_type="c++")
    mod = prog.compile(
        "cubin",
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )

    # run in single precision
    return mod.get_kernel("saxpy<float>"), mod


@pytest.fixture(scope="function")
def get_saxpy_kernel_ptx(init_cuda):
    prog = Program(SAXPY_KERNEL, code_type="c++")
    mod = prog.compile(
        "ptx",
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )
    ptx = mod._module
    return ptx, mod


@pytest.fixture(scope="function")
def get_saxpy_object_code(init_cuda):
    prog = Program(SAXPY_KERNEL, code_type="c++")
    mod = prog.compile(
        "cubin",
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )
    return mod


def test_get_kernel(init_cuda):
    kernel = """extern "C" __global__ void ABC() { }"""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        object_code = Program(kernel, "c++", options=ProgramOptions(relocatable_device_code=True)).compile("ptx")
        if any("The CUDA driver version is older than the backend version" in str(warning.message) for warning in w):
            pytest.skip("PTX version too new for current driver")

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
def test_read_only_kernel_attributes(get_saxpy_kernel, attr, expected_type):
    kernel, _ = get_saxpy_kernel
    method = getattr(kernel.attributes, attr)
    # get the value without providing a device ordinal
    value = method()
    assert value is not None

    # get the value for each device on the system
    for device in system.devices:
        value = method(device.device_id)
    assert isinstance(value, expected_type), f"Expected {attr} to be of type {expected_type}, but got {type(value)}"


def test_object_code_load_cubin(get_saxpy_kernel):
    _, mod = get_saxpy_kernel
    cubin = mod._module
    sym_map = mod._sym_map
    assert isinstance(cubin, bytes)
    mod = ObjectCode.from_cubin(cubin, symbol_mapping=sym_map)
    assert mod.code == cubin
    mod.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_ptx(get_saxpy_kernel_ptx):
    ptx, mod = get_saxpy_kernel_ptx
    sym_map = mod._sym_map
    mod_obj = ObjectCode.from_ptx(ptx, symbol_mapping=sym_map)
    assert mod.code == ptx
    if not Program._can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    mod_obj.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_cubin_from_file(get_saxpy_kernel, tmp_path):
    _, mod = get_saxpy_kernel
    cubin = mod._module
    sym_map = mod._sym_map
    assert isinstance(cubin, bytes)
    cubin_file = tmp_path / "test.cubin"
    cubin_file.write_bytes(cubin)
    mod = ObjectCode.from_cubin(str(cubin_file), symbol_mapping=sym_map)
    assert mod.code == str(cubin_file)
    mod.get_kernel("saxpy<double>")  # force loading


def test_object_code_handle(get_saxpy_object_code):
    mod = get_saxpy_object_code
    assert mod.handle is not None


@skipif_testing_with_compute_sanitizer
def test_saxpy_arguments(get_saxpy_kernel, cuda12_prerequisite_check):
    if not cuda12_prerequisite_check:
        pytest.skip("Test requires CUDA 12")
    krn, _ = get_saxpy_kernel

    assert krn.num_arguments == 5

    assert "ParamInfo" in str(type(krn).arguments_info.fget.__annotations__)
    arg_info = krn.arguments_info
    n_args = len(arg_info)
    assert n_args == krn.num_arguments

    class ExpectedStruct(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_float),
            ("x", ctypes.POINTER(ctypes.c_float)),
            ("y", ctypes.POINTER(ctypes.c_float)),
            ("out", ctypes.POINTER(ctypes.c_float)),
            ("N", ctypes.c_size_t),
        ]

    offsets = [p.offset for p in arg_info]
    sizes = [p.size for p in arg_info]
    members = [getattr(ExpectedStruct, name) for name, _ in ExpectedStruct._fields_]
    expected_offsets = tuple(m.offset for m in members)
    assert all(actual == expected for actual, expected in zip(offsets, expected_offsets))
    expected_sizes = tuple(m.size for m in members)
    assert all(actual == expected for actual, expected in zip(sizes, expected_sizes))


@skipif_testing_with_compute_sanitizer
@pytest.mark.parametrize("nargs", [0, 1, 2, 3, 16])
@pytest.mark.parametrize("c_type_name,c_type", [("int", ctypes.c_int), ("short", ctypes.c_short)], ids=["int", "short"])
def test_num_arguments(init_cuda, nargs, c_type_name, c_type, cuda12_prerequisite_check):
    if not cuda12_prerequisite_check:
        pytest.skip("Test requires CUDA 12")
    args_str = ", ".join([f"{c_type_name} p_{i}" for i in range(nargs)])
    src = f"__global__ void foo{nargs}({args_str}) {{ }}"
    prog = Program(src, code_type="c++")
    mod = prog.compile(
        "cubin",
        name_expressions=(f"foo{nargs}",),
    )
    krn = mod.get_kernel(f"foo{nargs}")
    assert krn.num_arguments == nargs

    class ExpectedStruct(ctypes.Structure):
        _fields_ = [(f"arg_{i}", c_type) for i in range(nargs)]

    members = tuple(getattr(ExpectedStruct, f"arg_{i}") for i in range(nargs))

    arg_info = krn.arguments_info
    assert all([actual.offset == expected.offset for actual, expected in zip(arg_info, members)])
    assert all([actual.size == expected.size for actual, expected in zip(arg_info, members)])


@skipif_testing_with_compute_sanitizer
def test_num_args_error_handling(deinit_all_contexts_function, cuda12_prerequisite_check):
    if not cuda12_prerequisite_check:
        pytest.skip("Test requires CUDA 12")
    src = "__global__ void foo(int a) { }"
    prog = Program(src, code_type="c++")
    mod = prog.compile(
        "cubin",
        name_expressions=("foo",),
    )
    krn = mod.get_kernel("foo")
    # empty driver's context stack using function from conftest
    deinit_all_contexts_function()
    # with no current context, cuKernelGetParamInfo would report
    # exception which we expect to handle by raising
    with pytest.raises(CUDAError):
        # assignment resolves linter error "B018: useless expression"
        _ = krn.num_arguments
