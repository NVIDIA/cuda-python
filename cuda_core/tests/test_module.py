# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import pickle
import warnings

import cuda.core
import pytest
from cuda.core import Device, ObjectCode, Program, ProgramOptions, Kernel
from cuda.core._utils.cuda_utils import CUDAError, driver, get_binding_version, handle_return

try:
    import numba
except ImportError:
    numba = None

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
def cuda12_4_prerequisite_check():
    # binding availability depends on cuda-python version
    # and version of underlying CUDA toolkit
    _py_major_ver, _ = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    return _py_major_ver >= 12 and _driver_ver >= 12040


def test_kernel_attributes_init_disabled():
    with pytest.raises(RuntimeError, match=r"^KernelAttributes cannot be instantiated directly\."):
        cuda.core._module.KernelAttributes()  # Ensure back door is locked.


def test_kernel_occupancy_init_disabled():
    with pytest.raises(RuntimeError, match=r"^KernelOccupancy cannot be instantiated directly\."):
        cuda.core._module.KernelOccupancy()  # Ensure back door is locked.


def test_kernel_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Kernel objects cannot be instantiated directly\."):
        cuda.core._module.Kernel()  # Ensure back door is locked.


def test_object_code_init_disabled():
    with pytest.raises(RuntimeError, match=r"^ObjectCode objects cannot be instantiated directly\."):
        ObjectCode()  # Reject at front door.


@pytest.fixture(scope="function")
def get_saxpy_kernel_cubin(init_cuda):
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
    # prepare program
    prog = Program(SAXPY_KERNEL, code_type="c++")
    mod = prog.compile(
        "ptx",
        name_expressions=("saxpy<float>", "saxpy<double>"),
    )
    ptx = mod._module
    return ptx, mod


@pytest.fixture(scope="function")
def get_saxpy_kernel_ltoir(init_cuda):
    # Create LTOIR code using link-time optimization
    prog = Program(SAXPY_KERNEL, code_type="c++", options=ProgramOptions(link_time_optimization=True))
    mod = prog.compile("ltoir", name_expressions=("saxpy<float>", "saxpy<double>"))
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
def test_read_only_kernel_attributes(get_saxpy_kernel_cubin, attr, expected_type):
    kernel, _ = get_saxpy_kernel_cubin
    method = getattr(kernel.attributes, attr)
    # get the value without providing a device ordinal
    value = method()
    assert value is not None

    # get the value for each device on the system, using either the device object or ordinal
    for device in Device.get_all_devices():
        value = method(device)
        value = method(device.device_id)
    assert isinstance(value, expected_type), f"Expected {attr} to be of type {expected_type}, but got {type(value)}"


def test_object_code_load_ptx(get_saxpy_kernel_ptx):
    ptx, mod = get_saxpy_kernel_ptx
    sym_map = mod._sym_map
    mod_obj = ObjectCode.from_ptx(ptx, symbol_mapping=sym_map)
    assert mod.code == ptx
    if not Program._can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    mod_obj.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_ptx_from_file(get_saxpy_kernel_ptx, tmp_path):
    ptx, mod = get_saxpy_kernel_ptx
    sym_map = mod._sym_map
    assert isinstance(ptx, bytes)
    ptx_file = tmp_path / "test.ptx"
    ptx_file.write_bytes(ptx)
    mod_obj = ObjectCode.from_ptx(str(ptx_file), symbol_mapping=sym_map)
    assert mod_obj.code == str(ptx_file)
    assert mod_obj.code_type == "ptx"
    if not Program._can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    mod_obj.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_cubin(get_saxpy_kernel_cubin):
    _, mod = get_saxpy_kernel_cubin
    cubin = mod._module
    sym_map = mod._sym_map
    assert isinstance(cubin, bytes)
    mod = ObjectCode.from_cubin(cubin, symbol_mapping=sym_map)
    assert mod.code == cubin
    mod.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_cubin_from_file(get_saxpy_kernel_cubin, tmp_path):
    _, mod = get_saxpy_kernel_cubin
    cubin = mod._module
    sym_map = mod._sym_map
    assert isinstance(cubin, bytes)
    cubin_file = tmp_path / "test.cubin"
    cubin_file.write_bytes(cubin)
    mod = ObjectCode.from_cubin(str(cubin_file), symbol_mapping=sym_map)
    assert mod.code == str(cubin_file)
    mod.get_kernel("saxpy<double>")  # force loading


def test_object_code_handle(get_saxpy_kernel_cubin):
    _, mod = get_saxpy_kernel_cubin
    assert mod.handle is not None


def test_object_code_load_ltoir(get_saxpy_kernel_ltoir):
    mod = get_saxpy_kernel_ltoir
    ltoir = mod._module
    sym_map = mod._sym_map
    assert isinstance(ltoir, bytes)
    mod_obj = ObjectCode.from_ltoir(ltoir, symbol_mapping=sym_map)
    assert mod_obj.code == ltoir
    assert mod_obj.code_type == "ltoir"
    # ltoir doesn't support kernel retrieval directly as it's used for linking
    assert mod_obj._handle is None
    # Test that get_kernel fails for unsupported code type
    with pytest.raises(RuntimeError, match=r'Unsupported code type "ltoir"'):
        mod_obj.get_kernel("saxpy<float>")


def test_object_code_load_ltoir_from_file(get_saxpy_kernel_ltoir, tmp_path):
    mod = get_saxpy_kernel_ltoir
    ltoir = mod._module
    sym_map = mod._sym_map
    assert isinstance(ltoir, bytes)
    ltoir_file = tmp_path / "test.ltoir"
    ltoir_file.write_bytes(ltoir)
    mod_obj = ObjectCode.from_ltoir(str(ltoir_file), symbol_mapping=sym_map)
    assert mod_obj.code == str(ltoir_file)
    assert mod_obj.code_type == "ltoir"
    # ltoir doesn't support kernel retrieval directly as it's used for linking
    assert mod_obj._handle is None


def test_saxpy_arguments(get_saxpy_kernel_cubin, cuda12_4_prerequisite_check):
    krn, _ = get_saxpy_kernel_cubin

    if cuda12_4_prerequisite_check:
        assert krn.num_arguments == 5
    else:
        with pytest.raises(NotImplementedError):
            _ = krn.num_arguments
        return

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


@pytest.mark.parametrize("nargs", [0, 1, 2, 3, 16])
@pytest.mark.parametrize("c_type_name,c_type", [("int", ctypes.c_int), ("short", ctypes.c_short)], ids=["int", "short"])
def test_num_arguments(init_cuda, nargs, c_type_name, c_type, cuda12_4_prerequisite_check):
    if not cuda12_4_prerequisite_check:
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


def test_num_args_error_handling(deinit_all_contexts_function, cuda12_4_prerequisite_check):
    if not cuda12_4_prerequisite_check:
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


@pytest.mark.parametrize("block_size", [32, 64, 96, 120, 128, 256])
@pytest.mark.parametrize("smem_size_per_block", [0, 32, 4096])
def test_occupancy_max_active_block_per_multiprocessor(get_saxpy_kernel_cubin, block_size, smem_size_per_block):
    kernel, _ = get_saxpy_kernel_cubin
    dev_props = Device().properties
    assert block_size <= dev_props.max_threads_per_block
    assert smem_size_per_block <= dev_props.max_shared_memory_per_block
    num_blocks_per_sm = kernel.occupancy.max_active_blocks_per_multiprocessor(block_size, smem_size_per_block)
    assert isinstance(num_blocks_per_sm, int)
    assert num_blocks_per_sm > 0
    kernel_threads_per_sm = num_blocks_per_sm * block_size
    kernel_smem_size_per_sm = num_blocks_per_sm * smem_size_per_block
    assert kernel_threads_per_sm <= dev_props.max_threads_per_multiprocessor
    assert kernel_smem_size_per_sm <= dev_props.max_shared_memory_per_multiprocessor
    assert kernel.attributes.num_regs() * num_blocks_per_sm <= dev_props.max_registers_per_multiprocessor


@pytest.mark.parametrize("block_size_limit", [32, 64, 96, 120, 128, 256, 0])
@pytest.mark.parametrize("smem_size_per_block", [0, 32, 4096])
def test_occupancy_max_potential_block_size_constant(get_saxpy_kernel_cubin, block_size_limit, smem_size_per_block):
    """Tests use case when shared memory needed is independent on the block size"""
    kernel, _ = get_saxpy_kernel_cubin
    dev_props = Device().properties
    assert block_size_limit <= dev_props.max_threads_per_block
    assert smem_size_per_block <= dev_props.max_shared_memory_per_block
    config_data = kernel.occupancy.max_potential_block_size(smem_size_per_block, block_size_limit)
    assert isinstance(config_data, tuple)
    assert len(config_data) == 2
    min_grid_size, max_block_size = config_data
    assert isinstance(min_grid_size, int)
    assert isinstance(max_block_size, int)
    assert min_grid_size > 0
    assert max_block_size > 0
    if block_size_limit > 0:
        assert max_block_size <= block_size_limit
    else:
        assert max_block_size <= dev_props.max_threads_per_block
    assert min_grid_size == config_data.min_grid_size
    assert max_block_size == config_data.max_block_size
    invalid_dsmem = Ellipsis
    with pytest.raises(TypeError):
        kernel.occupancy.max_potential_block_size(invalid_dsmem, block_size_limit)


@pytest.mark.skipif(numba is None, reason="Test requires numba to be installed")
@pytest.mark.parametrize("block_size_limit", [32, 64, 96, 120, 128, 277, 0])
def test_occupancy_max_potential_block_size_b2dsize(get_saxpy_kernel_cubin, block_size_limit):
    """Tests use case when shared memory needed depends on the block size"""
    kernel, _ = get_saxpy_kernel_cubin

    def shared_memory_needed(block_size: numba.intc) -> numba.size_t:
        "Size of dynamic shared memory needed by kernel of this block size"
        return 1024 * (block_size // 32)

    b2dsize_sig = numba.size_t(numba.intc)
    dsmem_needed_cfunc = numba.cfunc(b2dsize_sig)(shared_memory_needed)
    fn_ptr = ctypes.cast(dsmem_needed_cfunc.ctypes, ctypes.c_void_p).value
    b2dsize_fn = driver.CUoccupancyB2DSize(_ptr=fn_ptr)
    config_data = kernel.occupancy.max_potential_block_size(b2dsize_fn, block_size_limit)
    dev_props = Device().properties
    assert block_size_limit <= dev_props.max_threads_per_block
    min_grid_size, max_block_size = config_data
    assert isinstance(min_grid_size, int)
    assert isinstance(max_block_size, int)
    assert min_grid_size > 0
    assert max_block_size > 0
    if block_size_limit > 0:
        assert max_block_size <= block_size_limit
    else:
        assert max_block_size <= dev_props.max_threads_per_block


@pytest.mark.parametrize("num_blocks_per_sm, block_size", [(4, 32), (2, 64), (2, 96), (3, 120), (2, 128), (1, 256)])
def test_occupancy_available_dynamic_shared_memory_per_block(get_saxpy_kernel_cubin, num_blocks_per_sm, block_size):
    kernel, _ = get_saxpy_kernel_cubin
    dev_props = Device().properties
    assert block_size <= dev_props.max_threads_per_block
    assert num_blocks_per_sm * block_size <= dev_props.max_threads_per_multiprocessor
    smem_size = kernel.occupancy.available_dynamic_shared_memory_per_block(num_blocks_per_sm, block_size)
    assert smem_size <= dev_props.max_shared_memory_per_block
    assert num_blocks_per_sm * smem_size <= dev_props.max_shared_memory_per_multiprocessor


@pytest.mark.parametrize("cluster", [None, 2])
def test_occupancy_max_active_clusters(get_saxpy_kernel_cubin, cluster):
    kernel, _ = get_saxpy_kernel_cubin
    dev = Device()
    if dev.compute_capability < (9, 0):
        pytest.skip("Device with compute capability 90 or higher is required for cluster support")
    launch_config = cuda.core.LaunchConfig(grid=128, block=64, cluster=cluster)
    query_fn = kernel.occupancy.max_active_clusters
    max_active_clusters = query_fn(launch_config)
    assert isinstance(max_active_clusters, int)
    assert max_active_clusters >= 0
    max_active_clusters = query_fn(launch_config, stream=dev.default_stream)
    assert isinstance(max_active_clusters, int)
    assert max_active_clusters >= 0


def test_occupancy_max_potential_cluster_size(get_saxpy_kernel_cubin):
    kernel, _ = get_saxpy_kernel_cubin
    dev = Device()
    if dev.compute_capability < (9, 0):
        pytest.skip("Device with compute capability 90 or higher is required for cluster support")
    launch_config = cuda.core.LaunchConfig(grid=128, block=64)
    query_fn = kernel.occupancy.max_potential_cluster_size
    max_potential_cluster_size = query_fn(launch_config)
    assert isinstance(max_potential_cluster_size, int)
    assert max_potential_cluster_size >= 0
    max_potential_cluster_size = query_fn(launch_config, stream=dev.default_stream)
    assert isinstance(max_potential_cluster_size, int)
    assert max_potential_cluster_size >= 0


def test_module_serialization_roundtrip(get_saxpy_kernel_cubin):
    _, objcode = get_saxpy_kernel_cubin
    result = pickle.loads(pickle.dumps(objcode))  # noqa: S403, S301

    assert isinstance(result, ObjectCode)
    assert objcode.code == result.code
    assert objcode._sym_map == result._sym_map
    assert objcode.code_type == result.code_type


def test_kernel_from_handle(get_saxpy_kernel_cubin):
    """Test Kernel.from_handle() with a valid handle"""
    original_kernel, objcode = get_saxpy_kernel_cubin

    # Get the handle from the original kernel
    handle = int(original_kernel._handle)

    # Create a new Kernel from the handle
    kernel_from_handle = Kernel.from_handle(handle, objcode)
    assert isinstance(kernel_from_handle, Kernel)

    # Verify we can access kernel attributes
    max_threads = kernel_from_handle.attributes.max_threads_per_block()
    assert isinstance(max_threads, int)
    assert max_threads > 0


def test_kernel_from_handle_no_module(get_saxpy_kernel_cubin):
    """Test Kernel.from_handle() without providing a module"""
    original_kernel, _ = get_saxpy_kernel_cubin

    # Get the handle from the original kernel
    handle = int(original_kernel._handle)

    # Create a new Kernel from the handle without a module
    kernel_from_handle = Kernel.from_handle(handle)
    assert isinstance(kernel_from_handle, Kernel)

    # Verify we can still access kernel attributes
    max_threads = kernel_from_handle.attributes.max_threads_per_block()
    assert isinstance(max_threads, int)
    assert max_threads > 0


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param("not_an_int", id="str"),
        pytest.param(2.71828, id="float"),
        pytest.param(None, id="None"),
        pytest.param({"handle": 123}, id="dict"),
        pytest.param([456], id="list"),
        pytest.param((789,), id="tuple"),
        pytest.param(3 + 4j, id="complex"),
        pytest.param(b"\xde\xad\xbe\xef", id="bytes"),
        pytest.param({999}, id="set"),
        pytest.param(object(), id="object"),
    ],
)
def test_kernel_from_handle_type_validation(invalid_value):
    """Test Kernel.from_handle() with wrong handle types"""
    with pytest.raises(TypeError):
        Kernel.from_handle(invalid_value)


def test_kernel_from_handle_invalid_module_type(get_saxpy_kernel_cubin):
    """Test Kernel.from_handle() with invalid module parameter"""
    original_kernel, _ = get_saxpy_kernel_cubin
    handle = int(original_kernel._handle)

    # Invalid module type (should fail type assertion in _from_obj)
    with pytest.raises((TypeError, AssertionError)):
        Kernel.from_handle(handle, mod="not_an_objectcode")

    with pytest.raises((TypeError, AssertionError)):
        Kernel.from_handle(handle, mod=12345)


def test_kernel_from_handle_multiple_instances(get_saxpy_kernel_cubin):
    """Test creating multiple Kernel instances from the same handle"""
    original_kernel, objcode = get_saxpy_kernel_cubin
    handle = int(original_kernel._handle)

    # Create multiple Kernel instances from the same handle
    kernel1 = Kernel.from_handle(handle, objcode)
    kernel2 = Kernel.from_handle(handle, objcode)
    kernel3 = Kernel.from_handle(handle, objcode)

    # All should be valid Kernel objects
    assert isinstance(kernel1, Kernel)
    assert isinstance(kernel2, Kernel)
    assert isinstance(kernel3, Kernel)

    # All should reference the same underlying CUDA kernel handle
    assert int(kernel1._handle) == int(kernel2._handle) == int(kernel3._handle) == handle
