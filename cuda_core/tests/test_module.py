# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import pickle
import warnings

import cuda.core.experimental
import pytest
from cuda.core.experimental import Device, ObjectCode, Program, ProgramOptions, system
from cuda.core.experimental._utils.cuda_utils import CUDAError, driver, get_binding_version, handle_return

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
        cuda.core.experimental._module.KernelAttributes()  # Ensure back door is locked.


def test_kernel_occupancy_init_disabled():
    with pytest.raises(RuntimeError, match=r"^KernelOccupancy cannot be instantiated directly\."):
        cuda.core.experimental._module.KernelOccupancy()  # Ensure back door is locked.


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
    assert isinstance(ptx, str)
    ptx_file = tmp_path / "test.ptx"
    ptx_file.write_text(ptx)
    mod_obj = ObjectCode.from_ptx(str(ptx_file), symbol_mapping=sym_map)
    assert mod_obj.code == str(ptx_file)
    assert mod_obj._code_type == "ptx"
    if not Program._can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    mod_obj.get_kernel("saxpy<double>")  # force loading


def test_object_code_load_cubin(get_saxpy_kernel):
    _, mod = get_saxpy_kernel
    cubin = mod._module
    sym_map = mod._sym_map
    assert isinstance(cubin, bytes)
    mod = ObjectCode.from_cubin(cubin, symbol_mapping=sym_map)
    assert mod.code == cubin
    mod.get_kernel("saxpy<double>")  # force loading


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


def test_object_code_load_ltoir(get_saxpy_kernel_ltoir):
    mod = get_saxpy_kernel_ltoir
    ltoir = mod._module
    sym_map = mod._sym_map
    assert isinstance(ltoir, bytes)
    mod_obj = ObjectCode.from_ltoir(ltoir, symbol_mapping=sym_map)
    assert mod_obj.code == ltoir
    assert mod_obj._code_type == "ltoir"
    # ltoir doesn't support kernel retrieval directly as it's used for linking
    assert mod_obj._handle is None  # Should only be loaded when needed
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
    assert mod_obj._code_type == "ltoir"
    assert mod_obj._handle is None  # Should only be loaded when needed


def test_object_code_load_fatbin(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test fatbin loading using NVCC-generated fatbins.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    # Generate fatbin for multiple architectures
    archs = ["sm_75", "sm_90", current_arch]
    arch_flags = " ".join(f"--gpu-architecture={arch}" for arch in set(archs))

    fatbin_file = tmp_path / "kernel.fatbin"

    try:
        # Generate fatbin using nvcc
        cmd = f"nvcc --fatbin {arch_flags} -o {fatbin_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate fatbin with nvcc: {e}")

    # Test loading fatbin from bytes (in-memory)
    fatbin_bytes = fatbin_file.read_bytes()
    mod_obj_mem = ObjectCode.from_fatbin(fatbin_bytes, name="fatbin_memory")
    assert mod_obj_mem.code == fatbin_bytes
    assert mod_obj_mem._code_type == "fatbin"
    assert mod_obj_mem.name == "fatbin_memory"


def test_object_code_load_fatbin_from_file(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test fatbin loading from file path using NVCC-generated fatbins.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    # Generate fatbin for multiple architectures
    archs = ["sm_75", "sm_90", current_arch]
    arch_flags = " ".join(f"--gpu-architecture={arch}" for arch in set(archs))

    fatbin_file = tmp_path / "kernel.fatbin"

    try:
        # Generate fatbin using nvcc
        cmd = f"nvcc --fatbin {arch_flags} -o {fatbin_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate fatbin with nvcc: {e}")

    # Test loading fatbin from file path
    mod_obj_file = ObjectCode.from_fatbin(str(fatbin_file), name="fatbin_file")
    assert mod_obj_file.code == str(fatbin_file)
    assert mod_obj_file._code_type == "fatbin"
    assert mod_obj_file.name == "fatbin_file"


def test_object_code_load_object(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test object code loading using NVCC-generated object files.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    object_file = tmp_path / "kernel.o"

    try:
        # Generate object file using nvcc
        cmd = f"nvcc --device-c --gpu-architecture={current_arch} -o {object_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate object file with nvcc: {e}")

    # Test loading object from bytes (in-memory)
    object_bytes = object_file.read_bytes()
    mod_obj_mem = ObjectCode.from_object(object_bytes, name="object_memory")
    assert mod_obj_mem.code == object_bytes
    assert mod_obj_mem._code_type == "object"
    assert mod_obj_mem.name == "object_memory"
    # object code doesn't support direct kernel retrieval
    assert mod_obj_mem._handle is None  # Should only be loaded when needed
    # Test that get_kernel fails for unsupported code type
    with pytest.raises(RuntimeError, match=r'Unsupported code type "object"'):
        mod_obj_mem.get_kernel("simple_kernel")


def test_object_code_load_object_from_file(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test object code loading from file path using NVCC-generated object files.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    object_file = tmp_path / "kernel.o"

    try:
        # Generate object file using nvcc
        cmd = f"nvcc --device-c --gpu-architecture={current_arch} -o {object_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate object file with nvcc: {e}")

    # Test loading object from file path
    mod_obj_file = ObjectCode.from_object(str(object_file), name="object_file")
    assert mod_obj_file.code == str(object_file)
    assert mod_obj_file._code_type == "object"
    assert mod_obj_file.name == "object_file"
    assert mod_obj_file._handle is None  # Should only be loaded when needed


def test_object_code_load_library(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test library loading using NVCC-generated library files.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    object_file = tmp_path / "kernel.o"
    library_file = tmp_path / "libkernel.a"

    try:
        # Generate object file first
        cmd = f"nvcc --device-c --gpu-architecture={current_arch} -o {object_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        # Create library from object file
        cmd = f"ar rcs {library_file} {object_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate library with nvcc/ar: {e}")

    # Test loading library from bytes (in-memory)
    library_bytes = library_file.read_bytes()
    mod_obj_mem = ObjectCode.from_library(library_bytes, name="library_memory")
    assert mod_obj_mem.code == library_bytes
    assert mod_obj_mem._code_type == "library"
    assert mod_obj_mem.name == "library_memory"
    assert mod_obj_mem._handle is None  # Should only be loaded when needed


def test_object_code_load_library_from_file(get_saxpy_kernel_ltoir, tmp_path):
    """
    Test library loading from file path using NVCC-generated library files.
    TODO: Can drop NVCC from test dependency once #156 is resolved.
    """
    import shutil
    import subprocess

    # Check if NVCC is available
    if not shutil.which("nvcc"):
        pytest.skip("NVCC not available in PATH")

    # Create a simple CUDA kernel file
    kernel_source = """
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
"""

    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text(kernel_source)

    # Get current device architecture
    from cuda.core.experimental import Device

    current_arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

    object_file = tmp_path / "kernel.o"
    library_file = tmp_path / "libkernel.a"

    try:
        # Generate object file first
        cmd = f"nvcc --device-c --gpu-architecture={current_arch} -o {object_file} {cu_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        # Create library from object file
        cmd = f"ar rcs {library_file} {object_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate library with nvcc/ar: {e}")

    # Test loading library from file path
    mod_obj_file = ObjectCode.from_library(str(library_file), name="library_file")
    assert mod_obj_file.code == str(library_file)
    assert mod_obj_file._code_type == "library"
    assert mod_obj_file.name == "library_file"
    assert mod_obj_file._handle is None  # Should only be loaded when needed


def test_object_code_file_path_linker_integration(get_saxpy_kernel, tmp_path):
    """Test that ObjectCode created from file paths works with the Linker"""
    _, mod = get_saxpy_kernel
    cubin = mod._module
    assert isinstance(cubin, bytes)

    # Create temporary files for different code types
    test_files = {}
    for code_type in ["cubin", "ptx", "ltoir", "fatbin", "object", "library"]:
        file_path = tmp_path / f"test.{code_type}"
        file_path.write_bytes(cubin)  # Use cubin bytes as proxy for all types
        test_files[code_type] = str(file_path)

    # Create ObjectCode instances from file paths
    file_based_objects = []
    for code_type, file_path in test_files.items():
        if code_type == "cubin":
            obj = ObjectCode.from_cubin(file_path, name=f"file_{code_type}")
        elif code_type == "ptx":
            obj = ObjectCode.from_ptx(file_path, name=f"file_{code_type}")
        elif code_type == "ltoir":
            obj = ObjectCode.from_ltoir(file_path, name=f"file_{code_type}")
        elif code_type == "fatbin":
            obj = ObjectCode.from_fatbin(file_path, name=f"file_{code_type}")
        elif code_type == "object":
            obj = ObjectCode.from_object(file_path, name=f"file_{code_type}")
        elif code_type == "library":
            obj = ObjectCode.from_library(file_path, name=f"file_{code_type}")

        # Verify the ObjectCode was created correctly
        assert obj.code == file_path
        assert obj._code_type == code_type
        assert obj.name == f"file_{code_type}"
        assert isinstance(obj._module, str)  # Should store the file path
        file_based_objects.append(obj)

    # Test that these ObjectCode instances can be used with Linker
    # Note: We can't actually link most of these types together in practice,
    # but we can verify the linker accepts them and handles the file path correctly
    from cuda.core.experimental import Linker, LinkerOptions

    # Test with ptx which should be linkable (use only PTX for actual linking)
    ptx_obj = None
    for obj in file_based_objects:
        if obj._code_type == "ptx":
            ptx_obj = obj
            break

    if ptx_obj is not None:
        # Create a simple linker test - this will test that _add_code_object
        # handles file paths correctly by not crashing on the file path
        try:
            arch = "sm_" + "".join(f"{i}" for i in Device().compute_capability)
            options = LinkerOptions(arch=arch)
            # This should not crash - it should handle the file path in _add_code_object
            linker = Linker(ptx_obj, options=options)
            # We don't need to actually link since that might fail due to content,
            # but creating the linker tests our file path handling
            assert linker is not None
        except Exception as e:
            # If it fails, it should be due to content issues, not file path handling
            # The key is that it should not fail with "Expected type bytes, but got str"
            assert "Expected type bytes, but got str" not in str(e), f"File path handling failed: {e}"


def test_saxpy_arguments(get_saxpy_kernel, cuda12_4_prerequisite_check):
    krn, _ = get_saxpy_kernel

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
def test_occupancy_max_active_block_per_multiprocessor(get_saxpy_kernel, block_size, smem_size_per_block):
    kernel, _ = get_saxpy_kernel
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
def test_occupancy_max_potential_block_size_constant(get_saxpy_kernel, block_size_limit, smem_size_per_block):
    """Tests use case when shared memory needed is independent on the block size"""
    kernel, _ = get_saxpy_kernel
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
def test_occupancy_max_potential_block_size_b2dsize(get_saxpy_kernel, block_size_limit):
    """Tests use case when shared memory needed depends on the block size"""
    kernel, _ = get_saxpy_kernel

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
def test_occupancy_available_dynamic_shared_memory_per_block(get_saxpy_kernel, num_blocks_per_sm, block_size):
    kernel, _ = get_saxpy_kernel
    dev_props = Device().properties
    assert block_size <= dev_props.max_threads_per_block
    assert num_blocks_per_sm * block_size <= dev_props.max_threads_per_multiprocessor
    smem_size = kernel.occupancy.available_dynamic_shared_memory_per_block(num_blocks_per_sm, block_size)
    assert smem_size <= dev_props.max_shared_memory_per_block
    assert num_blocks_per_sm * smem_size <= dev_props.max_shared_memory_per_multiprocessor


@pytest.mark.parametrize("cluster", [None, 2])
def test_occupancy_max_active_clusters(get_saxpy_kernel, cluster):
    kernel, _ = get_saxpy_kernel
    dev = Device()
    if dev.compute_capability < (9, 0):
        pytest.skip("Device with compute capability 90 or higher is required for cluster support")
    launch_config = cuda.core.experimental.LaunchConfig(grid=128, block=64, cluster=cluster)
    query_fn = kernel.occupancy.max_active_clusters
    max_active_clusters = query_fn(launch_config)
    assert isinstance(max_active_clusters, int)
    assert max_active_clusters >= 0
    max_active_clusters = query_fn(launch_config, stream=dev.default_stream)
    assert isinstance(max_active_clusters, int)
    assert max_active_clusters >= 0


def test_occupancy_max_potential_cluster_size(get_saxpy_kernel):
    kernel, _ = get_saxpy_kernel
    dev = Device()
    if dev.compute_capability < (9, 0):
        pytest.skip("Device with compute capability 90 or higher is required for cluster support")
    launch_config = cuda.core.experimental.LaunchConfig(grid=128, block=64)
    query_fn = kernel.occupancy.max_potential_cluster_size
    max_potential_cluster_size = query_fn(launch_config)
    assert isinstance(max_potential_cluster_size, int)
    assert max_potential_cluster_size >= 0
    max_potential_cluster_size = query_fn(launch_config, stream=dev.default_stream)
    assert isinstance(max_potential_cluster_size, int)
    assert max_potential_cluster_size >= 0


def test_module_serialization_roundtrip(get_saxpy_kernel):
    _, objcode = get_saxpy_kernel
    result = pickle.loads(pickle.dumps(objcode))  # noqa: S403, S301

    assert isinstance(result, ObjectCode)
    assert objcode.code == result.code
    assert objcode._sym_map == result._sym_map
    assert objcode._code_type == result._code_type
