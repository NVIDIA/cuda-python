# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import helpers

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
import pytest
from conftest import skipif_need_cuda_headers

from cuda.core.experimental import (
    Device,
    DeviceMemoryResource,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
from cuda.core.experimental._memory import _SynchronousMemoryResource
from cuda.core.experimental._utils.cuda_utils import CUDAError


def test_launch_config_init(init_cuda):
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), shmem_size=0)
    assert config.grid == (1, 1, 1)
    assert config.block == (1, 1, 1)
    assert config.shmem_size == 0

    config = LaunchConfig(grid=(2, 2, 2), block=(2, 2, 2), shmem_size=1024)
    assert config.grid == (2, 2, 2)
    assert config.block == (2, 2, 2)
    assert config.shmem_size == 1024


def test_launch_config_invalid_values():
    with pytest.raises(ValueError):
        LaunchConfig(grid=0, block=1)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(0, 1), block=1)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(1, 1, 1), block=0)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(1, 1, 1), block=(0, 1))


def test_launch_config_shmem_size():
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), shmem_size=2048)
    assert config.shmem_size == 2048

    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1))
    assert config.shmem_size == 0


def test_launch_config_cluster_grid_conversion(init_cuda):
    """Test that LaunchConfig preserves original grid values and conversion happens in native config."""
    try:
        # Test case 1: 1D - Issue #867 example
        config = LaunchConfig(grid=4, cluster=2, block=32)
        assert config.grid == (4, 1, 1), f"Expected (4, 1, 1), got {config.grid}"
        assert config.cluster == (2, 1, 1), f"Expected (2, 1, 1), got {config.cluster}"
        assert config.block == (32, 1, 1), f"Expected (32, 1, 1), got {config.block}"

        # Test case 2: 2D grid and cluster
        config = LaunchConfig(grid=(2, 3), cluster=(2, 2), block=32)
        assert config.grid == (2, 3, 1), f"Expected (2, 3, 1), got {config.grid}"
        assert config.cluster == (2, 2, 1), f"Expected (2, 2, 1), got {config.cluster}"

        # Test case 3: 3D full specification
        config = LaunchConfig(grid=(2, 2, 2), cluster=(3, 3, 3), block=(8, 8, 8))
        assert config.grid == (2, 2, 2), f"Expected (2, 2, 2), got {config.grid}"
        assert config.cluster == (3, 3, 3), f"Expected (3, 3, 3), got {config.cluster}"

        # Test case 4: Identity case
        config = LaunchConfig(grid=1, cluster=1, block=32)
        assert config.grid == (1, 1, 1), f"Expected (1, 1, 1), got {config.grid}"

        # Test case 5: No cluster (should not convert grid)
        config = LaunchConfig(grid=4, block=32)
        assert config.grid == (4, 1, 1), f"Expected (4, 1, 1), got {config.grid}"
        assert config.cluster is None

    except CUDAError:
        pytest.skip("Driver or GPU not new enough for thread block clusters")


def test_launch_config_native_conversion(init_cuda):
    """Test that _to_native_launch_config correctly converts grid from cluster units to block units."""
    from cuda.core.experimental._launch_config import _to_native_launch_config

    try:
        # Test case 1: 1D - Issue #867 example
        config = LaunchConfig(grid=4, cluster=2, block=32)
        native_config = _to_native_launch_config(config)
        assert native_config.gridDimX == 8, f"Expected gridDimX=8, got {native_config.gridDimX}"
        assert native_config.gridDimY == 1, f"Expected gridDimY=1, got {native_config.gridDimY}"
        assert native_config.gridDimZ == 1, f"Expected gridDimZ=1, got {native_config.gridDimZ}"

        # Test case 2: 2D grid and cluster
        config = LaunchConfig(grid=(2, 3), cluster=(2, 2), block=32)
        native_config = _to_native_launch_config(config)
        assert native_config.gridDimX == 4, f"Expected gridDimX=4, got {native_config.gridDimX}"
        assert native_config.gridDimY == 6, f"Expected gridDimY=6, got {native_config.gridDimY}"
        assert native_config.gridDimZ == 1, f"Expected gridDimZ=1, got {native_config.gridDimZ}"

        # Test case 3: No cluster (should not convert grid)
        config = LaunchConfig(grid=4, block=32)
        native_config = _to_native_launch_config(config)
        assert native_config.gridDimX == 4, f"Expected gridDimX=4, got {native_config.gridDimX}"
        assert native_config.gridDimY == 1, f"Expected gridDimY=1, got {native_config.gridDimY}"
        assert native_config.gridDimZ == 1, f"Expected gridDimZ=1, got {native_config.gridDimZ}"

    except CUDAError:
        pytest.skip("Driver or GPU not new enough for thread block clusters")


def test_launch_invalid_values(init_cuda):
    code = 'extern "C" __global__ void my_kernel() {}'
    program = Program(code, "c++")
    mod = program.compile("cubin")

    stream = Device().create_stream()
    ker = mod.get_kernel("my_kernel")
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), shmem_size=0)

    with pytest.raises(ValueError):
        launch(None, ker, config)

    with pytest.raises(TypeError):
        launch(stream, None, config)

    with pytest.raises(TypeError):
        launch(stream, ker, None)

    launch(stream, config, ker)


# Parametrize: (python_type, cpp_type, init_value)
PARAMS = (
    (bool, "bool", True),
    (float, "double", 2.718),
    (np.bool, "bool", True),
    (np.int8, "signed char", -42),
    (np.int16, "signed short", -1234),
    (np.int32, "signed int", -123456),
    (np.int64, "signed long long", -123456789),
    (np.uint8, "unsigned char", 42),
    (np.uint16, "unsigned short", 1234),
    (np.uint32, "unsigned int", 123456),
    (np.uint64, "unsigned long long", 123456789),
    (np.float32, "float", 3.14),
    (np.float64, "double", 2.718),
    (ctypes.c_bool, "bool", True),
    (ctypes.c_int8, "signed char", -42),
    (ctypes.c_int16, "signed short", -1234),
    (ctypes.c_int32, "signed int", -123456),
    (ctypes.c_int64, "signed long long", -123456789),
    (ctypes.c_uint8, "unsigned char", 42),
    (ctypes.c_uint16, "unsigned short", 1234),
    (ctypes.c_uint32, "unsigned int", 123456),
    (ctypes.c_uint64, "unsigned long long", 123456789),
    (ctypes.c_float, "float", 3.14),
    (ctypes.c_double, "double", 2.718),
)
if helpers.CCCL_INCLUDE_PATHS is not None:
    PARAMS += (
        (np.float16, "half", 0.78),
        (np.complex64, "cuda::std::complex<float>", 1 + 2j),
        (np.complex128, "cuda::std::complex<double>", -3 - 4j),
        (complex, "cuda::std::complex<double>", 5 - 7j),
    )


@pytest.mark.parametrize("python_type, cpp_type, init_value", PARAMS)
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_launch_scalar_argument(python_type, cpp_type, init_value):
    dev = Device()
    dev.set_current()

    # Prepare pinned host array
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(np.dtype(python_type).itemsize)
    arr = np.from_dlpack(b).view(python_type)
    arr[:] = 0

    # Prepare scalar argument in Python
    scalar = python_type(init_value)

    # CUDA kernel templated on type T
    code = r"""
    template <typename T>
    __global__ void write_scalar(T* arr, T val) {
        arr[0] = val;
    }
    """

    # Compile and force instantiation for this type
    arch = "".join(f"{i}" for i in dev.compute_capability)
    if helpers.CCCL_INCLUDE_PATHS is not None:
        code = (
            r"""
        #include <cuda_fp16.h>
        #include <cuda/std/complex>
        """
            + code
        )
    pro_opts = ProgramOptions(std="c++17", arch=f"sm_{arch}", include_path=helpers.CCCL_INCLUDE_PATHS)
    prog = Program(code, code_type="c++", options=pro_opts)
    ker_name = f"write_scalar<{cpp_type}>"
    mod = prog.compile("cubin", name_expressions=(ker_name,))
    ker = mod.get_kernel(ker_name)

    # Launch with 1 thread
    config = LaunchConfig(grid=1, block=1)
    launch(dev.default_stream, config, ker, arr.ctypes.data, scalar)
    dev.default_stream.sync()

    # Check result
    assert arr[0] == init_value, f"Expected {init_value}, got {arr[0]}"


@skipif_need_cuda_headers  # cg
def test_cooperative_launch():
    dev = Device()
    dev.set_current()
    s = dev.create_stream(options={"nonblocking": True})

    # CUDA kernel templated on type T
    code = r"""
    #include <cooperative_groups.h>

    extern "C" __global__ void test_grid_sync() {
        namespace cg = cooperative_groups;
        auto grid = cg::this_grid();
        grid.sync();
    }
    """

    # Compile and force instantiation for this type
    arch = "".join(f"{i}" for i in dev.compute_capability)
    pro_opts = ProgramOptions(std="c++17", arch=f"sm_{arch}", include_path=helpers.CCCL_INCLUDE_PATHS)
    prog = Program(code, code_type="c++", options=pro_opts)
    ker = prog.compile("cubin").get_kernel("test_grid_sync")

    # # Launch without setting cooperative_launch
    # # Commented out as this seems to be a sticky error...
    # config = LaunchConfig(grid=1, block=1)
    # launch(s, config, ker)
    # from cuda.core.experimental._utils.cuda_utils import CUDAError
    # with pytest.raises(CUDAError) as e:
    #     s.sync()
    # assert "CUDA_ERROR_LAUNCH_FAILED" in str(e)

    # Crazy grid sizes would not work
    block = 128
    config = LaunchConfig(grid=dev.properties.max_grid_dim_x // block + 1, block=block, cooperative_launch=True)
    with pytest.raises(ValueError):
        launch(s, config, ker)

    # This works just fine
    config = LaunchConfig(grid=1, block=1, cooperative_launch=True)
    launch(s, config, ker)
    s.sync()


@pytest.mark.skipif(cp is None, reason="cupy not installed")
@pytest.mark.parametrize(
    "memory_resource_class",
    [
        "device_memory_resource",  # kludgy, but can go away after #726 is resolved
        pytest.param(
            LegacyPinnedMemoryResource,
            marks=pytest.mark.skipif(
                tuple(int(i) for i in np.__version__.split(".")[:3]) < (2, 2, 5),
                reason="need numpy 2.2.5+, numpy GH #28632",
            ),
        ),
    ],
)
def test_launch_with_buffers_allocated_by_memory_resource(init_cuda, memory_resource_class):
    """Test that kernels can access memory allocated by memory resources."""
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    # tell CuPy to use our stream as the current stream:
    cp.cuda.ExternalStream(int(stream.handle)).use()

    # Kernel that operates on memory
    code = """
    extern "C"
    __global__ void memory_ops(float* data, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < N) {
            // Access memory (device or pinned)
            data[tid] = data[tid] * 3.0f;
        }
    }
    """

    # Compile kernel
    arch = "".join(f"{i}" for i in dev.compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("memory_ops")

    # Create memory resource
    if memory_resource_class == "device_memory_resource":
        if dev.properties.memory_pools_supported:
            mr = DeviceMemoryResource(dev.device_id)
        else:
            mr = _SynchronousMemoryResource(dev.device_id)
        name = memory_resource_class
    else:
        mr = memory_resource_class()
        name = str(mr)

    # Allocate memory
    size = 1024
    dtype = np.float32
    element_size = dtype().itemsize
    total_size = size * element_size

    buffer = mr.allocate(total_size, stream=stream)

    # Create array view based on memory type
    if mr.is_host_accessible:
        # For pinned memory, use numpy
        array = np.from_dlpack(buffer).view(dtype=dtype)
    else:
        array = cp.from_dlpack(buffer).view(dtype=dtype)

    # Initialize data with random values
    if mr.is_host_accessible:
        rng = np.random.default_rng()
        array[:] = rng.random(size, dtype=dtype)
    else:
        rng = cp.random.default_rng()
        array[:] = rng.random(size, dtype=dtype)

    # Store original values for verification
    original = array.copy()

    # Sync before kernel launch
    stream.sync()

    # Launch kernel
    block = 256
    grid = (size + block - 1) // block
    config = LaunchConfig(grid=grid, block=block)

    launch(stream, config, kernel, buffer, np.uint64(size))
    stream.sync()

    # Verify kernel operations
    assert cp.allclose(array, original * 3.0), f"{name} operation failed"

    # Clean up
    buffer.close(stream)
    stream.close()

    cp.cuda.Stream.null.use()  # reset CuPy's current stream to the null stream

    # Verify buffer is properly closed
    assert buffer.handle is None, f"{name} buffer should be closed"
