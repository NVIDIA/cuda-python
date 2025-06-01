# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import pathlib

import numpy as np
import pytest

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch
from cuda.core.experimental._memory import _DefaultPinnedMemorySource


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
    (bool,         'bool',          True),
    (float, 'double',   2.718),
    (np.bool,         'bool',          True),
    (np.int8,         'signed char',   -42),
    (np.int16,        'signed short',  -1234),
    (np.int32,        'signed int',  -123456),
    (np.int64,        'signed long long',  -123456789),
    (np.uint8,        'unsigned char',  42),
    (np.uint16,       'unsigned short', 1234),
    (np.uint32,       'unsigned int', 123456),
    (np.uint64,       'unsigned long long', 123456789),
    (np.float32,      'float',    3.14),
    (np.float64,      'double',   2.718),
    (ctypes.c_bool,   'bool',          True),
    (ctypes.c_int8,   'signed char',   -42),
    (ctypes.c_int16,  'signed short',  -1234),
    (ctypes.c_int32,  'signed int',  -123456),
    (ctypes.c_int64,  'signed long long',  -123456789),
    (ctypes.c_uint8,  'unsigned char',  42),
    (ctypes.c_uint16, 'unsigned short', 1234),
    (ctypes.c_uint32, 'unsigned int', 123456),
    (ctypes.c_uint64, 'unsigned long long', 123456789),
    (ctypes.c_float,  'float',    3.14),
    (ctypes.c_double, 'double',   2.718),
)
if os.environ.get("CUDA_PATH"):
    PARAMS += (
        (np.float16,      'half',    0.78),
        (np.complex64, 'cuda::std::complex<float>', 1+2j),
        (np.complex128, 'cuda::std::complex<double>', -3-4j),
        (complex, 'cuda::std::complex<double>',   5-7j),
    )

@pytest.mark.parametrize("python_type, cpp_type, init_value", PARAMS)
def test_launch_scalar_argument(python_type, cpp_type, init_value):
    dev = Device()
    dev.set_current()

    # Prepare pinned host array
    mr = _DefaultPinnedMemorySource()
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
    if os.environ.get("CUDA_PATH"):
        include_path=str(pathlib.Path(os.environ["CUDA_PATH"]) / pathlib.Path("include"))
        code = r"""
        #include <cuda_fp16.h>
        #include <cuda/std/complex>
        """ + code
    else:
        include_path=None
    pro_opts = ProgramOptions(std="c++11", arch=f"sm_{arch}", include_path=include_path)
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
