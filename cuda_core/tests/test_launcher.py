# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest
import numpy as np
import ctypes
from cuda.core.experimental import Device, LaunchConfig, Stream, launch

def test_launch_config_init(init_cuda):
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None, shmem_size=0)
    assert config.grid == (1, 1, 1)
    assert config.block == (1, 1, 1)
    assert config.stream is None
    assert config.shmem_size == 0

    config = LaunchConfig(grid=(2, 2, 2), block=(2, 2, 2), stream=Device().create_stream(), shmem_size=1024)
    assert config.grid == (2, 2, 2)
    assert config.block == (2, 2, 2)
    assert isinstance(config.stream, Stream)
    assert config.shmem_size == 1024


def test_launch_config_cast_to_3_tuple():
    config = LaunchConfig(grid=1, block=1)
    assert config._cast_to_3_tuple(1) == (1, 1, 1)
    assert config._cast_to_3_tuple((1, 2)) == (1, 2, 1)
    assert config._cast_to_3_tuple((1, 2, 3)) == (1, 2, 3)

    # Edge cases
    assert config._cast_to_3_tuple(999) == (999, 1, 1)
    assert config._cast_to_3_tuple((999, 888)) == (999, 888, 1)
    assert config._cast_to_3_tuple((999, 888, 777)) == (999, 888, 777)


def test_launch_config_invalid_values():
    with pytest.raises(ValueError):
        LaunchConfig(grid=0, block=1)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(0, 1), block=1)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(1, 1, 1), block=0)

    with pytest.raises(ValueError):
        LaunchConfig(grid=(1, 1, 1), block=(0, 1))


def test_launch_config_stream(init_cuda):
    stream = Device().create_stream()
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=stream, shmem_size=0)
    assert config.stream == stream

    with pytest.raises(ValueError):
        LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream="invalid_stream", shmem_size=0)


def test_launch_config_shmem_size():
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None, shmem_size=2048)
    assert config.shmem_size == 2048

    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None)
    assert config.shmem_size == 0

# Example kernel function
def kernel_function(arr):
    # Perform a simple operation for demonstration
    for i in range(len(arr)):
        arr[i] += 1

def test_launch_with_python_scalars(init_cuda):
    """Test launching kernel with Python scalar arguments."""
    stream = Device().create_stream()
    
    config = LaunchConfig(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
        shmem_size=0
    )
    
    # Test various Python scalar types
    launch(kernel_function, config, 10)
    launch(kernel_function, config, 3.14)
    launch(kernel_function, config, True)

def test_launch_with_numpy_scalars(init_cuda):
    """Test launching kernel with NumPy scalar arguments."""
    stream = Device().create_stream()
    
    config = LaunchConfig(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
        shmem_size=0
    )
    
    # Test various NumPy scalar types
    launch(kernel_function, config, np.int32(42))
    launch(kernel_function, config, np.float64(3.14))
    launch(kernel_function, config, np.bool_(True))

def test_launch_with_ctypes_scalars(init_cuda):
    """Test launching kernel with ctypes scalar arguments."""
    stream = Device().create_stream()
    
    config = LaunchConfig(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
        shmem_size=0
    )
    
    # Test various ctypes scalar types
    launch(kernel_function, config, ctypes.c_int(42))
    launch(kernel_function, config, ctypes.c_float(3.14))
    launch(kernel_function, config, ctypes.c_bool(True))

def test_launch_error_cases(init_cuda):
    """Test error cases for launch function."""
    stream = Device().create_stream()
    
    config = LaunchConfig(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
        shmem_size=0
    )
    
    # Invalid kernel type
    with pytest.raises(ValueError):
        launch("not a kernel", config, 10)
    
    # None stream
    with pytest.raises(Exception):
        invalid_config = LaunchConfig(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            stream=None,
            shmem_size=0
        )
        launch(kernel_function, invalid_config, 10)

def test_launch_config_validation(init_cuda):
    """Test launch configuration validation."""
    stream = Device().create_stream()
    
    # Test with valid grid and block dimensions
    config = LaunchConfig(
        grid=(1, 2, 3),
        block=(4, 5, 6),
        stream=stream,
        shmem_size=0
    )
    launch(kernel_function, config, 10)
    
    # Test with alternate valid configurations
    config_alt = LaunchConfig(
        grid=(10, 1, 1),
        block=(256, 1, 1),
        stream=stream,
        shmem_size=128
    )
    launch(kernel_function, config_alt, 42)