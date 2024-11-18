# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental import Device, Stream, LaunchConfig
import pytest

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
