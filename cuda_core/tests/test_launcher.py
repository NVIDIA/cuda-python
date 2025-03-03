# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest

from cuda.core.experimental import Device, LaunchConfig, Program, launch


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

    with pytest.raises(AssertionError):
        launch(stream, None, config)

    with pytest.raises(AssertionError):
        launch(stream, ker, None)

    launch(stream, config, ker)
