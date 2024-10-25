from cuda.core.experimental._launcher import LaunchConfig
from cuda.core.experimental._stream import Stream
from cuda.core.experimental._device import Device
from cuda.core.experimental._utils import handle_return
from cuda import cuda

def test_launch_config_init():
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
    try:
        LaunchConfig(grid=0, block=1)
    except ValueError:
        assert True
    else:
        assert False

    try:
        LaunchConfig(grid=(0, 1), block=1)
    except ValueError:
        assert True
    else:
        assert False

    try:
        LaunchConfig(grid=(1, 1, 1), block=0)
    except ValueError:
        assert True
    else:
        assert False

    try:
        LaunchConfig(grid=(1, 1, 1), block=(0, 1))
    except ValueError:
        assert True
    else:
        assert False

def test_launch_config_stream():
    stream = Device().create_stream()
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=stream, shmem_size=0)
    assert config.stream == stream

    try:
        LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream="invalid_stream", shmem_size=0)
    except ValueError:
        assert True
    else:
        assert False

def test_launch_config_shmem_size():
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None, shmem_size=2048)
    assert config.shmem_size == 2048

    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None)
    assert config.shmem_size == 0