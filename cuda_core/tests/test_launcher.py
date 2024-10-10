from cuda.core._launcher import LaunchConfig

def test_launch_initialization():
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), stream=None, shmem_size=0)
    
    assert config.grid == (1, 1, 1)
    assert config.block == (1, 1, 1)
    assert config.shmem_size == 0