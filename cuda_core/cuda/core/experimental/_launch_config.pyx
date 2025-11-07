# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from cuda.core.experimental._device import Device
from cuda.core.experimental._utils.cuda_utils import (
    CUDAError,
    cast_to_3_tuple,
    driver,
    get_binding_version,
    handle_return,
)

# TODO: revisit this treatment for py313t builds
cdef bint _inited = False
cdef bint _use_ex = False


cdef void _lazy_init() except *:
    """Initialize module-level globals for driver version checks."""
    global _inited, _use_ex
    if _inited:
        return

    cdef tuple _py_major_minor
    cdef int _driver_ver

    # binding availability depends on cuda-python version
    _py_major_minor = get_binding_version()
    _driver_ver = handle_return(driver.cuDriverGetVersion())
    _use_ex = (_driver_ver >= 11080) and (_py_major_minor >= (11, 8))
    _inited = True


cdef class LaunchConfig:
    """Customizable launch options.

    Note
    ----
    When cluster is specified, the grid parameter represents the number of
    clusters (not blocks). The hierarchy is: grid (clusters) -> cluster (blocks) ->
    block (threads). Each dimension in grid specifies clusters in the grid, each dimension in
    cluster specifies blocks per cluster, and each dimension in block specifies
    threads per block.

    Attributes
    ----------
    grid : Union[tuple, int]
        Collection of threads that will execute a kernel function. When cluster
        is not specified, this represents the number of blocks, otherwise
        this represents the number of clusters.
    cluster : Union[tuple, int]
        Group of blocks (Thread Block Cluster) that will execute on the same
        GPU Processing Cluster (GPC). Blocks within a cluster have access to
        distributed shared memory and can be explicitly synchronized.
    block : Union[tuple, int]
        Group of threads (Thread Block) that will execute on the same
        streaming multiprocessor (SM). Threads within a thread blocks have
        access to shared memory and can be explicitly synchronized.
    shmem_size : int, optional
        Dynamic shared-memory size per thread block in bytes.
        (Default to size 0)
    cooperative_launch : bool, optional
        Whether this config can be used to launch a cooperative kernel.
    """

    # TODO: expand LaunchConfig to include other attributes
    # Note: attributes are declared in _launch_config.pxd

    def __init__(self, grid=None, cluster=None, block=None,
                 shmem_size=None, cooperative_launch=False):
        """Initialize LaunchConfig with validation.

        Parameters
        ----------
        grid : Union[tuple, int], optional
            Grid dimensions (number of blocks or clusters if cluster is specified)
        cluster : Union[tuple, int], optional
            Cluster dimensions (Thread Block Cluster)
        block : Union[tuple, int], optional
            Block dimensions (threads per block)
        shmem_size : int, optional
            Dynamic shared memory size in bytes (default: 0)
        cooperative_launch : bool, optional
            Whether to launch as cooperative kernel (default: False)
        """
        _lazy_init()

        # Convert and validate grid and block dimensions
        self.grid = cast_to_3_tuple("LaunchConfig.grid", grid)
        self.block = cast_to_3_tuple("LaunchConfig.block", block)

        # FIXME: Calling Device() strictly speaking is not quite right; we should instead
        # look up the device from stream. We probably need to defer the checks related to
        # device compute capability or attributes.
        # thread block clusters are supported starting H100
        if cluster is not None:
            if not _use_ex:
                err, drvers = driver.cuDriverGetVersion()
                drvers_fmt = f" (got driver version {drvers})" if err == driver.CUresult.CUDA_SUCCESS else ""
                raise CUDAError(f"thread block clusters require cuda.bindings & driver 11.8+{drvers_fmt}")
            cc = Device().compute_capability
            if cc < (9, 0):
                raise CUDAError(
                    f"thread block clusters are not supported on devices with compute capability < 9.0 (got {cc})"
                )
            self.cluster = cast_to_3_tuple("LaunchConfig.cluster", cluster)
        else:
            self.cluster = None

        # Handle shmem_size default
        if shmem_size is None:
            self.shmem_size = 0
        else:
            self.shmem_size = shmem_size

        # Handle cooperative_launch
        if cooperative_launch is None:
            self.cooperative_launch = False
        else:
            self.cooperative_launch = cooperative_launch

        # Validate cooperative launch support
        if self.cooperative_launch and not Device().properties.cooperative_launch:
            raise CUDAError("cooperative kernels are not supported on this device")

    def __repr__(self):
        """Return string representation of LaunchConfig."""
        return (f"LaunchConfig(grid={self.grid}, cluster={self.cluster}, "
                f"block={self.block}, shmem_size={self.shmem_size}, "
                f"cooperative_launch={self.cooperative_launch})")


cpdef object _to_native_launch_config(LaunchConfig config):
    """Convert LaunchConfig to native driver CUlaunchConfig.

    Parameters
    ----------
    config : LaunchConfig
        High-level launch configuration

    Returns
    -------
    driver.CUlaunchConfig
        Native CUDA driver launch configuration
    """
    _lazy_init()
    
    cdef object drv_cfg = driver.CUlaunchConfig()
    cdef list attrs
    cdef object attr
    cdef object dim
    cdef tuple grid_blocks
    
    # Handle grid dimensions and cluster configuration
    if config.cluster is not None:
        # Convert grid from cluster units to block units
        grid_blocks = (
            config.grid[0] * config.cluster[0],
            config.grid[1] * config.cluster[1],
            config.grid[2] * config.cluster[2],
        )
        drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = grid_blocks

        # Set up cluster attribute
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dim = attr.value.clusterDim
        dim.x, dim.y, dim.z = config.cluster
        attrs = [attr]
    else:
        drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = config.grid
        attrs = []

    drv_cfg.blockDimX, drv_cfg.blockDimY, drv_cfg.blockDimZ = config.block
    drv_cfg.sharedMemBytes = config.shmem_size

    if config.cooperative_launch:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        attr.value.cooperative = 1
        attrs.append(attr)

    drv_cfg.numAttrs = len(attrs)
    drv_cfg.attrs = attrs

    return drv_cfg
