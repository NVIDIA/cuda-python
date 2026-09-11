# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.string cimport memset

from typing import Any

from cuda.core._device import Device
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.cuda_utils import (
    CUDAError,
    cast_to_3_tuple,
    driver,
)

_LAUNCH_CONFIG_ATTRS = (
    'grid',
    'cluster',
    'block',
    'shmem_size',
    'is_cooperative',
    'programmatic_stream_serialization',
    'priority',
)

__all__ = ['LaunchConfig']


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
    grid : tuple | int
        Collection of threads that will execute a kernel function. When cluster
        is not specified, this represents the number of blocks, otherwise
        this represents the number of clusters.
    cluster : tuple | int
        Group of blocks (Thread Block Cluster) that will execute on the same
        GPU Processing Cluster (GPC). Blocks within a cluster have access to
        distributed shared memory and can be explicitly synchronized.
    block : tuple | int
        Group of threads (Thread Block) that will execute on the same
        streaming multiprocessor (SM). Threads within a thread blocks have
        access to shared memory and can be explicitly synchronized.
    shmem_size : int, optional
        Dynamic shared-memory size per thread block in bytes.
        (Default to size 0)
    is_cooperative : bool, optional
        Whether this config can be used to launch a cooperative kernel.
    programmatic_stream_serialization : bool, optional
        Whether to allow programmatic stream serialization (PDL). When True,
        the kernel may overlap with a previous kernel in the same stream that
        signals completion via programmatic means.
    priority : int, optional
        Execution priority of the kernel. Lower numbers represent higher
        priorities. The meaningful range of values is device-specific,
        given by ``[greatestPriority, leastPriority]`` as returned by
        ``cuCtxGetStreamPriorityRange`` (the same range used by
        :attr:`~cuda.core.StreamOptions.priority`); both bounds are 0 on
        a device that does not support multiple stream priorities. A
        nonzero value outside this range raises :class:`ValueError`.
        When omitted (or 0), the launch uses the stream's priority.
    """

    # TODO: expand LaunchConfig to include other attributes
    # Note: attributes are declared in _launch_config.pxd

    def __init__(
        self,
        grid: int | tuple[int, ...] | None = None,
        cluster: int | tuple[int, ...] | None = None,
        block: int | tuple[int, ...] | None = None,
        shmem_size: int | None = None,
        is_cooperative: bool = False,
        programmatic_stream_serialization: bool = False,
        priority: int | None = None,
    ) -> None:
        """Initialize LaunchConfig with validation.

        Parameters
        ----------
        grid : tuple | int, optional
            Grid dimensions (number of blocks or clusters if cluster is specified)
        cluster : tuple | int, optional
            Cluster dimensions (Thread Block Cluster)
        block : tuple | int, optional
            Block dimensions (threads per block)
        shmem_size : int, optional
            Dynamic shared memory size in bytes (default: 0)
        is_cooperative : bool, optional
            Whether to launch as cooperative kernel (default: False)
        programmatic_stream_serialization : bool, optional
            Whether to allow programmatic stream serialization / PDL (default: False)
        priority : int, optional
            Execution priority of the kernel. Lower numbers represent higher
            priorities. The meaningful range of values is device-specific,
            given by ``[greatestPriority, leastPriority]`` as returned by
            ``cuCtxGetStreamPriorityRange`` (the same range used by
            :attr:`~cuda.core.StreamOptions.priority`); both bounds are 0 on
            a device that does not support multiple stream priorities. A
            nonzero value outside this range raises :class:`ValueError`.
            When omitted (or 0), the launch uses the stream's priority.
        """
        # Convert and validate grid and block dimensions
        self.grid = cast_to_3_tuple("LaunchConfig.grid", grid)
        self.block = cast_to_3_tuple("LaunchConfig.block", block)

        # FIXME: Calling Device() strictly speaking is not quite right; we should instead
        # look up the device from stream. We probably need to defer the checks related to
        # device compute capability or attributes.
        # thread block clusters are supported starting H100
        if cluster is not None:
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

        self.is_cooperative = is_cooperative
        self.programmatic_stream_serialization = programmatic_stream_serialization

        # priority=0 is treated the same as an unset priority (see
        # _to_native_launch_config), so only nonzero values need validating
        # against the device's stream priority range.
        cdef int high, low
        cdef cydriver.CUresult res_code
        cdef int prio
        if priority:
            with nogil:
                res_code = cydriver.cuCtxGetStreamPriorityRange(&high, &low)
            if res_code != cydriver.CUresult.CUDA_SUCCESS:
                if res_code == cydriver.CUresult.CUDA_ERROR_INVALID_CONTEXT:
                    raise RuntimeError(
                        "No current CUDA context. Call dev.set_current() before creating a LaunchConfig with a priority."
                    )
                HANDLE_RETURN(res_code)
            prio = priority
            if not (low <= prio <= high):
                raise ValueError(f"{priority=} is out of range {[low, high]}")
            self.priority = prio

        if self.is_cooperative and not Device().properties.cooperative_launch:
            raise CUDAError("cooperative kernels are not supported on this device")

    def _identity(self) -> tuple[Any, ...]:
        return tuple(getattr(self, attr) for attr in _LAUNCH_CONFIG_ATTRS)

    def __repr__(self) -> str:
        """Return string representation of LaunchConfig."""
        parts = ', '.join(f'{attr}={getattr(self, attr)!r}' for attr in _LAUNCH_CONFIG_ATTRS)
        return f"LaunchConfig({parts})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaunchConfig):
            return NotImplemented
        return self._identity() == (<LaunchConfig>other)._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    cdef cydriver.CUlaunchConfig _to_native_launch_config(self):
        cdef cydriver.CUlaunchConfig drv_cfg
        cdef cydriver.CUlaunchAttribute attr
        memset(&drv_cfg, 0, sizeof(drv_cfg))
        self._attrs.resize(0)

        # Handle grid dimensions and cluster configuration
        if self.cluster is not None:
            # Convert grid from cluster units to block units
            drv_cfg.gridDimX = self.grid[0] * self.cluster[0]
            drv_cfg.gridDimY = self.grid[1] * self.cluster[1]
            drv_cfg.gridDimZ = self.grid[2] * self.cluster[2]

            # Set up cluster attribute
            attr.id = cydriver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.cluster
            self._attrs.push_back(attr)
        else:
            drv_cfg.gridDimX, drv_cfg.gridDimY, drv_cfg.gridDimZ = self.grid

        drv_cfg.blockDimX, drv_cfg.blockDimY, drv_cfg.blockDimZ = self.block
        drv_cfg.sharedMemBytes = self.shmem_size

        if self.is_cooperative:
            attr.id = cydriver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            attr.value.cooperative = 1
            self._attrs.push_back(attr)

        if self.programmatic_stream_serialization:
            attr.id = cydriver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attr.value.programmaticStreamSerializationAllowed = 1
            self._attrs.push_back(attr)

        if self.priority:
            attr.id = cydriver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PRIORITY
            attr.value.priority = self.priority
            self._attrs.push_back(attr)

        drv_cfg.numAttrs = self._attrs.size()
        drv_cfg.attrs = self._attrs.data()

        return drv_cfg


# TODO: once all modules are cythonized, this function can be dropped in favor of the cdef method above
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

    if config.is_cooperative:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        attr.value.cooperative = 1
        attrs.append(attr)

    if config.programmatic_stream_serialization:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        attrs.append(attr)

    if config.priority:
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PRIORITY
        attr.value.priority = config.priority
        attrs.append(attr)

    drv_cfg.numAttrs = len(attrs)
    drv_cfg.attrs = attrs

    return drv_cfg
