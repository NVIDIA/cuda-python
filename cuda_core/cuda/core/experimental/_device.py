# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional, Union

from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import GraphBuilder
from cuda.core.experimental._memory import Buffer, DeviceMemoryResource, MemoryResource, _SynchronousMemoryResource
from cuda.core.experimental._stream import IsStreamT, Stream, StreamOptions, default_stream
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    ComputeCapability,
    CUDAError,
    _check_driver_error,
    driver,
    handle_return,
    runtime,
)

_tls = threading.local()
_lock = threading.Lock()
_is_cuInit = False


class DeviceProperties:
    """
    A class to query various attributes of a CUDA device.

    Attributes are read-only and provide information about the device.
    """

    def __new__(self, *args, **kwargs):
        raise RuntimeError("DeviceProperties cannot be instantiated directly. Please use Device APIs.")

    __slots__ = ("_handle", "_cache")

    @classmethod
    def _init(cls, handle):
        self = super().__new__(cls)
        self._handle = handle
        self._cache = {}
        return self

    def _get_attribute(self, attr):
        """Retrieve the attribute value directly from the driver."""
        return handle_return(driver.cuDeviceGetAttribute(attr, self._handle))

    def _get_cached_attribute(self, attr):
        """Retrieve the attribute value, using cache if applicable."""
        if attr not in self._cache:
            self._cache[attr] = self._get_attribute(attr)
        return self._cache[attr]

    @property
    def max_threads_per_block(self) -> int:
        """
        Maximum number of threads per block.

        Returns:
            int: Maximum number of threads per block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

    @property
    def max_block_dim_x(self) -> int:
        """
        Maximum block dimension X.

        Returns:
            int: Maximum block dimension X.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)

    @property
    def max_block_dim_y(self) -> int:
        """
        Maximum block dimension Y.

        Returns:
            int: Maximum block dimension Y.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)

    @property
    def max_block_dim_z(self) -> int:
        """
        Maximum block dimension Z.

        Returns:
            int: Maximum block dimension Z.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)

    @property
    def max_grid_dim_x(self) -> int:
        """
        Maximum grid dimension X.

        Returns:
            int: Maximum grid dimension X.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)

    @property
    def max_grid_dim_y(self) -> int:
        """
        Maximum grid dimension Y.

        Returns:
            int: Maximum grid dimension Y.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

    @property
    def max_grid_dim_z(self) -> int:
        """
        Maximum grid dimension Z.

        Returns:
            int: Maximum grid dimension Z.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)

    @property
    def max_shared_memory_per_block(self) -> int:
        """
        Maximum shared memory available per block in bytes.

        Returns:
            int: Maximum shared memory available per block in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    @property
    def total_constant_memory(self) -> int:
        """
        Memory available on device for constant variables in a CUDA C kernel in bytes.

        Returns:
            int: Memory available on device for constant variables in a CUDA C kernel in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

    @property
    def warp_size(self) -> int:
        """
        Warp size in threads.

        Returns:
            int: Warp size in threads.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE)

    @property
    def max_pitch(self) -> int:
        """
        Maximum pitch in bytes allowed by memory copies.

        Returns:
            int: Maximum pitch in bytes allowed by memory copies.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH)

    @property
    def maximum_texture1d_width(self) -> int:
        """
        Maximum 1D texture width.

        Returns:
            int: Maximum 1D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)

    @property
    def maximum_texture1d_linear_width(self) -> int:
        """
        Maximum width for a 1D texture bound to linear memory.

        Returns:
            int: Maximum width for a 1D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)

    @property
    def maximum_texture1d_mipmapped_width(self) -> int:
        """
        Maximum mipmapped 1D texture width.

        Returns:
            int: Maximum mipmapped 1D texture width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_width(self) -> int:
        """
        Maximum 2D texture width.

        Returns:
            int: Maximum 2D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)

    @property
    def maximum_texture2d_height(self) -> int:
        """
        Maximum 2D texture height.

        Returns:
            int: Maximum 2D texture height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)

    @property
    def maximum_texture2d_linear_width(self) -> int:
        """
        Maximum width for a 2D texture bound to linear memory.

        Returns:
            int: Maximum width for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)

    @property
    def maximum_texture2d_linear_height(self) -> int:
        """
        Maximum height for a 2D texture bound to linear memory.

        Returns:
            int: Maximum height for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)

    @property
    def maximum_texture2d_linear_pitch(self) -> int:
        """
        Maximum pitch in bytes for a 2D texture bound to linear memory.

        Returns:
            int: Maximum pitch in bytes for a 2D texture bound to linear memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)

    @property
    def maximum_texture2d_mipmapped_width(self) -> int:
        """
        Maximum mipmapped 2D texture width.

        Returns:
            int: Maximum mipmapped 2D texture width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
        )

    @property
    def maximum_texture2d_mipmapped_height(self) -> int:
        """
        Maximum mipmapped 2D texture height.

        Returns:
            int: Maximum mipmapped 2D texture height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
        )

    @property
    def maximum_texture3d_width(self) -> int:
        """
        Maximum 3D texture width.

        Returns:
            int: Maximum 3D texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)

    @property
    def maximum_texture3d_height(self) -> int:
        """
        Maximum 3D texture height.

        Returns:
            int: Maximum 3D texture height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)

    @property
    def maximum_texture3d_depth(self) -> int:
        """
        Maximum 3D texture depth.

        Returns:
            int: Maximum 3D texture depth.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)

    @property
    def maximum_texture3d_width_alternate(self) -> int:
        """
        Alternate maximum 3D texture width.

        Returns:
            int: Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
        )

    @property
    def maximum_texture3d_height_alternate(self) -> int:
        """
        Alternate maximum 3D texture height.

        Returns:
            int: Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
        )

    @property
    def maximum_texture3d_depth_alternate(self) -> int:
        """
        Alternate maximum 3D texture depth.

        Returns:
            int: Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
        )

    @property
    def maximum_texturecubemap_width(self) -> int:
        """
        Maximum cubemap texture width or height.

        Returns:
            int: Maximum cubemap texture width or height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)

    @property
    def maximum_texture1d_layered_width(self) -> int:
        """
        Maximum 1D layered texture width.

        Returns:
            int: Maximum 1D layered texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)

    @property
    def maximum_texture1d_layered_layers(self) -> int:
        """
        Maximum layers in a 1D layered texture.

        Returns:
            int: Maximum layers in a 1D layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        )

    @property
    def maximum_texture2d_layered_width(self) -> int:
        """
        Maximum 2D layered texture width.

        Returns:
            int: Maximum 2D layered texture width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)

    @property
    def maximum_texture2d_layered_height(self) -> int:
        """
        Maximum 2D layered texture height.

        Returns:
            int: Maximum 2D layered texture height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_texture2d_layered_layers(self) -> int:
        """
        Maximum layers in a 2D layered texture.

        Returns:
            int: Maximum layers in a 2D layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        )

    @property
    def maximum_texturecubemap_layered_width(self) -> int:
        """
        Maximum cubemap layered texture width or height.

        Returns:
            int: Maximum cubemap layered texture width or height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_texturecubemap_layered_layers(self) -> int:
        """
        Maximum layers in a cubemap layered texture.

        Returns:
            int: Maximum layers in a cubemap layered texture.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
        )

    @property
    def maximum_surface1d_width(self) -> int:
        """
        Maximum 1D surface width.

        Returns:
            int: Maximum 1D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)

    @property
    def maximum_surface2d_width(self) -> int:
        """
        Maximum 2D surface width.

        Returns:
            int: Maximum 2D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)

    @property
    def maximum_surface2d_height(self) -> int:
        """
        Maximum 2D surface height.

        Returns:
            int: Maximum 2D surface height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)

    @property
    def maximum_surface3d_width(self) -> int:
        """
        Maximum 3D surface width.

        Returns:
            int: Maximum 3D surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)

    @property
    def maximum_surface3d_height(self) -> int:
        """
        Maximum 3D surface height.

        Returns:
            int: Maximum 3D surface height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)

    @property
    def maximum_surface3d_depth(self) -> int:
        """
        Maximum 3D surface depth.

        Returns:
            int: Maximum 3D surface depth.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)

    @property
    def maximum_surface1d_layered_width(self) -> int:
        """
        Maximum 1D layered surface width.

        Returns:
            int: Maximum 1D layered surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)

    @property
    def maximum_surface1d_layered_layers(self) -> int:
        """
        Maximum layers in a 1D layered surface.

        Returns:
            int: Maximum layers in a 1D layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
        )

    @property
    def maximum_surface2d_layered_width(self) -> int:
        """
        Maximum 2D layered surface width.

        Returns:
            int: Maximum 2D layered surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)

    @property
    def maximum_surface2d_layered_height(self) -> int:
        """
        Maximum 2D layered surface height.

        Returns:
            int: Maximum 2D layered surface height.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
        )

    @property
    def maximum_surface2d_layered_layers(self) -> int:
        """
        Maximum layers in a 2D layered surface.

        Returns:
            int: Maximum layers in a 2D layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS
        )

    @property
    def maximum_surfacecubemap_width(self) -> int:
        """
        Maximum cubemap surface width.

        Returns:
            int: Maximum cubemap surface width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)

    @property
    def maximum_surfacecubemap_layered_width(self) -> int:
        """
        Maximum cubemap layered surface width.

        Returns:
            int: Maximum cubemap layered surface width.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
        )

    @property
    def maximum_surfacecubemap_layered_layers(self) -> int:
        """
        Maximum layers in a cubemap layered surface.

        Returns:
            int: Maximum layers in a cubemap layered surface.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
        )

    @property
    def max_registers_per_block(self) -> int:
        """
        Maximum number of 32-bit registers available to a thread block.

        Returns:
            int: Maximum number of 32-bit registers available to a thread block.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)

    @property
    def clock_rate(self) -> int:
        """
        Typical clock frequency in kilohertz.

        Returns:
            int: Typical clock frequency in kilohertz.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)

    @property
    def texture_alignment(self) -> int:
        """
        Alignment requirement for textures.

        Returns:
            int: Alignment requirement for textures.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)

    @property
    def texture_pitch_alignment(self) -> int:
        """
        Pitch alignment requirement for textures.

        Returns:
            int: Pitch alignment requirement for textures.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)

    @property
    def gpu_overlap(self) -> bool:
        """
        Device can possibly copy memory and execute a kernel concurrently.

        Returns:
            bool: Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead
            async_engine_count.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP))

    @property
    def multiprocessor_count(self) -> int:
        """
        Number of multiprocessors on device.

        Returns:
            int: Number of multiprocessors on device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    @property
    def kernel_exec_timeout(self) -> bool:
        """
        Specifies whether there is a run time limit on kernels.

        Returns:
            bool: Specifies whether there is a run time limit on kernels.
        """
        return bool(self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT))

    @property
    def integrated(self) -> bool:
        """
        Device is integrated with host memory.

        Returns:
            bool: Device is integrated with host memory.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED))

    @property
    def can_map_host_memory(self) -> bool:
        """
        Device can map host memory into CUDA address space.

        Returns:
            bool: Device can map host memory into CUDA address space.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY))

    @property
    def compute_mode(self) -> int:
        """
        Compute mode (See CUcomputemode for details).

        Returns:
            int: Compute mode (See CUcomputemode for details).
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)

    @property
    def concurrent_kernels(self) -> bool:
        """
        Device can possibly execute multiple kernels concurrently.

        Returns:
            bool: Device can possibly execute multiple kernels concurrently.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

    @property
    def ecc_enabled(self) -> bool:
        """
        Device has ECC support enabled.

        Returns:
            bool: Device has ECC support enabled.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED))

    @property
    def pci_bus_id(self) -> int:
        """
        PCI bus ID of the device.

        Returns:
            int: PCI bus ID of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)

    @property
    def pci_device_id(self) -> int:
        """
        PCI device ID of the device.

        Returns:
            int: PCI device ID of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)

    @property
    def pci_domain_id(self) -> int:
        """
        PCI domain ID of the device.

        Returns:
            int: PCI domain ID of the device.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)

    @property
    def tcc_driver(self) -> bool:
        """
        Device is using TCC driver model.

        Returns:
            bool: Device is using TCC driver model.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER))

    @property
    def memory_clock_rate(self) -> int:
        """
        Peak memory clock frequency in kilohertz.

        Returns:
            int: Peak memory clock frequency in kilohertz.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    @property
    def global_memory_bus_width(self) -> int:
        """
        Global memory bus width in bits.

        Returns:
            int: Global memory bus width in bits.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

    @property
    def l2_cache_size(self) -> int:
        """
        Size of L2 cache in bytes.

        Returns:
            int: Size of L2 cache in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

    @property
    def max_threads_per_multiprocessor(self) -> int:
        """
        Maximum resident threads per multiprocessor.

        Returns:
            int: Maximum resident threads per multiprocessor.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

    @property
    def unified_addressing(self) -> bool:
        """
        Device shares a unified address space with the host.

        Returns:
            bool: Device shares a unified address space with the host.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING))

    @property
    def compute_capability_major(self) -> int:
        """
        Major compute capability version number.

        Returns:
            int: Major compute capability version number.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)

    @property
    def compute_capability_minor(self) -> int:
        """
        Minor compute capability version number.

        Returns:
            int: Minor compute capability version number.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

    @property
    def global_l1_cache_supported(self) -> bool:
        """
        Device supports caching globals in L1.

        Returns:
            bool: Device supports caching globals in L1.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED))

    @property
    def local_l1_cache_supported(self) -> bool:
        """
        Device supports caching locals in L1.

        Returns:
            bool: Device supports caching locals in L1.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED))

    @property
    def max_shared_memory_per_multiprocessor(self) -> int:
        """
        Maximum shared memory available per multiprocessor in bytes.

        Returns:
            int: Maximum shared memory available per multiprocessor in bytes.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )

    @property
    def max_registers_per_multiprocessor(self) -> int:
        """
        Maximum number of 32-bit registers available per multiprocessor.

        Returns:
            int: Maximum number of 32-bit registers available per multiprocessor.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        )

    @property
    def managed_memory(self) -> bool:
        """
        Device can allocate managed memory on this system.

        Returns:
            bool: Device can allocate managed memory on this system.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY))

    @property
    def multi_gpu_board(self) -> bool:
        """
        Device is on a multi-GPU board.

        Returns:
            bool: Device is on a multi-GPU board.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD))

    @property
    def multi_gpu_board_group_id(self) -> int:
        """
        Unique id for a group of devices on the same multi-GPU board.

        Returns:
            int: Unique id for a group of devices on the same multi-GPU board.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)

    @property
    def host_native_atomic_supported(self) -> bool:
        """
        Link between the device and the host supports all native atomic operations.

        Returns:
            bool: Link between the device and the host supports all native atomic operations.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)
        )

    @property
    def single_to_double_precision_perf_ratio(self) -> int:
        """
        Ratio of single precision performance (in floating-point operations per second) to double precision performance.

        Returns:
            int: Ratio of single precision performance (in floating-point operations per second) to double
            precision performance.
        """
        return self._get_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)

    @property
    def pageable_memory_access(self) -> bool:
        """
        Device supports coherently accessing pageable memory without calling cudaHostRegister on it.

        Returns:
            bool: Device supports coherently accessing pageable memory without calling cudaHostRegister on it.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS))

    @property
    def concurrent_managed_access(self) -> bool:
        """
        Device can coherently access managed memory concurrently with the CPU.

        Returns:
            bool: Device can coherently access managed memory concurrently with the CPU.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS))

    @property
    def compute_preemption_supported(self) -> bool:
        """
        Device supports compute preemption.

        Returns:
            bool: Device supports compute preemption.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
        )

    @property
    def can_use_host_pointer_for_registered_mem(self) -> bool:
        """
        Device can access host registered memory at the same virtual address as the CPU.

        Returns:
            bool: Device can access host registered memory at the same virtual address as the CPU.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
            )
        )

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def cooperative_launch(self) -> bool:
        """
        Device supports launching cooperative kernels via cuLaunchCooperativeKernel.

        Returns:
            bool: Device supports launching cooperative kernels via cuLaunchCooperativeKernel.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH))

    # TODO: A few attrs are missing here (NVIDIA/cuda-python#675)

    @property
    def max_shared_memory_per_block_optin(self) -> int:
        """
        Maximum optin shared memory per block.

        Returns:
            int: Maximum optin shared memory per block.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )

    @property
    def pageable_memory_access_uses_host_page_tables(self) -> bool:
        """
        Device accesses pageable memory via the host's page tables.

        Returns:
            bool: Device accesses pageable memory via the host's page tables.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
            )
        )

    @property
    def direct_managed_mem_access_from_host(self) -> bool:
        """
        The host can directly access managed memory on the device without migration.

        Returns:
            bool: The host can directly access managed memory on the device without migration.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
            )
        )

    @property
    def virtual_memory_management_supported(self) -> bool:
        """
        Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap and related APIs.

        Returns:
            bool: Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate,
            cuMemMap and related APIs.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def handle_type_posix_file_descriptor_supported(self) -> bool:
        """
        Device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle,
        if requested via cuMemCreate.

        Returns:
            bool: Device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle,
            if requested via cuMemCreate.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
            )
        )

    @property
    def handle_type_win32_handle_supported(self) -> bool:
        """
        Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle,
        if requested via cuMemCreate.

        Returns:
            bool: Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle,
            if requested via cuMemCreate.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED)
        )

    @property
    def handle_type_win32_kmt_handle_supported(self) -> bool:
        """
        Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle,
        if requested via cuMemCreate.

        Returns:
            bool: Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle,
            if requested via cuMemCreate.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
            )
        )

    @property
    def max_blocks_per_multiprocessor(self) -> int:
        """
        Maximum number of blocks per multiprocessor.

        Returns:
            int: Maximum number of blocks per multiprocessor.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)

    @property
    def generic_compression_supported(self) -> bool:
        """
        Device supports compression of memory.

        Returns:
            bool: Device supports compression of memory.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED)
        )

    @property
    def max_persisting_l2_cache_size(self) -> int:
        """
        Maximum L2 persisting lines capacity setting in bytes.

        Returns:
            int: Maximum L2 persisting lines capacity setting in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)

    @property
    def max_access_policy_window_size(self) -> int:
        """
        Maximum value of CUaccessPolicyWindow.num_bytes.

        Returns:
            int: Maximum value of CUaccessPolicyWindow.num_bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)

    @property
    def gpu_direct_rdma_with_cuda_vmm_supported(self) -> bool:
        """
        Device supports specifying the GPUDirect RDMA flag with cuMemCreate.

        Returns:
            bool: Device supports specifying the GPUDirect RDMA flag with cuMemCreate.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
            )
        )

    @property
    def reserved_shared_memory_per_block(self) -> int:
        """
        Shared memory reserved by CUDA driver per block in bytes.

        Returns:
            int: Shared memory reserved by CUDA driver per block in bytes.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
        )

    @property
    def sparse_cuda_array_supported(self) -> bool:
        """
        Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.

        Returns:
            bool: Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED)
        )

    @property
    def read_only_host_register_supported(self) -> bool:
        """
        Whether device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY.

        Returns:
            bool: True if device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register
            memory that must be mapped as read-only to the GPU, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)
        )

    @property
    def memory_pools_supported(self) -> bool:
        """
        Device supports using the cuMemAllocAsync and cuMemPool family of APIs.

        Returns:
            bool: Device supports using the cuMemAllocAsync and cuMemPool family of APIs.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED))

    @property
    def gpu_direct_rdma_supported(self) -> bool:
        """
        Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
        https://docs.nvidia.com/cuda/gpudirect-rdma for more information).

        Returns:
            bool: Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
            https://docs.nvidia.com/cuda/gpudirect-rdma for more information).
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED))

    @property
    def gpu_direct_rdma_flush_writes_options(self) -> int:
        """
        The returned attribute shall be interpreted as a bitmask, where the individual bits are described by
        the CUflushGPUDirectRDMAWritesOptions enum.

        Returns:
            int: The returned attribute shall be interpreted as a bitmask, where the individual bits are described by
            the CUflushGPUDirectRDMAWritesOptions enum.
        """
        return self._get_cached_attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
        )

    @property
    def gpu_direct_rdma_writes_ordering(self) -> int:
        """
        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated
        by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical values returned here.

        Returns:
            int: GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated
            by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING)

    @property
    def mempool_supported_handle_types(self) -> int:
        """
        Handle types supported with mempool based IPC.

        Returns:
            int: Handle types supported with mempool based IPC.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES)

    @property
    def deferred_mapping_cuda_array_supported(self) -> bool:
        """
        Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.

        Returns:
            bool: Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
            )
        )

    @property
    def numa_config(self) -> int:
        """
        NUMA configuration of a device: value is of type CUdeviceNumaConfig enum.

        Returns:
            int: NUMA configuration of a device: value is of type CUdeviceNumaConfig enum.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_CONFIG)

    @property
    def numa_id(self) -> int:
        """
        NUMA node ID of the GPU memory.

        Returns:
            int: NUMA node ID of the GPU memory.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_ID)

    @property
    def multicast_supported(self) -> bool:
        """
        Device supports switch multicast and reduction operations.

        Returns:
            bool: Device supports switch multicast and reduction operations.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED))

    @property
    def surface_alignment(self) -> int:
        """
        Surface alignment requirement in bytes.

        Returns:
            int: Surface alignment requirement in bytes.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT)

    @property
    def async_engine_count(self) -> int:
        """
        Number of asynchronous engines.

        Returns:
            int: Number of asynchronous engines.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)

    @property
    def can_tex2d_gather(self) -> bool:
        """
        Whether device supports 2D texture gather operations.

        Returns:
            bool: True if device supports 2D texture gather operations, False if not.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER))

    @property
    def maximum_texture2d_gather_width(self) -> int:
        """
        Maximum 2D texture gather width.

        Returns:
            int: Maximum 2D texture gather width.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH)

    @property
    def maximum_texture2d_gather_height(self) -> int:
        """
        Maximum 2D texture gather height.

        Returns:
            int: Maximum 2D texture gather height.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT)

    @property
    def stream_priorities_supported(self) -> bool:
        """
        Whether device supports stream priorities.

        Returns:
            bool: True if device supports stream priorities, False if not.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED)
        )

    @property
    def cooperative_multi_device_launch(self) -> bool:
        """
        Deprecated, cuLaunchCooperativeKernelMultiDevice is deprecated.

        Returns:
            bool: Deprecated, cuLaunchCooperativeKernelMultiDevice is deprecated.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH)
        )

    @property
    def can_flush_remote_writes(self) -> bool:
        """
        The CU_STREAM_WAIT_VALUE_FLUSH flag and the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the
        device. See Stream Memory Operations for additional details.

        Returns:
            bool: The CU_STREAM_WAIT_VALUE_FLUSH flag and the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported
            on the device. See Stream Memory Operations for additional details.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES))

    @property
    def host_register_supported(self) -> bool:
        """
        Device supports host memory registration via cudaHostRegister.

        Returns:
            bool: Device supports host memory registration via cudaHostRegister.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED))

    @property
    def virtual_address_management_supported(self) -> bool:
        """
        Deprecated, Use virtual_memory_management_supported.

        Returns:
            bool: Deprecated, Use virtual_memory_management_supported.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def timeline_semaphore_interop_supported(self) -> bool:
        """
        External timeline semaphore interop is supported on the device.

        Returns:
            bool: External timeline semaphore interop is supported on the device.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
            )
        )

    @property
    def cluster_launch(self) -> bool:
        """
        Indicates device supports cluster launch.

        Returns:
            bool: Indicates device supports cluster launch.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH))

    @property
    def can_use_64_bit_stream_mem_ops(self) -> bool:
        """
        64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.

        Returns:
            bool: 64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS)
        )

    @property
    def can_use_stream_wait_value_nor(self) -> bool:
        """
        CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.

        Returns:
            bool: CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR)
        )

    @property
    def dma_buf_supported(self) -> bool:
        """
        Device supports buffer sharing with dma_buf mechanism.

        Returns:
            bool: Device supports buffer sharing with dma_buf mechanism.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED))

    @property
    def ipc_event_supported(self) -> bool:
        """
        Device supports IPC Events.

        Returns:
            bool: Device supports IPC Events.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED))

    @property
    def mem_sync_domain_count(self) -> int:
        """
        Number of memory domains the device supports.

        Returns:
            int: Number of memory domains the device supports.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT)

    @property
    def tensor_map_access_supported(self) -> bool:
        """
        Device supports accessing memory using Tensor Map.

        Returns:
            bool: Device supports accessing memory using Tensor Map.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED)
        )

    @property
    def handle_type_fabric_supported(self) -> bool:
        """
        Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested
        with cuMemCreate().

        Returns:
            bool: Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested
            with cuMemCreate().
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED)
        )

    @property
    def unified_function_pointers(self) -> bool:
        """
        Device supports unified function pointers.

        Returns:
            bool: Device supports unified function pointers.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS))

    @property
    def mps_enabled(self) -> bool:
        """
        Indicates if contexts created on this device will be shared via MPS.

        Returns:
            bool: Indicates if contexts created on this device will be shared via MPS.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MPS_ENABLED))

    @property
    def host_numa_id(self) -> int:
        """
        NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.

        Returns:
            int: NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)

    @property
    def d3d12_cig_supported(self) -> bool:
        """
        Device supports CIG with D3D12.

        Returns:
            bool: Device supports CIG with D3D12.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED))

    @property
    def mem_decompress_algorithm_mask(self) -> int:
        """
        The returned valued shall be interpreted as a bitmask, where the individual bits are described by
        the CUmemDecompressAlgorithm enum.

        Returns:
            int: The returned valued shall be interpreted as a bitmask, where the individual bits are described by
            the CUmemDecompressAlgorithm enum.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK)

    @property
    def mem_decompress_maximum_length(self) -> int:
        """
        The returned valued is the maximum length in bytes of a single decompress operation that is allowed.

        Returns:
            int: The returned valued is the maximum length in bytes of a single decompress operation that is allowed.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH)

    @property
    def vulkan_cig_supported(self) -> bool:
        """
        Device supports CIG with Vulkan.

        Returns:
            bool: Device supports CIG with Vulkan.
        """
        return bool(self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED))

    @property
    def gpu_pci_device_id(self) -> int:
        """
        The combined 16-bit PCI device ID and 16-bit PCI vendor ID.

        Returns:
            int: The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID)

    @property
    def gpu_pci_subsystem_id(self) -> int:
        """
        The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.

        Returns:
            int: The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
        """
        return self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID)

    @property
    def host_numa_virtual_memory_management_supported(self) -> bool:
        """
        Device supports HOST_NUMA location with the virtual memory management APIs like cuMemCreate, cuMemMap and
        related APIs.

        Returns:
            bool: Device supports HOST_NUMA location with the virtual memory management APIs like cuMemCreate, cuMemMap
            and related APIs.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def host_numa_memory_pools_supported(self) -> bool:
        """
        Device supports HOST_NUMA location with the cuMemAllocAsync and cuMemPool family of APIs.

        Returns:
            bool: Device supports HOST_NUMA location with the cuMemAllocAsync and cuMemPool family of APIs.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED)
        )

    @property
    def host_numa_multinode_ipc_supported(self) -> bool:
        """
        Device supports HOST_NUMA location IPC between nodes in a multi-node system.

        Returns:
            bool: Device supports HOST_NUMA location IPC between nodes in a multi-node system.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED)
        )

    @property
    def host_memory_pools_supported(self) -> bool:
        """
        Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs.

        Returns:
            bool: Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED)
        )

    @property
    def host_virtual_memory_management_supported(self) -> bool:
        """
        Device supports HOST location with the virtual memory management APIs like cuMemCreate, cuMemMap and related
        APIs.

        Returns:
            bool: Device supports HOST location with the virtual memory management APIs like cuMemCreate, cuMemMap and
            related APIs.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )
        )

    @property
    def host_alloc_dma_buf_supported(self) -> bool:
        """
        Device supports page-locked host memory buffer sharing with dma_buf mechanism.

        Returns:
            bool: Device supports page-locked host memory buffer sharing with dma_buf mechanism.
        """
        return bool(
            self._get_cached_attribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED)
        )

    @property
    def only_partial_host_native_atomic_supported(self) -> bool:
        """
        Link between the device and the host supports only some native atomic operations.

        Returns:
            bool: Link between the device and the host supports only some native atomic operations.
        """
        return bool(
            self._get_cached_attribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED
            )
        )


_SUCCESS = driver.CUresult.CUDA_SUCCESS
_INVALID_CTX = driver.CUresult.CUDA_ERROR_INVALID_CONTEXT


class Device:
    """Represent a GPU and act as an entry point for cuda.core features.

    This is a singleton object that helps ensure interoperability
    across multiple libraries imported in the process to both see
    and use the same GPU device.

    While acting as the entry point, many other CUDA resources can be
    allocated such as streams and buffers. Any :obj:`~_context.Context` dependent
    resource created through this device, will continue to refer to
    this device's context.

    Newly returned :obj:`~_device.Device` objects are thread-local singletons
    for a specified device.

    Note
    ----
    Will not initialize the GPU.

    Parameters
    ----------
    device_id : int, optional
        Device ordinal to return a :obj:`~_device.Device` object for.
        Default value of `None` return the currently used device.

    """

    __slots__ = ("_id", "_mr", "_has_inited", "_properties")

    def __new__(cls, device_id: Optional[int] = None):
        global _is_cuInit
        if _is_cuInit is False:
            with _lock:
                handle_return(driver.cuInit(0))
                _is_cuInit = True

        # important: creating a Device instance does not initialize the GPU!
        if device_id is None:
            err, dev = driver.cuCtxGetDevice()
            if err == _SUCCESS:
                device_id = int(dev)
            elif err == _INVALID_CTX:
                ctx = handle_return(driver.cuCtxGetCurrent())
                assert int(ctx) == 0
                device_id = 0  # cudart behavior
            else:
                _check_driver_error(err)
        elif device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {device_id}")

        # ensure Device is singleton
        try:
            devices = _tls.devices
        except AttributeError:
            total = handle_return(driver.cuDeviceGetCount())
            devices = _tls.devices = []
            for dev_id in range(total):
                dev = super().__new__(cls)
                dev._id = dev_id
                # If the device is in TCC mode, or does not support memory pools for some other reason,
                # use the SynchronousMemoryResource which does not use memory pools.
                if (
                    handle_return(
                        driver.cuDeviceGetAttribute(
                            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev_id
                        )
                    )
                ) == 1:
                    dev._mr = DeviceMemoryResource(dev_id)
                else:
                    dev._mr = _SynchronousMemoryResource(dev_id)

                dev._has_inited = False
                dev._properties = None
                devices.append(dev)

        try:
            return devices[device_id]
        except IndexError:
            raise ValueError(f"device_id must be within [0, {len(devices)}), got {device_id}") from None

    def _check_context_initialized(self):
        if not self._has_inited:
            raise CUDAError(
                f"Device {self._id} is not yet initialized, perhaps you forgot to call .set_current() first?"
            )

    def _get_primary_context(self) -> driver.CUcontext:
        try:
            primary_ctxs = _tls.primary_ctxs
        except AttributeError:
            total = len(_tls.devices)
            primary_ctxs = _tls.primary_ctxs = [None] * total
        ctx = primary_ctxs[self._id]
        if ctx is None:
            ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
            primary_ctxs[self._id] = ctx
        return ctx

    def _get_current_context(self, check_consistency=False) -> driver.CUcontext:
        err, ctx = driver.cuCtxGetCurrent()

        # TODO: We want to just call this:
        # _check_driver_error(err)
        # but even the simplest success check causes 50-100 ns. Wait until we cythonize this file...
        if ctx is None:
            _check_driver_error(err)

        if int(ctx) == 0:
            raise CUDAError("No context is bound to the calling CPU thread.")
        if check_consistency:
            err, dev = driver.cuCtxGetDevice()
            if err != _SUCCESS:
                handle_return((err,))
            if int(dev) != self._id:
                raise CUDAError("Internal error (current device is not equal to Device.device_id)")
        return ctx

    @property
    def device_id(self) -> int:
        """Return device ordinal."""
        return self._id

    @property
    def pci_bus_id(self) -> str:
        """Return a PCI Bus Id string for this device."""
        bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self) -> str:
        """Return a UUID for the device.

        Returns 16-octets identifying the device. If the device is in
        MIG mode, returns its MIG UUID which uniquely identifies the
        subscribed MIG compute instance.

        Note
        ----
        MIG UUID is only returned when device is in MIG mode and the
        driver is older than CUDA 11.4.

        """
        driver_ver = handle_return(driver.cuDriverGetVersion())
        if 11040 <= driver_ver < 13000:
            uuid = handle_return(driver.cuDeviceGetUuid_v2(self._id))
        else:
            uuid = handle_return(driver.cuDeviceGetUuid(self._id))
        uuid = uuid.bytes.hex()
        # 8-4-4-4-12
        return f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"

    @property
    def name(self) -> str:
        """Return the device name."""
        # Use 256 characters to be consistent with CUDA Runtime
        name = handle_return(driver.cuDeviceGetName(256, self._id))
        name = name.split(b"\0")[0]
        return name.decode()

    @property
    def properties(self) -> DeviceProperties:
        """Return a :obj:`~_device.DeviceProperties` class with information about the device."""
        if self._properties is None:
            self._properties = DeviceProperties._init(self._id)

        return self._properties

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        if "compute_capability" in self.properties._cache:
            return self.properties._cache["compute_capability"]
        cc = ComputeCapability(self.properties.compute_capability_major, self.properties.compute_capability_minor)
        self.properties._cache["compute_capability"] = cc
        return cc

    @property
    def arch(self) -> str:
        """Return compute capability as a string (e.g., '75' for CC 7.5)."""
        return f"{self.compute_capability.major}{self.compute_capability.minor}"

    @property
    def context(self) -> Context:
        """Return the current :obj:`~_context.Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        self._check_context_initialized()
        ctx = self._get_current_context(check_consistency=True)
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`~_memory.MemoryResource` associated with this device."""
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        assert_type(mr, MemoryResource)
        self._mr = mr

    @property
    def default_stream(self) -> Stream:
        """Return default CUDA :obj:`~_stream.Stream` associated with this device.

        The type of default stream returned depends on if the environment
        variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

        If set, returns a per-thread default stream. Otherwise returns
        the legacy stream.

        """
        return default_stream()

    def __int__(self):
        """Return device_id."""
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def set_current(self, ctx: Context = None) -> Union[Context, None]:
        """Set device to be used for GPU executions.

        Initializes CUDA and sets the calling thread to a valid CUDA
        context. By default the primary context is used, but optional `ctx`
        parameter can be used to explicitly supply a :obj:`~_context.Context` object.

        Providing a `ctx` causes the previous set context to be popped and returned.

        Parameters
        ----------
        ctx : :obj:`~_context.Context`, optional
            Optional context to push onto this device's current thread stack.

        Returns
        -------
        Union[:obj:`~_context.Context`, None], optional
            Popped context.

        Examples
        --------
        Acts as an entry point of this object. Users always start a code by
        calling this method, e.g.

        >>> from cuda.core.experimental import Device
        >>> dev0 = Device(0)
        >>> dev0.set_current()
        >>> # ... do work on device 0 ...

        """
        if ctx is not None:
            assert_type(ctx, Context)
            if ctx._id != self._id:
                raise RuntimeError(
                    "the provided context was created on the device with"
                    f" id={ctx._id}, which is different from the target id={self._id}"
                )
            prev_ctx = handle_return(driver.cuCtxPopCurrent())
            handle_return(driver.cuCtxPushCurrent(ctx._handle))
            self._has_inited = True
            if int(prev_ctx) != 0:
                return Context._from_ctx(prev_ctx, self._id)
        else:
            # use primary ctx
            ctx = self._get_primary_context()
            handle_return(driver.cuCtxSetCurrent(ctx))
            self._has_inited = True

    def create_context(self, options: ContextOptions = None) -> Context:
        """Create a new :obj:`~_context.Context` object.

        Note
        ----
        The newly context will not be set as current.

        Parameters
        ----------
        options : :obj:`~_context.ContextOptions`, optional
            Customizable dataclass for context creation options.

        Returns
        -------
        :obj:`~_context.Context`
            Newly created context object.

        """
        raise NotImplementedError("WIP: https://github.com/NVIDIA/cuda-python/issues/189")

    def create_stream(self, obj: Optional[IsStreamT] = None, options: Optional[StreamOptions] = None) -> Stream:
        """Create a Stream object.

        New stream objects can be created in two different ways:

        1) Create a new CUDA stream with customizable ``options``.
        2) Wrap an existing foreign `obj` supporting the ``__cuda_stream__`` protocol.

        Option (2) internally holds a reference to the foreign object
        such that the lifetime is managed.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        obj : :obj:`~_stream.IsStreamT`, optional
            Any object supporting the ``__cuda_stream__`` protocol.
        options : :obj:`~_stream.StreamOptions`, optional
            Customizable dataclass for stream creation options.

        Returns
        -------
        :obj:`~_stream.Stream`
            Newly created stream object.

        """
        self._check_context_initialized()
        return Stream._init(obj=obj, options=options, device_id=self._id)

    def create_event(self, options: Optional[EventOptions] = None) -> Event:
        """Create an Event object without recording it to a Stream.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        options : :obj:`EventOptions`, optional
            Customizable dataclass for event creation options.

        Returns
        -------
        :obj:`~_event.Event`
            Newly created event object.

        """
        self._check_context_initialized()
        ctx = self._get_current_context()
        return Event._init(self._id, ctx, options)

    def allocate(self, size, stream: Optional[Stream] = None) -> Buffer:
        """Allocate device memory from a specified stream.

        Allocates device memory of `size` bytes on the specified `stream`
        using the memory resource currently associated with this Device.

        Parameter `stream` is optional, using a default stream by default.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.
        stream : :obj:`~_stream.Stream`, optional
            The stream establishing the stream ordering semantic.
            Default value of `None` uses default stream.

        Returns
        -------
        :obj:`~_memory.Buffer`
            Newly created buffer object.

        """
        self._check_context_initialized()
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    def sync(self):
        """Synchronize the device.

        Note
        ----
        Device must be initialized.

        """
        self._check_context_initialized()
        handle_return(runtime.cudaDeviceSynchronize())

    def create_graph_builder(self) -> GraphBuilder:
        """Create a new :obj:`~_graph.GraphBuilder` object.

        Returns
        -------
        :obj:`~_graph.GraphBuilder`
            Newly created graph builder object.

        """
        self._check_context_initialized()
        return GraphBuilder._init(stream=self.create_stream(), is_stream_owner=True)
