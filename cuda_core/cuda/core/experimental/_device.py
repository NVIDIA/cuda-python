# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import threading
from typing import Union

from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._memory import Buffer, MemoryResource, _DefaultAsyncMempool, _SynchronousMemoryResource
from cuda.core.experimental._stream import Stream, StreamOptions, default_stream
from cuda.core.experimental._utils import ComputeCapability, CUDAError, driver, handle_return, precondition, runtime

_tls = threading.local()
_tls_lock = threading.Lock()


# ruff: noqa
class DeviceProperties:
    """
    A class to query various attributes of a CUDA device.

    Attributes are read-only and provide information about the device.

    Attributes:
        max_threads_per_block (int): Maximum number of threads per block.
        max_block_dim_x (int): Maximum x-dimension of a block.
        max_block_dim_y (int): Maximum y-dimension of a block.
        max_block_dim_z (int): Maximum z-dimension of a block.
        max_grid_dim_x (int): Maximum x-dimension of a grid.
        max_grid_dim_y (int): Maximum y-dimension of a grid.
        max_grid_dim_z (int): Maximum z-dimension of a grid.
        max_shared_memory_per_block (int): Maximum amount of shared memory available to a thread block in bytes.
        total_constant_memory (int): Memory available on device for __constant__ variables in a CUDA C kernel in bytes.
        warp_size (int): Warp size in threads.
        max_pitch (int): Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cuMemAllocPitch().
        maximum_texture1d_width (int): Maximum 1D texture width.
        maximum_texture1d_linear_width (int): Maximum width for a 1D texture bound to linear memory.
        maximum_texture1d_mipmapped_width (int): Maximum mipmapped 1D texture width.
        maximum_texture2d_width (int): Maximum 2D texture width.
        maximum_texture2d_height (int): Maximum 2D texture height.
        maximum_texture2d_linear_width (int): Maximum width for a 2D texture bound to linear memory.
        maximum_texture2d_linear_height (int): Maximum height for a 2D texture bound to linear memory.
        maximum_texture2d_linear_pitch (int): Maximum pitch in bytes for a 2D texture bound to linear memory.
        maximum_texture2d_mipmapped_width (int): Maximum mipmapped 2D texture width.
        maximum_texture2d_mipmapped_height (int): Maximum mipmapped 2D texture height.
        maximum_texture3d_width (int): Maximum 3D texture width.
        maximum_texture3d_height (int): Maximum 3D texture height.
        maximum_texture3d_depth (int): Maximum 3D texture depth.
        maximum_texture3d_width_alternate (int): Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported.
        maximum_texture3d_height_alternate (int): Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported.
        maximum_texture3d_depth_alternate (int): Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported.
        maximum_texturecubemap_width (int): Maximum cubemap texture width or height.
        maximum_texture1d_layered_width (int): Maximum 1D layered texture width.
        maximum_texture1d_layered_layers (int): Maximum layers in a 1D layered texture.
        maximum_texture2d_layered_width (int): Maximum 2D layered texture width.
        maximum_texture2d_layered_height (int): Maximum 2D layered texture height.
        maximum_texture2d_layered_layers (int): Maximum layers in a 2D layered texture.
        maximum_texturecubemap_layered_width (int): Maximum cubemap layered texture width or height.
        maximum_texturecubemap_layered_layers (int): Maximum layers in a cubemap layered texture.
        maximum_surface1d_width (int): Maximum 1D surface width.
        maximum_surface2d_width (int): Maximum 2D surface width.
        maximum_surface2d_height (int): Maximum 2D surface height.
        maximum_surface3d_width (int): Maximum 3D surface width.
        maximum_surface3d_height (int): Maximum 3D surface height.
        maximum_surface3d_depth (int): Maximum 3D surface depth.
        maximum_surface1d_layered_width (int): Maximum 1D layered surface width.
        maximum_surface1d_layered_layers (int): Maximum layers in a 1D layered surface.
        maximum_surface2d_layered_width (int): Maximum 2D layered surface width.
        maximum_surface2d_layered_height (int): Maximum 2D layered surface height.
        maximum_surface2d_layered_layers (int): Maximum layers in a 2D layered surface.
        maximum_surfacecubemap_width (int): Maximum cubemap surface width.
        maximum_surfacecubemap_layered_width (int): Maximum cubemap layered surface width.
        maximum_surfacecubemap_layered_layers (int): Maximum layers in a cubemap layered surface.
        max_registers_per_block (int): Maximum number of 32-bit registers available to a thread block.
        clock_rate (int): The typical clock frequency in kilohertz.
        texture_alignment (int): Alignment requirement; texture base addresses aligned to textureAlign bytes do not need an offset applied to texture fetches.
        texture_pitch_alignment (int): Pitch alignment requirement for 2D texture references bound to pitched memory.
        gpu_overlap (bool): True if the device can concurrently copy memory between host and device while executing a kernel, False if not.
        multiprocessor_count (int): Number of multiprocessors on the device.
        kernel_exec_timeout (bool): True if there is a run time limit for kernels executed on the device, False if not.
        integrated (bool): True if the device is integrated with the memory subsystem, False if not.
        can_map_host_memory (bool): True if the device can map host memory into the CUDA address space, False if not.
        compute_mode (int): Compute mode that device is currently in.
        concurrent_kernels (bool): True if the device supports executing multiple kernels within the same context simultaneously, False if not.
        ecc_enabled (bool): True if error correction is enabled on the device, False if error correction is disabled or not supported by the device.
        pci_bus_id (int): PCI bus identifier of the device.
        pci_device_id (int): PCI device (also known as slot) identifier of the device.
        pci_domain_id (int): PCI domain identifier of the device.
        tcc_driver (bool): True if the device is using a TCC driver, False if not.
        memory_clock_rate (int): Peak memory clock frequency in kilohertz.
        global_memory_bus_width (int): Global memory bus width in bits.
        l2_cache_size (int): Size of L2 cache in bytes, 0 if the device doesn't have L2 cache.
        max_threads_per_multiprocessor (int): Maximum resident threads per multiprocessor.
        unified_addressing (bool): True if the device shares a unified address space with the host, False if not.
        compute_capability_major (int): Major compute capability version number.
        compute_capability_minor (int): Minor compute capability version number.
        global_l1_cache_supported (bool): True if device supports caching globals in L1 cache, False if caching globals in L1 cache is not supported by the device.
        local_l1_cache_supported (bool): True if device supports caching locals in L1 cache, False if caching locals in L1 cache is not supported by the device.
        max_shared_memory_per_multiprocessor (int): Maximum amount of shared memory available to a multiprocessor in bytes.
        max_registers_per_multiprocessor (int): Maximum number of 32-bit registers available to a multiprocessor.
        managed_memory (bool): True if device supports allocating managed memory on this system, False if allocating managed memory is not supported by the device on this system.
        multi_gpu_board (bool): True if device is on a multi-GPU board, False if not.
        multi_gpu_board_group_id (int): Unique identifier for a group of devices associated with the same board.
        host_native_atomic_supported (bool): True if Link between the device and the host supports native atomic operations, False if not.
        single_to_double_precision_perf_ratio (int): Ratio of single precision performance (in floating-point operations per second) to double precision performance.
        pageable_memory_access (bool): True if device supports coherently accessing pageable memory without calling cudaHostRegister on it, False if not.
        concurrent_managed_access (bool): True if device can coherently access managed memory concurrently with the CPU, False if not.
        compute_preemption_supported (bool): True if device supports Compute Preemption, False if not.
        can_use_host_pointer_for_registered_mem (bool): True if device can access host registered memory at the same virtual address as the CPU, False if not.
        max_shared_memory_per_block_optin (int): The maximum per block shared memory size supported on this device.
        pageable_memory_access_uses_host_page_tables (bool): True if device accesses pageable memory via the host's page tables, False if not.
        direct_managed_mem_access_from_host (bool): True if the host can directly access managed memory on the device without migration, False if not.
        virtual_memory_management_supported (bool): True if device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap and related APIs, False if not.
        handle_type_posix_file_descriptor_supported (bool): True if device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle, False if not.
        handle_type_win32_handle_supported (bool): True if device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle, False if not.
        handle_type_win32_kmt_handle_supported (bool): True if device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle, False if not.
        max_blocks_per_multiprocessor (int): Maximum number of thread blocks that can reside on a multiprocessor.
        generic_compression_supported (bool): True if device supports compressible memory allocation via cuMemCreate, False if not.
        max_persisting_l2_cache_size (int): Maximum L2 persisting lines capacity setting in bytes.
        max_access_policy_window_size (int): Maximum value of CUaccessPolicyWindow::num_bytes.
        gpu_direct_rdma_with_cuda_vmm_supported (bool): True if device supports specifying the GPUDirect RDMA flag with cuMemCreate, False if not.
        reserved_shared_memory_per_block (int): Amount of shared memory per block reserved by CUDA driver in bytes.
        sparse_cuda_array_supported (bool): True if device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, False if not.
        read_only_host_register_supported (bool): True if device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU, False if not.
        memory_pools_supported (bool): True if device supports using the cuMemAllocAsync and cuMemPool family of APIs, False if not.
        gpu_direct_rdma_supported (bool): True if device supports GPUDirect RDMA APIs, False if not.
        gpu_direct_rdma_flush_writes_options (int): The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the CUflushGPUDirectRDMAWritesOptions enum.
        gpu_direct_rdma_writes_ordering (int): GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute.
        mempool_supported_handle_types (int): Bitmask of handle types supported with mempool based IPC.
        deferred_mapping_cuda_array_supported (bool): True if device supports deferred mapping CUDA arrays and CUDA mipmapped arrays, False if not.
        numa_config (int): NUMA configuration of a device.
        numa_id (int): NUMA node ID of the GPU memory.
        multicast_supported (bool): True if device supports switch multicast and reduction operations, False if not.
    """

    def __init__(self):
        raise RuntimeError("DeviceProperties should not be instantiated directly")

    slots = "_handle"

    def _init(handle):
        self = DeviceProperties.__new__(DeviceProperties)
        self._handle = handle
        return self

    @property
    def max_threads_per_block(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self._handle
            )
        )

    @property
    def max_block_dim_x(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, self._handle)
        )

    @property
    def max_block_dim_y(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, self._handle)
        )

    @property
    def max_block_dim_z(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, self._handle)
        )

    @property
    def max_grid_dim_x(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, self._handle)
        )

    @property
    def max_grid_dim_y(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, self._handle)
        )

    @property
    def max_grid_dim_z(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, self._handle)
        )

    @property
    def max_shared_memory_per_block(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, self._handle
            )
        )

    @property
    def total_constant_memory(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, self._handle
            )
        )

    @property
    def warp_size(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, self._handle)
        )

    @property
    def max_pitch(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH, self._handle)
        )

    @property
    def maximum_texture1d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture1d_linear_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture1d_mipmapped_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture2d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture2d_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, self._handle
            )
        )

    @property
    def maximum_texture2d_linear_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture2d_linear_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, self._handle
            )
        )

    @property
    def maximum_texture2d_linear_pitch(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, self._handle
            )
        )

    @property
    def maximum_texture2d_mipmapped_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture2d_mipmapped_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, self._handle
            )
        )

    @property
    def maximum_texture3d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture3d_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, self._handle
            )
        )

    @property
    def maximum_texture3d_depth(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, self._handle
            )
        )

    @property
    def maximum_texture3d_width_alternate(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, self._handle
            )
        )

    @property
    def maximum_texture3d_height_alternate(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, self._handle
            )
        )

    @property
    def maximum_texture3d_depth_alternate(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, self._handle
            )
        )

    @property
    def maximum_texturecubemap_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture1d_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture1d_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def maximum_texture2d_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_texture2d_layered_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, self._handle
            )
        )

    @property
    def maximum_texture2d_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def maximum_texturecubemap_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_texturecubemap_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def maximum_surface1d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, self._handle
            )
        )

    @property
    def maximum_surface2d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, self._handle
            )
        )

    @property
    def maximum_surface2d_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, self._handle
            )
        )

    @property
    def maximum_surface3d_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, self._handle
            )
        )

    @property
    def maximum_surface3d_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, self._handle
            )
        )

    @property
    def maximum_surface3d_depth(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, self._handle
            )
        )

    @property
    def maximum_surface1d_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_surface1d_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def maximum_surface2d_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_surface2d_layered_height(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, self._handle
            )
        )

    @property
    def maximum_surface2d_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def maximum_surfacecubemap_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, self._handle
            )
        )

    @property
    def maximum_surfacecubemap_layered_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, self._handle
            )
        )

    @property
    def maximum_surfacecubemap_layered_layers(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, self._handle
            )
        )

    @property
    def max_registers_per_block(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, self._handle
            )
        )

    @property
    def clock_rate(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, self._handle)
        )

    @property
    def texture_alignment(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, self._handle)
        )

    @property
    def texture_pitch_alignment(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, self._handle
            )
        )

    @property
    def gpu_overlap(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, self._handle)
            )
        )

    @property
    def multiprocessor_count(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self._handle
            )
        )

    @property
    def kernel_exec_timeout(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, self._handle
                )
            )
        )

    @property
    def integrated(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, self._handle)
            )
        )

    @property
    def can_map_host_memory(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, self._handle
                )
            )
        )

    @property
    def compute_mode(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, self._handle)
        )

    @property
    def concurrent_kernels(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, self._handle
                )
            )
        )

    @property
    def ecc_enabled(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED, self._handle)
            )
        )

    @property
    def pci_bus_id(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, self._handle)
        )

    @property
    def pci_device_id(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, self._handle)
        )

    @property
    def pci_domain_id(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, self._handle)
        )

    @property
    def tcc_driver(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER, self._handle)
            )
        )

    @property
    def memory_clock_rate(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, self._handle)
        )

    @property
    def global_memory_bus_width(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, self._handle
            )
        )

    @property
    def l2_cache_size(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, self._handle)
        )

    @property
    def max_threads_per_multiprocessor(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, self._handle
            )
        )

    @property
    def unified_addressing(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, self._handle
                )
            )
        )

    @property
    def compute_capability_major(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self._handle
            )
        )

    @property
    def compute_capability_minor(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self._handle
            )
        )

    @property
    def global_l1_cache_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, self._handle
                )
            )
        )

    @property
    def local_l1_cache_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, self._handle
                )
            )
        )

    @property
    def max_shared_memory_per_multiprocessor(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, self._handle
            )
        )

    @property
    def max_registers_per_multiprocessor(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, self._handle
            )
        )

    @property
    def managed_memory(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, self._handle)
            )
        )

    @property
    def multi_gpu_board(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, self._handle)
            )
        )

    @property
    def multi_gpu_board_group_id(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, self._handle
            )
        )

    @property
    def host_native_atomic_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, self._handle
                )
            )
        )

    @property
    def single_to_double_precision_perf_ratio(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, self._handle
            )
        )

    @property
    def pageable_memory_access(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, self._handle
                )
            )
        )

    @property
    def concurrent_managed_access(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, self._handle
                )
            )
        )

    @property
    def compute_preemption_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, self._handle
                )
            )
        )

    @property
    def can_use_host_pointer_for_registered_mem(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, self._handle
                )
            )
        )

    @property
    def max_shared_memory_per_block_optin(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self._handle
            )
        )

    @property
    def pageable_memory_access_uses_host_page_tables(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                    self._handle,
                )
            )
        )

    @property
    def direct_managed_mem_access_from_host(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, self._handle
                )
            )
        )

    @property
    def virtual_memory_management_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, self._handle
                )
            )
        )

    @property
    def handle_type_posix_file_descriptor_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                    self._handle,
                )
            )
        )

    @property
    def handle_type_win32_handle_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, self._handle
                )
            )
        )

    @property
    def handle_type_win32_kmt_handle_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED, self._handle
                )
            )
        )

    @property
    def max_blocks_per_multiprocessor(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, self._handle
            )
        )

    @property
    def generic_compression_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, self._handle
                )
            )
        )

    @property
    def max_persisting_l2_cache_size(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, self._handle
            )
        )

    @property
    def max_access_policy_window_size(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, self._handle
            )
        )

    @property
    def gpu_direct_rdma_with_cuda_vmm_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, self._handle
                )
            )
        )

    @property
    def reserved_shared_memory_per_block(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, self._handle
            )
        )

    @property
    def sparse_cuda_array_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, self._handle
                )
            )
        )

    @property
    def read_only_host_register_supported(self):
        return bool(handle_return(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY)))

    @property
    def memory_pools_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, self._handle
                )
            )
        )

    @property
    def gpu_direct_rdma_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, self._handle
                )
            )
        )

    @property
    def gpu_direct_rdma_flush_writes_options(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, self._handle
            )
        )

    @property
    def gpu_direct_rdma_writes_ordering(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, self._handle
            )
        )

    @property
    def mempool_supported_handle_types(self):
        return handle_return(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, self._handle
            )
        )

    @property
    def deferred_mapping_cuda_array_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED, self._handle
                )
            )
        )

    @property
    def numa_config(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_CONFIG, self._handle)
        )

    @property
    def numa_id(self):
        return handle_return(
            driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_ID, self._handle)
        )

    @property
    def multicast_supported(self):
        return bool(
            handle_return(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, self._handle
                )
            )
        )


# ruff: enable


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

    def __new__(cls, device_id=None):
        # important: creating a Device instance does not initialize the GPU!
        if device_id is None:
            device_id = handle_return(runtime.cudaGetDevice())
            assert isinstance(device_id, int), f"{device_id=}"
        else:
            total = handle_return(runtime.cudaGetDeviceCount())
            if not isinstance(device_id, int) or not (0 <= device_id < total):
                raise ValueError(f"device_id must be within [0, {total}), got {device_id}")

        # ensure Device is singleton
        with _tls_lock:
            if not hasattr(_tls, "devices"):
                total = handle_return(runtime.cudaGetDeviceCount())
                _tls.devices = []
                for dev_id in range(total):
                    dev = super().__new__(cls)
                    dev._id = dev_id
                    # If the device is in TCC mode, or does not support memory pools for some other reason,
                    # use the SynchronousMemoryResource which does not use memory pools.
                    if (
                        handle_return(
                            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
                        )
                    ) == 1:
                        dev._mr = _DefaultAsyncMempool(dev_id)
                    else:
                        dev._mr = _SynchronousMemoryResource(dev_id)

                    dev._has_inited = False
                    dev._properties = None
                    _tls.devices.append(dev)

        return _tls.devices[device_id]

    def _check_context_initialized(self, *args, **kwargs):
        if not self._has_inited:
            raise CUDAError("the device is not yet initialized, perhaps you forgot to call .set_current() first?")

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
        if driver_ver >= 11040:
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
        """Return information about the compute-device."""
        if self._properties is None:
            self._properties = DeviceProperties._init(self._id)

        return self._properties

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        major = handle_return(
            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self._id)
        )
        minor = handle_return(
            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self._id)
        )
        return ComputeCapability(major, minor)

    @property
    @precondition(_check_context_initialized)
    def context(self) -> Context:
        """Return the current :obj:`~_context.Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        ctx = handle_return(driver.cuCtxGetCurrent())
        assert int(ctx) != 0
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`~_memory.MemoryResource` associated with this device."""
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        if not isinstance(mr, MemoryResource):
            raise TypeError
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
            if not isinstance(ctx, Context):
                raise TypeError("a Context object is required")
            if ctx._id != self._id:
                raise RuntimeError(
                    "the provided context was created on a different "
                    f"device {ctx._id} other than the target {self._id}"
                )
            prev_ctx = handle_return(driver.cuCtxPopCurrent())
            handle_return(driver.cuCtxPushCurrent(ctx._handle))
            self._has_inited = True
            if int(prev_ctx) != 0:
                return Context._from_ctx(prev_ctx, self._id)
        else:
            ctx = handle_return(driver.cuCtxGetCurrent())
            if int(ctx) == 0:
                # use primary ctx
                ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                handle_return(driver.cuCtxPushCurrent(ctx))
            else:
                ctx_id = handle_return(driver.cuCtxGetDevice())
                if ctx_id != self._id:
                    # use primary ctx
                    ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                    handle_return(driver.cuCtxPushCurrent(ctx))
                else:
                    # no-op, a valid context already exists and is set current
                    pass
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
        raise NotImplementedError("TODO")

    @precondition(_check_context_initialized)
    def create_stream(self, obj=None, options: StreamOptions = None) -> Stream:
        """Create a Stream object.

        New stream objects can be created in two different ways:

        1) Create a new CUDA stream with customizable `options`.
        2) Wrap an existing foreign `obj` supporting the __cuda_stream__ protocol.

        Option (2) internally holds a reference to the foreign object
        such that the lifetime is managed.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        obj : Any, optional
            Any object supporting the __cuda_stream__ protocol.
        options : :obj:`~_stream.StreamOptions`, optional
            Customizable dataclass for stream creation options.

        Returns
        -------
        :obj:`~_stream.Stream`
            Newly created stream object.

        """
        return Stream._init(obj=obj, options=options)

    @precondition(_check_context_initialized)
    def allocate(self, size, stream=None) -> Buffer:
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
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    @precondition(_check_context_initialized)
    def sync(self):
        """Synchronize the device.

        Note
        ----
        Device must be initialized.

        """
        handle_return(runtime.cudaDeviceSynchronize())
