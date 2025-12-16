# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use `cuda.core` to show the properties of the
# CUDA devices in the system.
#
# ################################################################################

import sys

from cuda.core.experimental import Device, system


# Convert boolean to YES or NO string
def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


# Convert value in bytes to MB or GB string
BYTES_TO_MBYTES = 1 / (1024 * 1024)
BYTES_TO_GBYTES = BYTES_TO_MBYTES / 1024


def _bytes_to_mbytes(value):
    return f"{value * BYTES_TO_MBYTES:.2f}MB"


def _bytes_to_gbytes(value):
    return f"{value * BYTES_TO_GBYTES:.2f}GB"


# Convert value in KHz to GHz string
KHZ_TO_GHZ = 1e-6


def _khz_to_ghz(value):
    return f"{value * KHZ_TO_GHZ:.2f}GHz"


# Print device properties to stdout
def print_device_properties(properties):
    print("Properties:\n------------")
    print(f"- Can map host memory into the CUDA address space: {_yes_no(properties.can_map_host_memory)}")
    print(
        "- Can access host registered memory at the same virtual address as the CPU: "
        + f"{_yes_no(properties.can_use_host_pointer_for_registered_mem)}"
    )
    print(f"- Clock rate: {_khz_to_ghz(properties.clock_rate)}")
    print(f"- Peak memory clock frequency: {_khz_to_ghz(properties.memory_clock_rate)}")
    print(
        "- Performance ratio (single precision)/(double precision): "
        + f"{properties.single_to_double_precision_perf_ratio}"
    )
    print(
        f"- Compute capability: major={properties.compute_capability_major}, "
        + f"minor={properties.compute_capability_minor}"
    )
    print(f"- Compute mode: {properties.compute_mode} (0 - default, 2 - prohibited, 3 - exclusive process)")
    print(f"- Support for Compute Preemption: {_yes_no(properties.compute_preemption_supported)}")
    print(
        "- Support for concurrent kernels execution within the same context: "
        + f"{_yes_no(properties.concurrent_kernels)}"
    )
    print(
        "- Support for coherent access to managed memory concurrently with CPU: "
        + f"{_yes_no(properties.concurrent_managed_access)}"
    )
    print(
        "- Support for deferred mapping in CUDA arrays and CUDA mipmapped arrays: "
        + f"{_yes_no(properties.deferred_mapping_cuda_array_supported)}"
    )
    print(
        "- Support for direct access of managed memory on device without migration: "
        + f"{_yes_no(properties.direct_managed_mem_access_from_host)}"
    )
    print(f"- ECC enabled: {_yes_no(properties.ecc_enabled)}")
    print(f"- Support for generic compression: {_yes_no(properties.generic_compression_supported)}")
    print(f"- Support for caching globals in L1 cache: {_yes_no(properties.global_l1_cache_supported)}")
    print(f"- Support for caching locals in L1 cache: {_yes_no(properties.local_l1_cache_supported)}")
    print(f"- Global memory bus widths: {properties.global_memory_bus_width} bits")
    print(f"- Support for GPUDirect RDMA: {_yes_no(properties.gpu_direct_rdma_supported)}")
    print(f"- GPUDirect RDMA flush-writes options bitmask: 0b{properties.gpu_direct_rdma_flush_writes_options:032b}")
    print(
        f"- GPUDirect RDMA writes ordering: {properties.gpu_direct_rdma_writes_ordering} "
        + "(0 - none, 100 - this device can consume remote writes, "
        + "200 - any CUDA device can consume remote writes to this device)"
    )
    print(
        "- Can concurrently copy memory between host and device while executing kernel: "
        + f"{_yes_no(properties.gpu_overlap)}"
    )
    print(
        "- Support for exporting memory to a posix file descriptor: "
        + f"{_yes_no(properties.handle_type_posix_file_descriptor_supported)}"
    )
    print(
        "- Support for exporting memory to a Win32 NT handle: "
        + f"{_yes_no(properties.handle_type_win32_handle_supported)}"
    )
    print(
        "- Support for exporting memory to a Win32 KMT handle: "
        + f"{_yes_no(properties.handle_type_win32_kmt_handle_supported)}"
    )
    print(
        "- Link between device and host supports native atomic operations: "
        + f"{_yes_no(properties.host_native_atomic_supported)}"
    )
    print(f"- Device is integrated with memory subsystem: {_yes_no(properties.integrated)}")
    print(f"- Kernel execution timeout: {_yes_no(properties.kernel_exec_timeout)}")
    print(f"- L2 cache size: {_bytes_to_mbytes(properties.l2_cache_size)}")
    print(f"- Max L2 persisting lines capacity: {_bytes_to_mbytes(properties.max_persisting_l2_cache_size)}")
    print(f"- Support for managed memory allocation: {_yes_no(properties.managed_memory)}")
    print(f"- Max access policy window size: {_bytes_to_mbytes(properties.max_access_policy_window_size)}")
    print(f"- Max x-dimension of a block: {properties.max_block_dim_x}")
    print(f"- Max y-dimension of a block: {properties.max_block_dim_y}")
    print(f"- Max z-dimension of a block: {properties.max_block_dim_z}")
    print(f"- Max blocks in a multiprocessor: {properties.max_blocks_per_multiprocessor}")
    print(f"- Max x-dimension of a grid: {properties.max_grid_dim_x}")
    print(f"- Max y-dimension of a grid: {properties.max_grid_dim_y}")
    print(f"- Max z-dimension of a grid: {properties.max_grid_dim_z}")
    print(f"- Max pitch allowed by the memory copy functions: {_bytes_to_gbytes(properties.max_pitch)}")
    print(f"- Max number of 32-bit registers per block: {properties.max_registers_per_block}")
    print(f"- Max number of 32-bit registers in a multiprocessor: {properties.max_registers_per_multiprocessor}")
    print(f"- Max shared memory per block: {properties.max_shared_memory_per_block}B")
    print(f"- Max optin shared memory per block: {properties.max_shared_memory_per_block_optin}B")
    print(f"- Max shared memory available to a multiprocessor: {properties.max_shared_memory_per_multiprocessor}B")
    print(f"- Max threads per block: {properties.max_threads_per_block}")
    print(f"- Max threads per multiprocessor: {properties.max_threads_per_multiprocessor}")
    print(f"- Warp size: {properties.warp_size}")
    print(f"- Max 1D surface width: {properties.maximum_surface1d_width}")
    print(f"- Max layers in 1D layered surface: {properties.maximum_surface1d_layered_layers}")
    print(f"- Max 1D layered surface width: {properties.maximum_surface1d_layered_width}")
    print(f"- Max 2D surface width: {properties.maximum_surface2d_width}")
    print(f"- Max 2D surface height: {properties.maximum_surface2d_height}")
    print(f"- Max layers in 2D layered surface: {properties.maximum_surface2d_layered_layers}")
    print(f"- Max 2D layered surface width: {properties.maximum_surface2d_layered_width}")
    print(f"- Max 2D layered surface height: {properties.maximum_surface2d_layered_height}")
    print(f"- Max 3D surface width: {properties.maximum_surface3d_width}")
    print(f"- Max 3D surface height: {properties.maximum_surface3d_height}")
    print(f"- Max 3D surface depth: {properties.maximum_surface3d_depth}")
    print(f"- Max cubemap surface width: {properties.maximum_surfacecubemap_width}")
    print(f"- Max layers in a cubemap layered surface: {properties.maximum_surfacecubemap_layered_layers}")
    print(f"- Max cubemap layered surface width: {properties.maximum_surfacecubemap_layered_width}")
    print(f"- Max 1D texture width: {properties.maximum_texture1d_width}")
    print(f"- Max width for a 1D texture bound to linear memory: {properties.maximum_texture1d_linear_width}")
    print(f"- Max layers in 1D layered texture: {properties.maximum_texture1d_layered_layers}")
    print(f"- Max 1D layered texture width: {properties.maximum_texture1d_layered_width}")
    print(f"- Max mipmapped 1D texture width: {properties.maximum_texture1d_mipmapped_width}")
    print(f"- Max 2D texture width: {properties.maximum_texture2d_width}")
    print(f"- Max 2D texture height: {properties.maximum_texture2d_height}")
    print(f"- Max width for a 2D texture bound to linear memory: {properties.maximum_texture2d_linear_width}")
    print(f"- Max height for a 2D texture bound to linear memory: {properties.maximum_texture2d_linear_height}")
    print(
        "- Max pitch for a 2D texture bound to linear memory: "
        + f"{_bytes_to_mbytes(properties.maximum_texture2d_linear_pitch)}"
    )
    print(f"- Max layers in 2D layered texture: {properties.maximum_texture2d_layered_layers}")
    print(f"- Max 2D layered texture width: {properties.maximum_texture2d_layered_width}")
    print(f"- Max 2D layered texture height: {properties.maximum_texture2d_layered_height}")
    print(f"- Max mipmapped 2D texture width: {properties.maximum_texture2d_mipmapped_width}")
    print(f"- Max mipmapped 2D texture height: {properties.maximum_texture2d_mipmapped_height}")
    print(f"- Max 3D texture width: {properties.maximum_texture3d_width}")
    print(f"- Max 3D texture height: {properties.maximum_texture3d_height}")
    print(f"- Max 3D texture depth: {properties.maximum_texture3d_depth}")
    print(f"- Alternate max 3D texture width: {properties.maximum_texture3d_width_alternate}")
    print(f"- Alternate max 3D texture height: {properties.maximum_texture3d_height_alternate}")
    print(f"- Alternate max 3D texture depth: {properties.maximum_texture3d_depth_alternate}")
    print(f"- Max cubemap texture width or height: {properties.maximum_texturecubemap_width}")
    print(f"- Max layers in a cubemap layered texture: {properties.maximum_texturecubemap_layered_layers}")
    print(f"- Max cubemap layered texture width or height: {properties.maximum_texturecubemap_layered_width}")
    print(f"- Texture base address alignment requirement: {properties.texture_alignment}B")
    print(
        "- Pitch alignment requirement for 2D texture references bound to pitched memory: "
        + f"{properties.texture_pitch_alignment}B"
    )
    print(f"- Support for memory pools: {_yes_no(properties.memory_pools_supported)}")
    print(
        "- Bitmask of handle types supported with memory pool-based IPC: "
        + f"0b{properties.mempool_supported_handle_types:032b}"
    )
    print(f"- Multi-GPU board: {_yes_no(properties.multi_gpu_board)}")
    print(f"- Multi-GPU board group ID: {properties.multi_gpu_board_group_id}")
    print(f"- Support for switch multicast and reduction operations: {_yes_no(properties.multicast_supported)}")
    print(f"- Number of multiprocessors: {properties.multiprocessor_count}")
    print(f"- NUMA configuration: {properties.numa_config}")
    print(f"- NUMA node ID of GPU memory: {properties.numa_id}")
    print(f"- Support for coherently accessing pageable memory: {_yes_no(properties.pageable_memory_access)}")
    print(
        "- Access pageable memory via host's page tables: "
        + f"{_yes_no(properties.pageable_memory_access_uses_host_page_tables)}"
    )
    print(f"- PCI bus ID: {properties.pci_bus_id}")
    print(f"- PCI device (slot) ID: {properties.pci_device_id}")
    print(f"- PCI domain ID: {properties.pci_domain_id}")
    print(
        "- Support for registering memory that must be mapped to GPU as read-only: "
        + f"{_yes_no(properties.read_only_host_register_supported)}"
    )
    print(
        "- Amount of shared memory per block reserved by CUDA driver: "
        + f"{properties.reserved_shared_memory_per_block}B"
    )
    print(
        "- Support for sparse CUDA arrays and sparse CUDA mipmapped arrays: "
        + f"{_yes_no(properties.sparse_cuda_array_supported)}"
    )
    print(f"- Using TCC driver: {_yes_no(properties.tcc_driver)}")
    print(f"- Constant memory available: {properties.total_constant_memory}B")
    print(f"- Support for unified address space with host: {_yes_no(properties.unified_addressing)}")
    print(f"- Support for virtual memory management: {_yes_no(properties.virtual_memory_management_supported)}")


# Print info about all CUDA devices in the system
def show_device_properties():
    ndev = system.get_num_devices()
    print(f"Number of GPUs: {ndev}")

    for device_id in range(ndev):
        device = Device(device_id)
        print(f"DEVICE {device.name} (id={device_id})")

        device.set_current()
        # Extend example to show device context information after #189 is resolved.
        # ctx = device.context

        cc = device.compute_capability
        prop = device.properties

        print(f"Device compute capability: major={cc[0]}, minor={cc[1]}")
        print(f"Architecture: sm_{cc[0]}{cc[1]}")
        print(f"PCI bus id={device.pci_bus_id}")
        print_device_properties(prop)
        print("*****************************************************\n\n")


if __name__ == "__main__":
    assert len(sys.argv) == 1, "no command-line arguments expected"
    show_device_properties()
