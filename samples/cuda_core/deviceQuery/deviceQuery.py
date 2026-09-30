# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0"]
# ///

"""
Device Query using CUDA Core API

This sample enumerates the properties of the CUDA devices present in the
system. It has two output modes:

  * Default:  the classic ``nvidia-smi``-style summary of compute
              capability, memory, kernel launch limits, texture sizes,
              and feature flags (roughly what the C-side ``deviceQuery``
              sample prints).
  * ``--verbose``: additionally dumps the long-tail of
              ``Device.properties`` fields -- surface/texture edge cases,
              GPUDirect RDMA flags, NUMA info, memory-pool handle types,
              sparse/virtual-memory support -- for callers who need a
              full capability report.
"""

import platform
import sys

# cuda.bindings used for properties not yet exposed in cuda.core (see comments below)
try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
    from cuda.core import Device, system
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def print_property(label, value, indent=2):
    """
    Helper function to print device properties with aligned formatting.

    Parameters
    ----------
    label : str
        Property label
    value : any
        Property value
    indent : int
        Number of spaces for indentation (default: 2)
    """
    field_width = 47
    spaces = " " * indent
    print(f"{spaces}{label:<{field_width}}{value}")


def fmt_bytes(size_in_bytes):
    """Format bytes to human-readable string with MBytes."""
    return f"{size_in_bytes / (1024 * 1024):.0f} MBytes ({size_in_bytes} bytes)"


def fmt_hz(rate_in_khz):
    """Format frequency in kHz to MHz and GHz."""
    return f"{rate_in_khz * 1e-3:.0f} MHz ({rate_in_khz * 1e-6:.2f} GHz)"


def fmt_yes_no(val):
    """Format boolean value to Yes/No string."""
    return "Yes" if val else "No"


def convert_sm_ver_to_cores(major, minor):
    """
    Maps SM version to the number of CUDA cores per SM.

    Information taken from:
    https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

    Parameters
    ----------
    major : int
        Major compute capability version
    minor : int
        Minor compute capability version

    Returns
    -------
    int
        Number of CUDA cores per SM, or 0 if unknown
    """
    sm_to_cores = {
        (3, 0): 192,
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,
        (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128,
        (10, 1): 128,
        (10, 3): 128,
        (11, 0): 128,
        (12, 0): 128,
        (12, 1): 128,
    }
    return sm_to_cores.get((major, minor), 0)


def print_verbose_extras(props):
    """Print the long-tail of ``Device.properties`` fields.

    These are the fields that are useful for capability probing but too
    numerous for the default ``nvidia-smi``-style summary. Kept alphabetized
    within each subsection for easy scanning.
    """
    print()
    print("  Verbose device properties:")
    # Memory / caching / addressing
    print_property("Support for caching globals in L1:", fmt_yes_no(props.global_l1_cache_supported), indent=4)
    print_property("Support for caching locals in L1:", fmt_yes_no(props.local_l1_cache_supported), indent=4)
    print_property("Max persisting L2 cache size:", f"{props.max_persisting_l2_cache_size} bytes", indent=4)
    print_property(
        "Max access policy window size:",
        f"{props.max_access_policy_window_size} bytes",
        indent=4,
    )
    print_property(
        "Reserved shared memory per block:",
        f"{props.reserved_shared_memory_per_block} bytes",
        indent=4,
    )
    print_property(
        "Max shared memory per block (opt-in):",
        f"{props.max_shared_memory_per_block_optin} bytes",
        indent=4,
    )

    # Concurrency / preemption
    print_property(
        "Concurrent kernel execution (same context):",
        fmt_yes_no(props.concurrent_kernels),
        indent=4,
    )
    print_property(
        "Concurrent managed access (host + device):",
        fmt_yes_no(props.concurrent_managed_access),
        indent=4,
    )
    print_property(
        "Direct managed access from host (no migration):",
        fmt_yes_no(props.direct_managed_mem_access_from_host),
        indent=4,
    )
    print_property(
        "Pageable memory access:",
        fmt_yes_no(props.pageable_memory_access),
        indent=4,
    )
    print_property(
        "Pageable memory via host page tables:",
        fmt_yes_no(props.pageable_memory_access_uses_host_page_tables),
        indent=4,
    )
    print_property(
        "Host native atomic support:",
        fmt_yes_no(props.host_native_atomic_supported),
        indent=4,
    )
    print_property(
        "Can use host pointer for registered mem:",
        fmt_yes_no(props.can_use_host_pointer_for_registered_mem),
        indent=4,
    )
    print_property(
        "Read-only host-registered memory:",
        fmt_yes_no(props.read_only_host_register_supported),
        indent=4,
    )

    # GPUDirect / RDMA
    print_property(
        "GPUDirect RDMA:",
        fmt_yes_no(props.gpu_direct_rdma_supported),
        indent=4,
    )
    print_property(
        "GPUDirect RDMA flush-writes bitmask:",
        f"0b{props.gpu_direct_rdma_flush_writes_options:032b}",
        indent=4,
    )
    print_property(
        "GPUDirect RDMA writes ordering:",
        props.gpu_direct_rdma_writes_ordering,
        indent=4,
    )

    # Shareable handle support (IPC)
    print_property(
        "POSIX FD memory handle support:",
        fmt_yes_no(props.handle_type_posix_file_descriptor_supported),
        indent=4,
    )
    print_property(
        "Win32 NT handle support:",
        fmt_yes_no(props.handle_type_win32_handle_supported),
        indent=4,
    )
    print_property(
        "Win32 KMT handle support:",
        fmt_yes_no(props.handle_type_win32_kmt_handle_supported),
        indent=4,
    )
    print_property(
        "Memory pool IPC handle bitmask:",
        f"0b{props.mempool_supported_handle_types:032b}",
        indent=4,
    )
    print_property(
        "Memory pools supported:",
        fmt_yes_no(props.memory_pools_supported),
        indent=4,
    )

    # Multi-GPU / multicast
    print_property("Multi-GPU board:", fmt_yes_no(props.multi_gpu_board), indent=4)
    print_property("Multi-GPU board group ID:", props.multi_gpu_board_group_id, indent=4)
    print_property(
        "Switch multicast/reduction ops:",
        fmt_yes_no(props.multicast_supported),
        indent=4,
    )

    # NUMA
    print_property("NUMA configuration:", props.numa_config, indent=4)
    print_property("NUMA node ID of GPU memory:", props.numa_id, indent=4)

    # Surface limits (compact one-liners; textures are already covered above)
    print_property(
        "Max 1D surface width:",
        props.maximum_surface1d_width,
        indent=4,
    )
    print_property(
        "Max 2D surface (WxH):",
        f"{props.maximum_surface2d_width}x{props.maximum_surface2d_height}",
        indent=4,
    )
    print_property(
        "Max 3D surface (WxHxD):",
        f"{props.maximum_surface3d_width}x{props.maximum_surface3d_height}x{props.maximum_surface3d_depth}",
        indent=4,
    )
    print_property(
        "Max cubemap surface width:",
        props.maximum_surfacecubemap_width,
        indent=4,
    )

    # CUDA arrays / VMM / compression
    print_property(
        "Sparse CUDA arrays supported:",
        fmt_yes_no(props.sparse_cuda_array_supported),
        indent=4,
    )
    print_property(
        "Deferred mapping CUDA arrays:",
        fmt_yes_no(props.deferred_mapping_cuda_array_supported),
        indent=4,
    )
    print_property(
        "Virtual memory management supported:",
        fmt_yes_no(props.virtual_memory_management_supported),
        indent=4,
    )
    print_property(
        "Generic compression supported:",
        fmt_yes_no(props.generic_compression_supported),
        indent=4,
    )

    # Perf ratio (single vs double)
    print_property(
        "Single/double precision perf ratio:",
        props.single_to_double_precision_perf_ratio,
        indent=4,
    )


def print_device_info(dev_id, device, verbose=False):
    """
    Print detailed information for a single CUDA device.
    Uses device.properties (cuda.core) for most fields; cuda.bindings for
    runtime version and global memory (not yet in high-level API).

    When ``verbose`` is True, appends the long-tail property list to the
    per-device output.
    """
    device.set_current()
    props = device.properties

    print()
    print(f"Device {dev_id}: {device.name}")

    # cuda.bindings workaround: runtime version not in cuda.core
    driver_major, driver_minor = system.get_user_mode_driver_version()
    err, runtime_version = cudart.cudaRuntimeGetVersion()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Failed to get CUDA runtime version: {err}")
    runtime_major = runtime_version // 1000
    runtime_minor = (runtime_version % 1000) // 10

    print_property(
        "CUDA Driver Version / Runtime Version",
        f"{driver_major}.{driver_minor} / {runtime_major}.{runtime_minor}",
    )
    print_property(
        "CUDA Capability Major/Minor version number:",
        f"{props.compute_capability_major}.{props.compute_capability_minor}",
    )

    # cuda.bindings workaround: global memory (free/total) not in device.properties
    err, _free_mem, total_mem_bytes = cuda.cuMemGetInfo()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get memory info: {err}")
    print_property("Total amount of global memory:", fmt_bytes(total_mem_bytes))

    sm_cores = convert_sm_ver_to_cores(props.compute_capability_major, props.compute_capability_minor)
    total_cores = sm_cores * props.multiprocessor_count
    print_property(
        f"({props.multiprocessor_count:3d}) Multiprocessors, ({sm_cores:3d}) CUDA Cores/MP:",
        f"{total_cores} CUDA Cores",
    )

    print_property("GPU Max Clock rate:", fmt_hz(props.clock_rate))
    print_property("Memory Clock rate:", f"{props.memory_clock_rate * 1e-3:.0f} Mhz")
    print_property("Memory Bus Width:", f"{props.global_memory_bus_width}-bit")
    if props.l2_cache_size > 0:
        print_property("L2 Cache Size:", f"{props.l2_cache_size} bytes")

    print_property(
        "Maximum Texture Dimension Size (x,y,z)",
        f"1D=({props.maximum_texture1d_width}), "
        f"2D=({props.maximum_texture2d_width}, {props.maximum_texture2d_height}), "
        f"3D=({props.maximum_texture3d_width}, {props.maximum_texture3d_height}, "
        f"{props.maximum_texture3d_depth})",
    )
    print_property(
        "Maximum Layered 1D Texture Size, (num) layers",
        f"1D=({props.maximum_texture1d_layered_width}), {props.maximum_texture1d_layered_layers} layers",
    )
    print_property(
        "Maximum Layered 2D Texture Size, (num) layers",
        f"2D=({props.maximum_texture2d_layered_width}, "
        f"{props.maximum_texture2d_layered_height}), "
        f"{props.maximum_texture2d_layered_layers} layers",
    )

    print_property("Total amount of constant memory:", f"{props.total_constant_memory} bytes")
    print_property(
        "Total amount of shared memory per block:",
        f"{props.max_shared_memory_per_block} bytes",
    )
    print_property(
        "Total shared memory per multiprocessor:",
        f"{props.max_shared_memory_per_multiprocessor} bytes",
    )
    print_property("Total number of registers available per block:", props.max_registers_per_block)

    print_property("Warp size:", props.warp_size)
    print_property(
        "Maximum number of threads per multiprocessor:",
        props.max_threads_per_multiprocessor,
    )
    print_property("Maximum number of threads per block:", props.max_threads_per_block)
    print_property(
        "Max dimension size of a thread block (x,y,z):",
        f"({props.max_block_dim_x}, {props.max_block_dim_y}, {props.max_block_dim_z})",
    )
    print_property(
        "Max dimension size of a grid size    (x,y,z):",
        f"({props.max_grid_dim_x}, {props.max_grid_dim_y}, {props.max_grid_dim_z})",
    )
    print_property("Maximum memory pitch:", f"{props.max_pitch} bytes")
    print_property("Texture alignment:", f"{props.texture_alignment} bytes")

    print_property(
        "Concurrent copy and kernel execution:",
        f"{fmt_yes_no(props.gpu_overlap)} with {props.async_engine_count} copy engine(s)",
    )
    print_property("Run time limit on kernels:", fmt_yes_no(props.kernel_exec_timeout))

    print_property("Integrated GPU sharing Host Memory:", fmt_yes_no(props.integrated))
    print_property(
        "Support host page-locked memory mapping:",
        fmt_yes_no(props.can_map_host_memory),
    )
    print_property("Device has ECC support:", "Enabled" if props.ecc_enabled else "Disabled")
    if platform.system() == "Windows":
        mode = "TCC (Tesla Compute Cluster Driver)" if props.tcc_driver else "WDDM (Windows Display Driver Model)"
        print_property("CUDA Device Driver Mode (TCC or WDDM):", mode)

    print_property(
        "Device supports Unified Addressing (UVA):",
        fmt_yes_no(props.unified_addressing),
    )
    print_property("Device supports Managed Memory:", fmt_yes_no(props.managed_memory))
    print_property(
        "Device supports Compute Preemption:",
        fmt_yes_no(props.compute_preemption_supported),
    )
    print_property("Supports Cooperative Kernel Launch:", fmt_yes_no(props.cooperative_launch))

    print_property(
        "Device PCI Domain ID / Bus ID / location ID:",
        f"{props.pci_domain_id} / {props.pci_bus_id} / {props.pci_device_id}",
    )
    compute_modes = {
        0: ("Default (multiple host threads can use cudaSetDevice() with device simultaneously)"),
        1: ("Exclusive (only one host thread in one process is able to use cudaSetDevice() with this device)"),
        2: "Prohibited (no host thread can use cudaSetDevice() with this device)",
        3: ("Exclusive Process (many threads in one process is able to use cudaSetDevice() with this device)"),
    }
    print_property("Compute Mode:", "")
    print(f"     < {compute_modes.get(props.compute_mode, 'Unknown')} >")

    if verbose:
        print_verbose_extras(props)


def print_p2p_access_info(devices):
    """
    Print peer-to-peer access information for multi-GPU systems.

    Parameters
    ----------
    devices : tuple of Device
        Tuple of CUDA device objects
    """
    print()
    print("Peer-to-Peer (P2P) access support:")
    for i, dev_i in enumerate(devices):
        for j, dev_j in enumerate(devices):
            if i == j:
                continue
            try:
                can_access = dev_i.can_access_peer(dev_j)
                print(f"> Peer access from {dev_i.name} (GPU{i}) -> {dev_j.name} (GPU{j}) : {fmt_yes_no(can_access)}")
            except Exception as e:
                print(f"Warning: Could not check peer access between device {i} and {j}: {e}")


def query_devices(show_p2p=True, verbose=False):
    """
    Query and display information about all CUDA devices.

    Parameters
    ----------
    show_p2p : bool
        Whether to show peer-to-peer access information (default: True)
    verbose : bool
        When True, include the long-tail property dump for each device
        (default: False).

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        print("[CUDA Device Query using CUDA Core API]")
        devices = Device.get_all_devices()
    except Exception as e:
        print(f"Error: Failed to get devices: {e}")
        import traceback

        traceback.print_exc()
        return False

    if len(devices) == 0:
        print("There are no available device(s) that support CUDA")
        return True

    print(f"Detected {len(devices)} CUDA Capable device(s)")

    for dev_id, device in enumerate(devices):
        try:
            print_device_info(dev_id, device, verbose=verbose)
        except Exception as e:
            print(f"Error: Failed to get information for device {dev_id}: {e}")
            import traceback

            traceback.print_exc()
            return False

    if show_p2p and len(devices) >= 2:
        print_p2p_access_info(devices)

    return True


def main():
    """
    Main entry point for the device query sample.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Query CUDA Device Properties using CUDA Core API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-p2p", action="store_true", help="Skip peer-to-peer access information")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also dump the long-tail Device.properties fields (surfaces, GPUDirect flags, NUMA, etc.)",
    )

    args = parser.parse_args()

    success = query_devices(show_p2p=not args.no_p2p, verbose=args.verbose)

    if success:
        print("\nDone")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
