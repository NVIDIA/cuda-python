# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Programmatic Dependent Launch overlap tests."""

import helpers
import numpy as np
import pytest

from cuda.core import LaunchConfig, LegacyPinnedMemoryResource, Program, ProgramOptions, launch


def compile_pdl_overlap_kernels(device):
    """Compile primary/secondary kernels used to detect PDL same-stream overlap.

    The primary triggers programmatic launch completion then spins briefly looking
    for a flag written by the secondary. Seeing that flag proves both grids were
    resident at once. clock64 budgets are in GPU cycles: long enough for the
    secondary to boot, short enough for a unit test.

    Returns:
        (primary_kernel, secondary_kernel)
    """
    code = r"""
    #include <cuda_device_runtime_api.h>

    extern "C" __global__ void primary_kernel(int* secondary_started, int* overlapped) {
        cudaTriggerProgrammaticLaunchCompletion();

        const long long deadline = clock64() + 100000000LL;  // ~50ms @ ~2GHz
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (clock64() < deadline) {
                if (atomicAdd(secondary_started, 0) != 0) {
                    atomicExch(overlapped, 1);
                    return;
                }
                __nanosleep(1000);
            }
        }
    }

    extern "C" __global__ void secondary_kernel(int* secondary_started) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicExch(secondary_started, 1);
        }
    }
    """
    arch = "".join(f"{i}" for i in device.compute_capability)
    pro_opts = ProgramOptions(std="c++17", arch=f"sm_{arch}", include_path=helpers.CUDA_INCLUDE_PATH)
    prog = Program(code, code_type="c++", options=pro_opts)
    mod = prog.compile("cubin")
    return mod.get_kernel("primary_kernel"), mod.get_kernel("secondary_kernel")


def run_pdl_overlap_check(device, *, via_graph: bool = False):
    """Run the shared same-stream primary/secondary PDL overlap protocol.

    Both paths launch primary then secondary on one stream. Asserts no overlap
    without ``programmatic_stream_serialization``, then retries a few times with
    it enabled. Overlap is opportunistic → miss is xfail.

    Args:
        device: Current CUDA device (compute capability >= 9.0 required).
        via_graph: If True, stream-capture the same-stream launches into a CUDA
            graph and launch that graph; otherwise launch kernels directly.
    """
    if device.compute_capability < (9, 0):
        pytest.skip("Programmatic Dependent Launch requires compute capability >= 9.0")

    stream = device.create_stream(options={"nonblocking": True})
    primary, secondary = compile_pdl_overlap_kernels(device)

    mr = LegacyPinnedMemoryResource()
    secondary_started = np.from_dlpack(mr.allocate(4)).view(np.int32)
    overlapped = np.from_dlpack(mr.allocate(4)).view(np.int32)

    primary_cfg = LaunchConfig(grid=1, block=1)
    secondary_cfg = LaunchConfig(grid=1, block=1, programmatic_stream_serialization=True)
    secondary_serial_cfg = LaunchConfig(grid=1, block=1)

    def _run(secondary_launch_cfg: LaunchConfig) -> int:
        secondary_started[0] = 0
        overlapped[0] = 0
        if via_graph:
            gb = stream.create_graph_builder().begin_building()
            launch(gb, primary_cfg, primary, secondary_started.ctypes.data, overlapped.ctypes.data)
            launch(gb, secondary_launch_cfg, secondary, secondary_started.ctypes.data)
            graph = gb.end_building().complete()
            try:
                graph.launch(stream)
                stream.sync()
            finally:
                graph.close()
                gb.close()
        else:
            launch(stream, primary_cfg, primary, secondary_started.ctypes.data, overlapped.ctypes.data)
            launch(stream, secondary_launch_cfg, secondary, secondary_started.ctypes.data)
            stream.sync()
        return int(overlapped[0])

    path = "same-stream (graph)" if via_graph else "same-stream"
    assert _run(secondary_serial_cfg) == 0, (
        f"Expected no overlap when programmatic_stream_serialization is False ({path})"
    )

    saw_overlap = False
    for _ in range(5):
        if _run(secondary_cfg) == 1:
            saw_overlap = True
            break

    if not saw_overlap:
        # Overlap is never guaranteed by the driver, so a miss is reported as an
        # expected failure rather than turning a busy GPU into a red CI run.
        pytest.xfail(
            f"PDL (Programmatic Dependent Launch) {path} overlap was not observed. "
            "If this keeps xfailing in CI, manually re-check on a quiet Hopper+ GPU."
        )

    print(
        f"PDL {path} overlap verified on {device.name} compute capability {device.compute_capability}",
        flush=True,
    )
