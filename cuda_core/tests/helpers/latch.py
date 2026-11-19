# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest

import helpers
from cuda.core import (
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)


class LatchKernel:
    """
    Manages a kernel that blocks stream progress until released.

    The kernel gives up after ``timeout_sec`` so that a test that never releases
    it cannot hang. The timeout is counted in GPU clock cycles at the peak clock
    rate, so it can last longer in wall-clock time. A test that expects to
    release the latch should check :attr:`timed_out` after synchronizing: if the
    latch ran into its timeout, the release protocol did not drive the test.
    """

    def __init__(self, device, timeout_sec=60):
        if helpers.CUDA_INCLUDE_PATH is None:
            pytest.skip("need CUDA header")
        code = """
               #include <cuda/atomic>

               extern "C"
               __global__ void latch(int* val, unsigned long long timeout_cycles) {
                   cuda::atomic_ref<int, cuda::thread_scope_system> signal{val[0]};
                   cuda::atomic_ref<int, cuda::thread_scope_system> timed_out{val[1]};

                   // Start time
                   unsigned long long start = clock64();

                   while (true) {
                       // Check if signal is set
                       if (signal.load(cuda::memory_order_relaxed)) {
                           break;
                       }

                       // Check for timeout
                       if (clock64() - start >= timeout_cycles) {
                           timed_out.store(1, cuda::memory_order_relaxed);
                           break;  // Timeout reached
                       }

                       // Avoid 100% spin.
                       __nanosleep(1000000); // 1 ms
                   }
               }
               """
        program_options = ProgramOptions(
            std="c++17",
            arch=f"sm_{''.join(f'{i}' for i in device.compute_capability)}",
            include_path=helpers.CCCL_INCLUDE_PATHS,
        )
        prog = Program(code, code_type="c++", options=program_options)
        mod = prog.compile(target_type="cubin")
        self.kernel = mod.get_kernel("latch")

        mr = LegacyPinnedMemoryResource()
        self.buffer = mr.allocate(8)
        self.busy_wait_flag[0] = 0
        self.busy_wait_flag[1] = 0
        clock_rate_hz = device.properties.clock_rate * 1000
        self.timeout_cycles = int(timeout_sec * clock_rate_hz)

    def launch(self, stream):
        """Launch the latch kernel, blocking stream progress via busy waiting."""
        self.busy_wait_flag[1] = 0
        config = LaunchConfig(grid=1, block=1)
        launch(stream, config, self.kernel, int(self.buffer.handle), self.timeout_cycles)

    def release(self):
        """Release the latch, allowing stream progress."""
        self.busy_wait_flag[0] = 1

    @property
    def timed_out(self):
        """Whether the last launch ended by timeout rather than release; valid after the stream syncs."""
        return bool(self.busy_wait_flag[1])

    @property
    def busy_wait_flag(self):
        return ctypes.cast(int(self.buffer.handle), ctypes.POINTER(ctypes.c_int32))
