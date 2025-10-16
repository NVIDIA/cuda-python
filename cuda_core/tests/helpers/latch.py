# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import (
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
import helpers
import ctypes

class LatchKernel:
    """
    Manages a kernel that blocks progress until released.
    """

    def __init__(self, device):
        code = """
               #include <cuda/atomic>

               extern "C"
               __global__ void latch(int* val) {
                   cuda::atomic_ref<int, cuda::thread_scope_system> signal{*val};
                   while (true) {
                       if (signal.load(cuda::memory_order_relaxed)) {
                           break;
                       }
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
        self.buffer = mr.allocate(4)
        self.busy_wait_flag[0] = 0

    def launch(self, stream):
        config = LaunchConfig(grid=1, block=1)
        launch(stream, config, self.kernel, self.busy_wait_flag_address)

    def release(self):
        self.busy_wait_flag[0] = 1

    @property
    def busy_wait_flag_address(self):
        return int(self.buffer.handle)

    @property
    def busy_wait_flag(self):
        return ctypes.cast(self.busy_wait_flag_address, ctypes.POINTER(ctypes.c_int32))

    def close(self):
       buffer = getattr(self, 'buffer', None)
       if buffer is not None:
           buffer.close()

    def __del__(self):
        self.close()


