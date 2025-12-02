# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core import (
    LaunchConfig,
    Program,
    ProgramOptions,
    launch,
)


class NanosleepKernel:
    """
    Manages a kernel that sleeps for a specified duration using clock64().
    """

    def __init__(self, device, sleep_duration_ms: int = 20):
        """
        Initialize the nanosleep kernel.

        Args:
            device: CUDA device to compile the kernel for
            sleep_duration_ms: Duration to sleep in milliseconds (default: 20)
        """
        code = f"""
        extern "C"
        __global__ void nanosleep_kernel() {{
            // The maximum sleep duration is approximately 1 millisecond.
            unsigned int one_ms = 1000000U;
            for (unsigned int i = 0; i < {sleep_duration_ms}; ++i) {{
                __nanosleep(one_ms);
            }}
        }}
        """
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        prog = Program(code, code_type="c++", options=program_options)
        mod = prog.compile("cubin")
        self.kernel = mod.get_kernel("nanosleep_kernel")

    def launch(self, stream):
        """Launch the nanosleep kernel on the given stream."""
        config = LaunchConfig(grid=1, block=1)
        launch(stream, config, self.kernel)
