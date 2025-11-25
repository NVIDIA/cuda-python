# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import (
    LaunchConfig,
    Program,
    ProgramOptions,
    launch,
)


class NanosleepKernel:
    """
    Manages a kernel that sleeps for a specified duration using clock64().
    """

    def __init__(self, device, sleep_duration_seconds=0.020):
        """
        Initialize the nanosleep kernel.

        Args:
            device: CUDA device to compile the kernel for
            sleep_duration_seconds: Duration to sleep in seconds (default: 0.020 for 20ms)
        """
        clock_rate_hz = device.properties.clock_rate * 1000
        sleep_cycles = int(sleep_duration_seconds * clock_rate_hz)
        code = f"""
        extern "C"
        __global__ void nanosleep_kernel() {{
            long long int start = clock64();
            while (clock64() - start < {sleep_cycles}) {{
                __nanosleep(1000000); // 1 ms yield to avoid 100% spin
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
