# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inferior for cuda-gdb: compile with debug=True and launch, keep Program alive."""

from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

CODE = """
extern "C" __global__ void kernel() {
    int x = 0;
    x += 1;  // ISSUE_2422_SOURCE_LINE
}
"""


def main() -> None:
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    prog = Program(CODE, "c++", ProgramOptions(debug=True, arch=f"sm_{dev.arch}"))
    mod = prog.compile("cubin")
    k = mod.get_kernel("kernel")
    launch(stream, LaunchConfig(grid=1, block=1), k)
    stream.sync()


if __name__ == "__main__":
    main()
