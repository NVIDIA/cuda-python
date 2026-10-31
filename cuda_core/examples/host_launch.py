# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example reads a pinned host buffer in a stream-ordered host function.
# The callback observes the first kernel's result before the second overwrites it.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core"]
# ///

import ctypes
from contextlib import closing

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, Program, host_launch, launch


def main():
    dev = Device(0)
    dev.set_current()
    size = 256
    code = r"""
    extern "C" __global__ void fill(unsigned char *buffer, unsigned long long value) {
        buffer[threadIdx.x] = static_cast<unsigned char>(value);
    }
    """
    arch = "sm_" + "".join(str(part) for part in dev.compute_capability)
    program = Program(code, code_type="c++", options={"arch": arch})
    module = program.compile("cubin")
    kernel = module.get_kernel("fill")
    config = LaunchConfig(grid=1, block=size)
    checksums = []

    with (
        closing(dev.create_stream()) as stream,
        dev.allocate(size, stream=stream) as device_buffer,
        LegacyPinnedMemoryResource().allocate(size) as host_buffer,
    ):

        def record_checksum():
            data = ctypes.string_at(int(host_buffer.handle), size)
            checksums.append(sum(data))

        launch(stream, config, kernel, device_buffer, ctypes.c_uint64(0x11))
        device_buffer.copy_to(host_buffer, stream=stream)
        host_launch(stream, record_checksum)
        launch(stream, config, kernel, device_buffer, ctypes.c_uint64(0x22))
        device_buffer.copy_to(host_buffer, stream=stream)
        stream.sync()

        assert checksums == [0x11 * size]
        assert ctypes.string_at(int(host_buffer.handle), size) == b"\x22" * size

    print("Stream-ordered host function observed the first kernel's checksum.")


if __name__ == "__main__":
    main()
