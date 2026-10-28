# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example copies a device buffer into a NumPy array in chunks through two
# pinned staging slots. Each chunk is a device-to-pinned copy followed by
# HOST_COPY_FN, which copies the slot into the array without the GIL. Each slot
# has its own stream, so the copy into one slot overlaps the host copy out of
# the other, and stream order keeps a slot from being reused too early.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core", "numpy"]
# ///

from contextlib import closing

import numpy as np

from cuda.core import HOST_COPY_FN, Buffer, Device, LegacyPinnedMemoryResource, host_launch, pack_host_copy_args


def main():
    dev = Device(0)
    dev.set_current()
    chunk = 4 << 20
    total = 16 * chunk
    result = np.empty(total, dtype=np.uint8)
    pinned_mr = LegacyPinnedMemoryResource()

    with (
        closing(dev.create_stream()) as stream0,
        closing(dev.create_stream()) as stream1,
        dev.allocate(total, stream=stream0) as device_buffer,
        pinned_mr.allocate(chunk) as slot0,
        pinned_mr.allocate(chunk) as slot1,
    ):
        device_buffer.fill(0x5A, stream=stream0)
        stream1.wait(stream0)

        streams = (stream0, stream1)
        slots = (slot0, slot1)
        for i, offset in enumerate(range(0, total, chunk)):
            stream, slot = streams[i % 2], slots[i % 2]
            src = Buffer.from_handle(int(device_buffer.handle) + offset, chunk, owner=device_buffer)
            src.copy_to(slot, stream=stream)
            dst = memoryview(result)[offset : offset + chunk]
            host_launch(stream, HOST_COPY_FN, user_data=pack_host_copy_args(dst, slot, chunk))

        stream0.sync()
        stream1.sync()

    assert (result == 0x5A).all()
    print(f"Copied {total >> 20} MiB into a NumPy array through two pinned slots.")


if __name__ == "__main__":
    main()
