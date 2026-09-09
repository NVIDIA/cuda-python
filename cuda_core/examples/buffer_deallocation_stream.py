# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example transfers a buffer from a producer stream to a consumer stream.
# An event orders the consumer after the producer. The buffer then records the
# consumer stream for its eventual deallocation.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core"]
# ///

import ctypes

from cuda.core import Device, LegacyPinnedMemoryResource


def produce_data(device, stream, size, value):
    """Allocate and fill a buffer on the producer stream."""
    buffer = device.allocate(size, stream=stream)
    buffer.fill(value, stream=stream)
    ready = stream.record()
    return buffer, ready


def consume_data(buffer, ready, output, stream):
    """Submit consumer work and transfer the deallocation stream."""
    stream.wait(ready)
    buffer.set_deallocation_stream(stream)
    buffer.copy_to(output, stream=stream)


def main():
    device = Device()
    device.set_current()
    producer_stream = device.create_stream()
    consumer_stream = device.create_stream()
    pinned_mr = LegacyPinnedMemoryResource()

    size = 4096
    value = 42
    buffer = None
    ready = None
    output = None

    try:
        output = pinned_mr.allocate(size)
        buffer, ready = produce_data(device, producer_stream, size, value)
        consume_data(buffer, ready, output, consumer_stream)

        # No stream argument is needed. The buffer now records consumer_stream.
        # The free operation runs after the copy on that stream.
        buffer.close()
        buffer = None
        consumer_stream.sync()

        result = ctypes.string_at(int(output.handle), output.size)
        assert result == bytes([value]) * size
        print("Buffer deallocation stream transfer completed.")
    finally:
        if buffer is not None:
            buffer.close()
        if output is not None:
            output.close()
        if ready is not None:
            ready.close()
        consumer_stream.close()
        producer_stream.close()


if __name__ == "__main__":
    main()
