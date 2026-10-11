# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates the batched memory copy API (copy_batch) for
# performing multiple async memory transfers in a single driver call. It
# covers homogeneous batches (all copies share one CopyOptions),
# heterogeneous batches (per-copy attributes), and verifies equivalence
# with sequential Buffer.copy_to calls.
#
# Requires CUDA 13+ (cuMemcpyBatchAsync is not available on CUDA 12).
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core"]
# ///

import ctypes
import sys

from cuda.core import Device, Host, LegacyPinnedMemoryResource, ManagedMemoryResource
from cuda.core.utils import CopyOptions, MemcpySrcAccessOrder, copy_batch


def readback(any_buf, pinned_mr, *, stream):
    """Copy a buffer to a new pinned buffer and return the bytes."""
    host_buf = pinned_mr.allocate(any_buf.size)
    any_buf.copy_to(host_buf, stream=stream)
    stream.sync()

    ptr = ctypes.cast(int(host_buf.handle), ctypes.POINTER(ctypes.c_byte))
    data = ctypes.string_at(ptr, host_buf.size)
    host_buf.close()
    return data


def main(dev: Device):
    dev.set_current()
    stream = dev.create_stream()
    pinned_mr = LegacyPinnedMemoryResource()
    device_mr = dev.memory_resource

    num_copies = 4
    buf_size = 4096

    # ---- Allocate source (pinned) and destination (device) buffers ----------

    srcs = []
    dsts = []
    for i in range(num_copies):
        src = pinned_mr.allocate(buf_size)
        dst = device_mr.allocate(buf_size, stream=stream)

        # Fill each source with a distinct byte pattern so we can verify
        fill_byte = (i + 1) % 256
        src.fill(fill_byte, stream=stream)

        srcs.append(src)
        dsts.append(dst)

    # ---- 1. Homogeneous batch: all copies share a single CopyOptions -----

    print("1. Homogeneous batched H2D copy...", file=sys.stderr)

    options = CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM)
    copy_batch(stream, srcs, dsts, options=options)

    for i, dst in enumerate(dsts):
        expected_byte = (i + 1) % 256
        data = readback(dst, pinned_mr, stream=stream)
        assert all(b == expected_byte for b in data), f"Copy {i}: expected byte {expected_byte}, got {data[:8]!r}..."

    print("   All copies verified.", file=sys.stderr)

    # ---- 2. Equivalence with sequential Buffer.copy_to ----------------------

    print("2. Verifying batched == sequential copy_to...", file=sys.stderr)

    # Re-fill sources with new patterns
    for i, src in enumerate(srcs):
        src.fill((i + 100) % 256, stream=stream)

    # Sequential path: individual copy_to calls
    seq_dsts = [device_mr.allocate(buf_size, stream=stream) for _ in range(num_copies)]
    for src, dst in zip(srcs, seq_dsts):
        src.copy_to(dst, stream=stream)

    # Batched path: single copy_batch call
    batch_dsts = [device_mr.allocate(buf_size, stream=stream) for _ in range(num_copies)]
    copy_batch(stream, srcs, batch_dsts, options=CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM))

    # Compare results
    for i in range(num_copies):
        seq_data = readback(seq_dsts[i], pinned_mr, stream=stream)
        batch_data = readback(batch_dsts[i], pinned_mr, stream=stream)
        assert seq_data == batch_data, f"Copy {i}: sequential and batched results differ"

    print("   Batched and sequential results match.", file=sys.stderr)

    # ---- 3. Heterogeneous batch: per-copy attributes ------------------------
    #
    # src_access_order controls how the driver accesses source memory:
    #   STREAM           - source read respects stream ordering (pinned/device memory)
    #   DURING_API_CALL  - source read during the API call itself (ephemeral host ptrs)
    #   ANY              - driver picks best strategy (pageable or HMM-backed memory)

    print("3. Heterogeneous batch with per-copy attributes...", file=sys.stderr)

    per_copy_options = [
        CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
        CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
        CopyOptions(src_access_order=MemcpySrcAccessOrder.STREAM),
        CopyOptions(src_access_order=MemcpySrcAccessOrder.ANY),
    ]
    hetero_dsts = [device_mr.allocate(buf_size, stream=stream) for _ in range(num_copies)]
    copy_batch(stream, srcs, hetero_dsts, options=per_copy_options)

    for i in range(num_copies):
        expected_byte = (i + 100) % 256
        data = readback(hetero_dsts[i], pinned_mr, stream=stream)
        assert all(b == expected_byte for b in data), f"Heterogeneous copy {i}: expected byte {expected_byte}"

    print("   Heterogeneous batch verified.", file=sys.stderr)

    # ---- 4. Location hints with managed memory ------------------------------
    #
    # When copying managed-memory buffers, src_location_hint and
    # dst_location_hint tell the driver where the data currently lives and
    # where it is going, enabling optimized transfer paths.

    print("4. Batched copy with location hints (managed memory)...", file=sys.stderr)

    managed_mr = ManagedMemoryResource()
    managed_srcs = [managed_mr.allocate(buf_size, stream=stream) for _ in range(2)]
    managed_dsts = [managed_mr.allocate(buf_size, stream=stream) for _ in range(2)]

    for i, src in enumerate(managed_srcs):
        src.fill((i + 200) % 256, stream=stream)

    hint_options = CopyOptions(
        src_access_order=MemcpySrcAccessOrder.STREAM,
        src_location_hint=dev,
        dst_location_hint=Host(),
    )
    copy_batch(stream, managed_srcs, managed_dsts, options=hint_options)

    for i in range(2):
        expected_byte = (i + 200) % 256
        data = readback(managed_dsts[i], pinned_mr, stream=stream)
        assert all(b == expected_byte for b in data), f"Managed copy {i}: expected byte {expected_byte}"

    print("   Location-hinted batch verified.", file=sys.stderr)

    # ---- Cleanup ------------------------------------------------------------

    all_bufs = srcs + dsts + seq_dsts + batch_dsts + hetero_dsts + managed_srcs + managed_dsts
    for buf in all_bufs:
        buf.close(stream)
    stream.close()

    print("Batched memcpy example completed!")


if __name__ == "__main__":
    main(Device(0))
