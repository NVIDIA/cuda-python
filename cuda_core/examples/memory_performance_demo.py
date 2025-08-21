#!/usr/bin/env python3

"""
Performance demonstration for DeviceMemoryResource release threshold optimization.

This script demonstrates the performance improvement achieved by setting a higher
release threshold for the memory pool used by DeviceMemoryResource.

The optimization prevents the memory pool from immediately releasing memory back
to the OS when there are no active allocations, which can cause significant
performance overhead for subsequent allocations.
"""

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from cuda.core.experimental import Device, DeviceMemoryResource


def benchmark_allocations(mr, num_allocations=1000, size=1024):
    """Benchmark allocation/deallocation performance."""
    print(f"Benchmarking {num_allocations} allocations of {size} bytes...")

    start_time = time.perf_counter()

    for _ in range(num_allocations):
        buffer = mr.allocate(size)
        buffer.close()  # Immediate deallocation

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / num_allocations * 1_000_000  # microseconds

    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per allocation: {avg_time:.2f} μs")
    return total_time


def main():
    """Demonstrate the performance benefit of release threshold optimization."""
    print("=== DeviceMemoryResource Performance Demo ===")
    print()

    device = Device()
    device.set_current()

    print(f"Using device: {device.device_id}")
    print()

    # Create DeviceMemoryResource (with release threshold optimization)
    mr = DeviceMemoryResource(device.device_id)
    print("Created DeviceMemoryResource with release threshold optimization")

    # Warm up
    print("Warming up...")
    for _ in range(100):
        buffer = mr.allocate(1024)
        buffer.close()

    # Benchmark
    print("\nBenchmarking allocation performance...")
    benchmark_allocations(mr, num_allocations=1000, size=1024)

    print("\nNote: With the release threshold optimization, subsequent allocations")
    print("should be significantly faster as memory is retained in the pool rather")
    print("than being released back to the OS and re-allocated from the OS.")


if __name__ == "__main__":
    main()
