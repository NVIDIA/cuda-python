# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use CUDA graphs to capture and execute
# multiple kernel launches with minimal overhead. The graph performs a
# sequence of vector operations: add, multiply, and subtract.
#
# ################################################################################

import time

import cupy as cp
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch


def main():
    # CUDA kernels for vector operations
    code = """
    template<typename T>
    __global__ void vector_add(const T* A, const T* B, T* C, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
            C[i] = A[i] + B[i];
        }
    }

    template<typename T>
    __global__ void vector_multiply(const T* A, const T* B, T* C, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
            C[i] = A[i] * B[i];
        }
    }

    template<typename T>
    __global__ void vector_subtract(const T* A, const T* B, T* C, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
            C[i] = A[i] - B[i];
        }
    }
    """

    # Initialize device and stream
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    # tell CuPy to use our stream as the current stream:
    cp.cuda.ExternalStream(int(stream.handle)).use()

    # Compile the program
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin", name_expressions=("vector_add<float>", "vector_multiply<float>", "vector_subtract<float>")
    )

    # Get kernel functions
    add_kernel = mod.get_kernel("vector_add<float>")
    multiply_kernel = mod.get_kernel("vector_multiply<float>")
    subtract_kernel = mod.get_kernel("vector_subtract<float>")

    # Prepare data
    size = 1000000
    dtype = cp.float32

    # Create input arrays
    rng = cp.random.default_rng(42)  # Fixed seed for reproducibility
    a = rng.random(size, dtype=dtype)
    b = rng.random(size, dtype=dtype)
    c = rng.random(size, dtype=dtype)

    # Create output arrays
    result1 = cp.empty_like(a)
    result2 = cp.empty_like(a)
    result3 = cp.empty_like(a)

    # Prepare launch configuration
    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    config = LaunchConfig(grid=grid_size, block=block_size)

    # Sync before graph capture
    dev.sync()

    print("Building CUDA graph...")

    # Build the graph
    graph_builder = stream.create_graph_builder()
    graph_builder.begin_building()

    # Add multiple kernel launches to the graph
    # Kernel 1: result1 = a + b
    launch(graph_builder, config, add_kernel, a.data.ptr, b.data.ptr, result1.data.ptr, cp.uint64(size))

    # Kernel 2: result2 = result1 * c
    launch(graph_builder, config, multiply_kernel, result1.data.ptr, c.data.ptr, result2.data.ptr, cp.uint64(size))

    # Kernel 3: result3 = result2 - a
    launch(graph_builder, config, subtract_kernel, result2.data.ptr, a.data.ptr, result3.data.ptr, cp.uint64(size))

    # Complete the graph
    graph = graph_builder.end_building().complete()

    print("Graph built successfully!")

    # Upload the graph to the stream
    graph.upload(stream)

    # Execute the entire graph with a single launch
    print("Executing graph...")
    start_time = time.time()
    graph.launch(stream)
    stream.sync()
    end_time = time.time()

    graph_execution_time = end_time - start_time
    print(f"Graph execution time: {graph_execution_time:.6f} seconds")

    # Verify results
    expected_result1 = a + b
    expected_result2 = expected_result1 * c
    expected_result3 = expected_result2 - a

    print("Verifying results...")
    assert cp.allclose(result1, expected_result1, rtol=1e-5, atol=1e-5), "Result 1 mismatch"
    assert cp.allclose(result2, expected_result2, rtol=1e-5, atol=1e-5), "Result 2 mismatch"
    assert cp.allclose(result3, expected_result3, rtol=1e-5, atol=1e-5), "Result 3 mismatch"
    print("All results verified successfully!")

    # Demonstrate performance benefit by running the same operations without graph
    print("\nRunning same operations without graph for comparison...")

    # Reset results
    result1.fill(0)
    result2.fill(0)
    result3.fill(0)

    start_time = time.time()

    # Individual kernel launches
    launch(stream, config, add_kernel, a.data.ptr, b.data.ptr, result1.data.ptr, cp.uint64(size))
    launch(stream, config, multiply_kernel, result1.data.ptr, c.data.ptr, result2.data.ptr, cp.uint64(size))
    launch(stream, config, subtract_kernel, result2.data.ptr, a.data.ptr, result3.data.ptr, cp.uint64(size))

    stream.sync()
    end_time = time.time()

    individual_execution_time = end_time - start_time
    print(f"Individual kernel execution time: {individual_execution_time:.6f} seconds")

    # Calculate speedup
    speedup = individual_execution_time / graph_execution_time
    print(f"Graph provides {speedup:.2f}x speedup")

    # Verify results again
    assert cp.allclose(result1, expected_result1, rtol=1e-5, atol=1e-5), "Result 1 mismatch"
    assert cp.allclose(result2, expected_result2, rtol=1e-5, atol=1e-5), "Result 2 mismatch"
    assert cp.allclose(result3, expected_result3, rtol=1e-5, atol=1e-5), "Result 3 mismatch"

    cp.cuda.Stream.null.use()  # reset CuPy's current stream to the null stream

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
