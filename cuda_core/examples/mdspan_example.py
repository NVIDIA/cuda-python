# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to write a C++ kernel that takes cuda::std::mdspan
# as kernel arguments, JIT-compile it using cuda.core.experimental.Program,
# and prepare input/output CuPy arrays to launch this kernel and verify the result.
#
# NOTE: This is a skeleton/demonstration code that is not yet runnable.
# It is designed to guide the cuda.core design by exploring how mdspan layout
# information should be handled on both the host and device sides.
#
# The example covers three scenarios:
# 1. 2D input/output arrays in C-order (row-major)
# 2. 2D input/output arrays in F-order (column-major)
# 3. 2D input/output arrays with strided access (second axis skipped by 1 step)
#
# ################################################################################

import os, sys
import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch


# prepare include
cuda_path = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME"))
if cuda_path is None:
    print("this demo requires a valid CUDA_PATH environment variable set", file=sys.stderr)
    sys.exit(0)
cuda_include = os.path.join(cuda_path, "include")
assert os.path.isdir(cuda_include)
include_path = [cuda_include]
cccl_include = os.path.join(cuda_include, "cccl")
if os.path.isdir(cccl_include):
    include_path.insert(0, cccl_include)


# ################################################################################
# C++ Kernel Code with cuda::std::mdspan
# ################################################################################

# This kernel performs element-wise addition on 2D arrays using mdspan.
# mdspan provides a multi-dimensional view over contiguous or strided memory.
code = """
#include <cuda/std/mdspan>

// Kernel for element-wise addition of 2D arrays
// Template parameters:
//   T: element type (e.g., float, double)
//   LayoutPolicy: layout policy (e.g., layout_right for C-order, layout_left for F-order)
template<typename T, typename LayoutPolicy>
__global__ void mdspan_add_2d(
    cuda::std::mdspan<const T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, LayoutPolicy> input1,
    cuda::std::mdspan<const T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, LayoutPolicy> input2,
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, LayoutPolicy> output
) {
    // Calculate global thread indices
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we're within bounds
    if (row < input1.extent(0) && col < input1.extent(1)) {
        // Perform element-wise addition
        // mdspan handles the layout internally
        output(row, col) = input1(row, col) + input2(row, col);
    }
}

// Kernel variant for strided mdspan with custom layout
template<typename T>
__global__ void mdspan_add_2d_strided(
    cuda::std::mdspan<const T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_stride> input1,
    cuda::std::mdspan<const T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_stride> input2,
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_stride> output
) {
    // Calculate global thread indices
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we're within bounds
    if (row < input1.extent(0) && col < input1.extent(1)) {
        // Perform element-wise addition
        output(row, col) = input1(row, col) + input2(row, col);
    }
}
"""


# ################################################################################
# Helper Functions (To Be Implemented)
# ################################################################################

def prepare_mdspan_args_c_order(arr, dtype, shape):
    """
    Prepare mdspan arguments for C-order (row-major) 2D array.
    
    TODO: Determine the exact structure of mdspan on device side:
    - What information needs to be passed? (data pointer, extents, strides?)
    - How should the layout be represented?
    - What is the correct argument passing convention?
    
    Parameters
    ----------
    arr : cupy.ndarray
        Input CuPy array
    dtype : numpy.dtype
        Data type of the array
    shape : tuple
        Shape of the array (rows, cols)
    
    Returns
    -------
    tuple
        Arguments to pass to the kernel (needs investigation)
    """
    # Placeholder: This needs to be determined based on mdspan layout
    # For C-order (row-major): layout_right in mdspan
    # Possible arguments: pointer, extent0, extent1, stride0, stride1?
    data_ptr = arr.data.ptr
    rows, cols = shape
    # TODO: Determine if we need to pass strides explicitly
    # For C-order: stride0 = cols, stride1 = 1
    return (data_ptr, rows, cols)  # Placeholder return


def prepare_mdspan_args_f_order(arr, dtype, shape):
    """
    Prepare mdspan arguments for F-order (column-major) 2D array.
    
    TODO: Determine the exact structure of mdspan on device side:
    - What information needs to be passed for F-order layout?
    - How does layout_left differ from layout_right in argument passing?
    
    Parameters
    ----------
    arr : cupy.ndarray
        Input CuPy array
    dtype : numpy.dtype
        Data type of the array
    shape : tuple
        Shape of the array (rows, cols)
    
    Returns
    -------
    tuple
        Arguments to pass to the kernel (needs investigation)
    """
    # Placeholder: This needs to be determined based on mdspan layout
    # For F-order (column-major): layout_left in mdspan
    # Possible arguments: pointer, extent0, extent1, stride0, stride1?
    data_ptr = arr.data.ptr
    rows, cols = shape
    # TODO: Determine if we need to pass strides explicitly
    # For F-order: stride0 = 1, stride1 = rows
    return (data_ptr, rows, cols)  # Placeholder return


def prepare_mdspan_args_strided(arr, dtype, shape, strides):
    """
    Prepare mdspan arguments for strided 2D array with custom layout.
    
    TODO: Determine the exact structure of mdspan with layout_stride:
    - How to pass stride information to the kernel?
    - What is the argument structure for layout_stride mdspan?
    
    Parameters
    ----------
    arr : cupy.ndarray
        Input CuPy array
    dtype : numpy.dtype
        Data type of the array
    shape : tuple
        Shape of the array (rows, cols)
    strides : tuple
        Strides in bytes for each dimension
    
    Returns
    -------
    tuple
        Arguments to pass to the kernel (needs investigation)
    """
    # Placeholder: This needs to be determined based on mdspan layout
    # For custom strides: layout_stride in mdspan
    data_ptr = arr.data.ptr
    rows, cols = shape
    # Convert byte strides to element strides
    stride0 = strides[0] // arr.itemsize
    stride1 = strides[1] // arr.itemsize
    # TODO: Determine the correct argument structure for layout_stride
    return (data_ptr, rows, cols, stride0, stride1)  # Placeholder return


# ################################################################################
# Example 1: C-order (row-major) 2D arrays
# ################################################################################

def example_c_order():
    """Demonstrate mdspan with C-order (row-major) arrays."""
    print("=" * 70)
    print("Example 1: C-order (row-major) 2D arrays")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program with C++17 or later for mdspan support
    program_options = ProgramOptions(
        std="c++17",  # mdspan requires C++17 or later
        arch=f"sm_{dev.arch}",
        include_path=include_path,
    )
    prog = Program(code, code_type="c++", options=program_options)
    
    # Compile the kernel for float type with layout_right (C-order)
    # TODO: Determine the correct template instantiation syntax
    kernel_name = "mdspan_add_2d<float, cuda::std::layout_right>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare input/output arrays in C-order
    dtype = cp.float32
    shape = (128, 256)  # rows x cols
    rng = cp.random.default_rng()
    
    # Create C-order arrays explicitly
    input1 = rng.random(shape, dtype=dtype)
    input2 = rng.random(shape, dtype=dtype)
    output = cp.empty(shape, dtype=dtype, order='C')
    
    # Verify arrays are in C-order
    assert input1.flags['C_CONTIGUOUS']
    assert input2.flags['C_CONTIGUOUS']
    assert output.flags['C_CONTIGUOUS']
    
    dev.sync()  # Sync CuPy stream
    
    # TODO: Prepare mdspan kernel arguments
    # This is the main unknown: how to pass mdspan from host to device
    # Possible approaches:
    # 1. Pass pointer + extents + strides separately
    # 2. Pass a structure that matches mdspan layout
    # 3. Use a helper wrapper that constructs mdspan on device
    
    # Placeholder argument preparation
    args_input1 = prepare_mdspan_args_c_order(input1, dtype, shape)
    args_input2 = prepare_mdspan_args_c_order(input2, dtype, shape)
    args_output = prepare_mdspan_args_c_order(output, dtype, shape)
    
    # Prepare launch configuration
    block = (16, 16)  # 2D thread block
    grid = ((shape[1] + block[0] - 1) // block[0],  # cols
            (shape[0] + block[1] - 1) // block[1])  # rows
    config = LaunchConfig(grid=grid, block=block)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args_input1, *args_input2, *args_output)
    # s.sync()
    
    # Verify result
    # expected = input1 + input2
    # assert cp.allclose(output, expected)
    
    print("C-order example prepared (not executed)")
    print(f"  Input1 shape: {input1.shape}, strides: {input1.strides}, order: C")
    print(f"  Input2 shape: {input2.shape}, strides: {input2.strides}, order: C")
    print(f"  Output shape: {output.shape}, strides: {output.strides}, order: C")
    print(f"  Launch grid: {grid}, block: {block}")
    print()


# ################################################################################
# Example 2: F-order (column-major) 2D arrays
# ################################################################################

def example_f_order():
    """Demonstrate mdspan with F-order (column-major) arrays."""
    print("=" * 70)
    print("Example 2: F-order (column-major) 2D arrays")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{dev.arch}",
        include_path=include_path,
    )
    prog = Program(code, code_type="c++", options=program_options)
    
    # Compile the kernel for float type with layout_left (F-order)
    kernel_name = "mdspan_add_2d<float, cuda::std::layout_left>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare input/output arrays in F-order
    dtype = cp.float32
    shape = (128, 256)  # rows x cols
    rng = cp.random.default_rng()
    
    # Create F-order arrays explicitly
    input1 = cp.asfortranarray(rng.random(shape, dtype=dtype))
    input2 = cp.asfortranarray(rng.random(shape, dtype=dtype))
    output = cp.empty(shape, dtype=dtype, order='F')
    
    # Verify arrays are in F-order
    assert input1.flags['F_CONTIGUOUS']
    assert input2.flags['F_CONTIGUOUS']
    assert output.flags['F_CONTIGUOUS']
    
    dev.sync()  # Sync CuPy stream
    
    # TODO: Prepare mdspan kernel arguments for F-order
    args_input1 = prepare_mdspan_args_f_order(input1, dtype, shape)
    args_input2 = prepare_mdspan_args_f_order(input2, dtype, shape)
    args_output = prepare_mdspan_args_f_order(output, dtype, shape)
    
    # Prepare launch configuration
    block = (16, 16)
    grid = ((shape[1] + block[0] - 1) // block[0],
            (shape[0] + block[1] - 1) // block[1])
    config = LaunchConfig(grid=grid, block=block)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args_input1, *args_input2, *args_output)
    # s.sync()
    
    # Verify result
    # expected = input1 + input2
    # assert cp.allclose(output, expected)
    
    print("F-order example prepared (not executed)")
    print(f"  Input1 shape: {input1.shape}, strides: {input1.strides}, order: F")
    print(f"  Input2 shape: {input2.shape}, strides: {input2.strides}, order: F")
    print(f"  Output shape: {output.shape}, strides: {output.strides}, order: F")
    print(f"  Launch grid: {grid}, block: {block}")
    print()


# ################################################################################
# Example 3: Strided arrays (second axis with step 2, i.e., arr[:, ::2])
# ################################################################################

def example_strided():
    """Demonstrate mdspan with strided arrays."""
    print("=" * 70)
    print("Example 3: Strided arrays (second axis skipped by 1 step)")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{dev.arch}",
        include_path=include_path,
    )
    prog = Program(code, code_type="c++", options=program_options)
    
    # Compile the kernel for float type with layout_stride
    kernel_name = "mdspan_add_2d_strided<float>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare input/output arrays with strided views
    dtype = cp.float32
    base_shape = (128, 512)  # Base array shape
    rng = cp.random.default_rng()
    
    # Create base arrays in C-order
    base_input1 = rng.random(base_shape, dtype=dtype)
    base_input2 = rng.random(base_shape, dtype=dtype)
    base_output = cp.empty(base_shape, dtype=dtype, order='C')
    
    # Create strided views: skip every other element in second axis
    # arr[:, ::2] means: take all rows, every 2nd column
    input1 = base_input1[:, ::2]
    input2 = base_input2[:, ::2]
    output = base_output[:, ::2]
    
    # Check the resulting shapes and strides
    # Shape should be (128, 256) - half the columns
    # Strides will be different from contiguous arrays
    assert input1.shape == (128, 256)
    assert input2.shape == (128, 256)
    assert output.shape == (128, 256)
    
    dev.sync()  # Sync CuPy stream
    
    print(f"  Strided view shape: {input1.shape}")
    print(f"  Strided view strides (bytes): {input1.strides}")
    print(f"  Base array strides (bytes): {base_input1.strides}")
    print(f"  Stride ratio: {input1.strides[1] / dtype().itemsize} elements")
    
    # TODO: Prepare mdspan kernel arguments for strided layout
    args_input1 = prepare_mdspan_args_strided(input1, dtype, input1.shape, input1.strides)
    args_input2 = prepare_mdspan_args_strided(input2, dtype, input2.shape, input2.strides)
    args_output = prepare_mdspan_args_strided(output, dtype, output.shape, output.strides)
    
    # Prepare launch configuration
    block = (16, 16)
    grid = ((input1.shape[1] + block[0] - 1) // block[0],
            (input1.shape[0] + block[1] - 1) // block[1])
    config = LaunchConfig(grid=grid, block=block)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args_input1, *args_input2, *args_output)
    # s.sync()
    
    # Verify result
    # expected = input1 + input2
    # assert cp.allclose(output, expected)
    
    print("Strided example prepared (not executed)")
    print(f"  Input1 shape: {input1.shape}, strides: {input1.strides}")
    print(f"  Input2 shape: {input2.shape}, strides: {input2.strides}")
    print(f"  Output shape: {output.shape}, strides: {output.strides}")
    print(f"  Launch grid: {grid}, block: {block}")
    print()


# ################################################################################
# Main execution
# ################################################################################

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CUDA mdspan Example Skeleton")
    print("=" * 70)
    print()
    print("This is a skeleton/demonstration code to guide cuda.core design.")
    print("The main question to answer:")
    print("  How does mdspan layout look on the device side?")
    print()
    print("Key unknowns:")
    print("  1. What arguments to pass to kernels with mdspan parameters?")
    print("  2. How to represent different layouts (C-order, F-order, strided)?")
    print("  3. What is the ABI/calling convention for mdspan arguments?")
    print()
    
    # Run the three examples
    example_c_order()
    example_f_order()
    example_strided()
    
    print("=" * 70)
    print("All examples prepared successfully!")
    print("=" * 70)
