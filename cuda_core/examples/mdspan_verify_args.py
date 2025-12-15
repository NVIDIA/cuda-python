# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to verify mdspan kernel arguments passed from the host
# using mdspan accessors and printf. This helps understand what data is actually
# being passed to the kernel for different layout types.
#
# NOTE: This is a skeleton/demonstration code that is not yet runnable.
# It is designed to help investigate the mdspan device-side representation.
#
# The example covers three scenarios:
# 1. C-order (layout_right) - prints pointer, extents
# 2. F-order (layout_left) - prints pointer, extents
# 3. Strided (layout_stride) - prints pointer, extents, and explicit strides
#
# ################################################################################

import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# ################################################################################
# C++ Kernel Code for Verifying mdspan Arguments
# ################################################################################

# Verification kernels that print mdspan properties using printf
code_verify = """
#include <cuda/std/mdspan>
#include <cstdio>

// Kernel to verify layout_right (C-order) mdspan arguments
template<typename T>
__global__ void verify_mdspan_layout_right(
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_right> arr
) {
    // Only thread 0 prints to avoid cluttered output
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("=== layout_right (C-order) mdspan ===\\n");
        printf("Data pointer: %p\\n", arr.data_handle());
        printf("Extent 0 (rows): %zu\\n", arr.extent(0));
        printf("Extent 1 (cols): %zu\\n", arr.extent(1));
        printf("Size: %zu\\n", arr.size());
        
        // For layout_right, strides are implicit but we can query them
        printf("Stride 0: %zu\\n", arr.stride(0));
        printf("Stride 1: %zu\\n", arr.stride(1));
        
        // Verify memory layout: for layout_right (C-order)
        // stride(0) should equal extent(1), stride(1) should be 1
        printf("Expected stride(0) = extent(1): %s\\n", 
               (arr.stride(0) == arr.extent(1)) ? "PASS" : "FAIL");
        printf("Expected stride(1) = 1: %s\\n", 
               (arr.stride(1) == 1) ? "PASS" : "FAIL");
        
        // Test element access
        if (arr.extent(0) > 0 && arr.extent(1) > 0) {
            printf("First element arr(0,0): %f\\n", static_cast<float>(arr(0, 0)));
        }
    }
}

// Kernel to verify layout_left (F-order) mdspan arguments
template<typename T>
__global__ void verify_mdspan_layout_left(
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_left> arr
) {
    // Only thread 0 prints to avoid cluttered output
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("=== layout_left (F-order) mdspan ===\\n");
        printf("Data pointer: %p\\n", arr.data_handle());
        printf("Extent 0 (rows): %zu\\n", arr.extent(0));
        printf("Extent 1 (cols): %zu\\n", arr.extent(1));
        printf("Size: %zu\\n", arr.size());
        
        // For layout_left, strides are implicit but we can query them
        printf("Stride 0: %zu\\n", arr.stride(0));
        printf("Stride 1: %zu\\n", arr.stride(1));
        
        // Verify memory layout: for layout_left (F-order)
        // stride(0) should be 1, stride(1) should equal extent(0)
        printf("Expected stride(0) = 1: %s\\n", 
               (arr.stride(0) == 1) ? "PASS" : "FAIL");
        printf("Expected stride(1) = extent(0): %s\\n", 
               (arr.stride(1) == arr.extent(0)) ? "PASS" : "FAIL");
        
        // Test element access
        if (arr.extent(0) > 0 && arr.extent(1) > 0) {
            printf("First element arr(0,0): %f\\n", static_cast<float>(arr(0, 0)));
        }
    }
}

// Kernel to verify layout_stride mdspan arguments
template<typename T>
__global__ void verify_mdspan_layout_stride(
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_stride> arr
) {
    // Only thread 0 prints to avoid cluttered output
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("=== layout_stride mdspan ===\\n");
        printf("Data pointer: %p\\n", arr.data_handle());
        printf("Extent 0 (rows): %zu\\n", arr.extent(0));
        printf("Extent 1 (cols): %zu\\n", arr.extent(1));
        printf("Size: %zu\\n", arr.size());
        
        // For layout_stride, strides are stored explicitly
        printf("Stride 0 (explicit): %zu\\n", arr.stride(0));
        printf("Stride 1 (explicit): %zu\\n", arr.stride(1));
        
        // The mapping can be queried
        printf("Required span size: %zu\\n", arr.mapping().required_span_size());
        
        // Test element access
        if (arr.extent(0) > 0 && arr.extent(1) > 0) {
            printf("First element arr(0,0): %f\\n", static_cast<float>(arr(0, 0)));
            if (arr.extent(1) > 1) {
                printf("Second element arr(0,1): %f\\n", static_cast<float>(arr(0, 1)));
            }
        }
    }
}
"""


# ################################################################################
# Helper Functions (To Be Implemented)
# ################################################################################

def prepare_mdspan_args_layout_right(arr, dtype, shape):
    """
    Prepare mdspan arguments for layout_right (C-order) 2D array.
    
    TODO: Determine the exact structure needed for kernel launch.
    Based on the source code, layout_right::mapping stores {extents} and
    mdspan stores {data_handle, mapping, accessor}.
    
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
    data_ptr = arr.data.ptr
    rows, cols = shape
    # TODO: Determine exact argument structure
    return (data_ptr, rows, cols)


def prepare_mdspan_args_layout_left(arr, dtype, shape):
    """
    Prepare mdspan arguments for layout_left (F-order) 2D array.
    
    TODO: Determine the exact structure needed for kernel launch.
    Based on the source code, layout_left::mapping stores {extents} and
    mdspan stores {data_handle, mapping, accessor}.
    
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
    data_ptr = arr.data.ptr
    rows, cols = shape
    # TODO: Determine exact argument structure
    return (data_ptr, rows, cols)


def prepare_mdspan_args_layout_stride(arr, dtype, shape, strides):
    """
    Prepare mdspan arguments for layout_stride 2D array.
    
    TODO: Determine the exact structure needed for kernel launch.
    Based on the source code, layout_stride::mapping stores {extents, stride_array}
    and mdspan stores {data_handle, mapping, accessor}.
    
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
    data_ptr = arr.data.ptr
    rows, cols = shape
    # Convert byte strides to element strides
    stride0 = strides[0] // arr.itemsize
    stride1 = strides[1] // arr.itemsize
    # TODO: Determine exact argument structure
    return (data_ptr, rows, cols, stride0, stride1)


# ################################################################################
# Example 1: Verify layout_right (C-order) mdspan
# ################################################################################

def verify_layout_right():
    """Verify layout_right (C-order) mdspan arguments."""
    print("=" * 70)
    print("Verifying layout_right (C-order) mdspan")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program with C++17 or later for mdspan support
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{dev.arch}",
    )
    prog = Program(code_verify, code_type="c++", options=program_options)
    
    # Compile the verification kernel for float type with layout_right
    kernel_name = "verify_mdspan_layout_right<float>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare test array in C-order
    dtype = cp.float32
    shape = (4, 8)  # Small array for testing
    
    # Create C-order array with known values
    arr = cp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape, order='C')
    
    # Verify array is in C-order
    assert arr.flags['C_CONTIGUOUS']
    
    print(f"Array shape: {arr.shape}")
    print(f"Array strides (bytes): {arr.strides}")
    print(f"Array strides (elements): ({arr.strides[0]//arr.itemsize}, {arr.strides[1]//arr.itemsize})")
    print(f"First element: {arr[0, 0]}")
    print()
    
    dev.sync()  # Sync CuPy stream
    
    # TODO: Prepare mdspan kernel arguments
    args = prepare_mdspan_args_layout_right(arr, dtype, shape)
    
    # Launch kernel (single thread is enough for verification)
    config = LaunchConfig(grid=1, block=1)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args)
    # s.sync()
    
    print("Verification kernel prepared (not executed)")
    print()


# ################################################################################
# Example 2: Verify layout_left (F-order) mdspan
# ################################################################################

def verify_layout_left():
    """Verify layout_left (F-order) mdspan arguments."""
    print("=" * 70)
    print("Verifying layout_left (F-order) mdspan")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{dev.arch}",
    )
    prog = Program(code_verify, code_type="c++", options=program_options)
    
    # Compile the verification kernel for float type with layout_left
    kernel_name = "verify_mdspan_layout_left<float>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare test array in F-order
    dtype = cp.float32
    shape = (4, 8)  # Small array for testing
    
    # Create F-order array with known values
    arr = cp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape, order='F')
    
    # Verify array is in F-order
    assert arr.flags['F_CONTIGUOUS']
    
    print(f"Array shape: {arr.shape}")
    print(f"Array strides (bytes): {arr.strides}")
    print(f"Array strides (elements): ({arr.strides[0]//arr.itemsize}, {arr.strides[1]//arr.itemsize})")
    print(f"First element: {arr[0, 0]}")
    print()
    
    dev.sync()  # Sync CuPy stream
    
    # TODO: Prepare mdspan kernel arguments
    args = prepare_mdspan_args_layout_left(arr, dtype, shape)
    
    # Launch kernel (single thread is enough for verification)
    config = LaunchConfig(grid=1, block=1)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args)
    # s.sync()
    
    print("Verification kernel prepared (not executed)")
    print()


# ################################################################################
# Example 3: Verify layout_stride mdspan
# ################################################################################

def verify_layout_stride():
    """Verify layout_stride mdspan arguments."""
    print("=" * 70)
    print("Verifying layout_stride mdspan")
    print("=" * 70)
    
    # Setup device and stream
    dev = Device()
    dev.set_current()
    s = dev.create_stream()
    
    # Prepare program
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{dev.arch}",
    )
    prog = Program(code_verify, code_type="c++", options=program_options)
    
    # Compile the verification kernel for float type with layout_stride
    kernel_name = "verify_mdspan_layout_stride<float>"
    mod = prog.compile("cubin", name_expressions=(kernel_name,))
    ker = mod.get_kernel(kernel_name)
    
    # Prepare test array with strided view
    dtype = cp.float32
    base_shape = (4, 16)  # Base array shape
    
    # Create base array in C-order with known values
    base_arr = cp.arange(base_shape[0] * base_shape[1], dtype=dtype).reshape(base_shape, order='C')
    
    # Create strided view: skip every other element in second axis
    # arr[:, ::2] means: take all rows, every 2nd column
    arr = base_arr[:, ::2]
    
    print(f"Base array shape: {base_arr.shape}")
    print(f"Base array strides (bytes): {base_arr.strides}")
    print(f"Strided view shape: {arr.shape}")
    print(f"Strided view strides (bytes): {arr.strides}")
    print(f"Strided view strides (elements): ({arr.strides[0]//arr.itemsize}, {arr.strides[1]//arr.itemsize})")
    print(f"First element: {arr[0, 0]}")
    print(f"Second element: {arr[0, 1]}")
    print()
    
    dev.sync()  # Sync CuPy stream
    
    # TODO: Prepare mdspan kernel arguments
    args = prepare_mdspan_args_layout_stride(arr, dtype, arr.shape, arr.strides)
    
    # Launch kernel (single thread is enough for verification)
    config = LaunchConfig(grid=1, block=1)
    
    # TODO: Launch kernel with proper mdspan arguments
    # launch(s, config, ker, *args)
    # s.sync()
    
    print("Verification kernel prepared (not executed)")
    print()


# ################################################################################
# Main execution
# ################################################################################

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CUDA mdspan Argument Verification Example")
    print("=" * 70)
    print()
    print("This example demonstrates how to verify mdspan kernel arguments")
    print("using printf to inspect:")
    print("  - Data pointer address")
    print("  - Extents (dimensions)")
    print("  - Strides (for layout_stride)")
    print()
    print("Key investigation points:")
    print("  1. What is the actual parameter passing mechanism?")
    print("  2. How are extents encoded in the kernel arguments?")
    print("  3. How are strides encoded for layout_stride?")
    print()
    
    # Run the three verification examples
    verify_layout_right()
    verify_layout_left()
    verify_layout_stride()
    
    print("=" * 70)
    print("All verification examples prepared successfully!")
    print("=" * 70)
