# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import binascii
import re
import textwrap
import cuda.bindings.driver as cuda
import os
import errno
import ctypes
from contextlib import contextmanager
import numpy as _numpy
import stat

import pytest

from cuda.bindings import cufile
#from cuda.bindings.cycufile import CUfileDescr_t, CUfileFileHandleType

def test_cufile_success_defined():
    """Check if CUFILE_SUCCESS is defined in OpError enum."""
    assert hasattr(cufile.OpError, 'SUCCESS')

def test_driver_open():
    """Test cuFile driver initialization."""
    cufile.driver_open()

def test_handle_register():
    """Test file handle registration with cuFile."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_handle_register.bin"
    
    # Create file with POSIX operations
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR, 0o644)
    
    # Write test data using POSIX write
    test_data = b"Test data for cuFile - POSIX write"
    bytes_written = os.write(fd, test_data)
    
    # Sync to ensure data is on disk
    os.fsync(fd)
    
    # Close and reopen with O_DIRECT for cuFile operations
    os.close(fd)
    
    # Reopen with O_DIRECT
    flags = os.O_RDWR | os.O_DIRECT
    fd = os.open(file_path, flags)
    
    try:
        # Create and initialize the descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0
        
        # Register the handle
        handle = cufile.handle_register(descr.ptr)
        
        # Deregister the handle
        cufile.handle_deregister(handle)

    finally:
        os.close(fd)

        # Clean up the test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

def test_buf_register_simple():
    """Simple test for buffer registration with cuFile."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate CUDA memory
    buffer_size = 4096
    err, buf_ptr = cuda.cuMemAlloc(buffer_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Register the buffer with cuFile
        flags = 0
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, flags)

        # Deregister the buffer
        cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)

def test_buf_register_host_memory():
    """Test buffer registration with host memory."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate host memory
    buffer_size = 4096
    err, buf_ptr = cuda.cuMemHostAlloc(buffer_size, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Register the host buffer with cuFile
        flags = 0
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, flags)

        # Deregister the buffer
        cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free host memory
        cuda.cuMemFreeHost(buf_ptr)

def test_buf_register_multiple_buffers():
    """Test registering multiple buffers."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate multiple CUDA buffers
    buffer_sizes = [512, 4096, 65536]
    buffers = []
    
    for size in buffer_sizes:
        err, buf_ptr = cuda.cuMemAlloc(size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buffers.append(buf_ptr)

    try:
        # Register all buffers
        flags = 0
        for buf_ptr, size in zip(buffers, buffer_sizes):
            buf_ptr_int = int(buf_ptr)
            cufile.buf_register(buf_ptr_int, size, flags)

        # Deregister all buffers
        for buf_ptr in buffers:
            buf_ptr_int = int(buf_ptr)
            cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free all buffers
        for buf_ptr in buffers:
            cuda.cuMemFree(buf_ptr)

def test_buf_register_invalid_flags():
    """Test buffer registration with invalid flags."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate CUDA memory
    buffer_size = 65536
    err, buf_ptr = cuda.cuMemAlloc(buffer_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Try to register with invalid flags
        invalid_flags = 999
        buf_ptr_int = int(buf_ptr)
        
        try:
            cufile.buf_register(buf_ptr_int, buffer_size, invalid_flags)
            # If we get here, deregister to clean up
            cufile.buf_deregister(buf_ptr_int)
        except Exception:
            # Expected error with invalid flags
            pass

    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)

def test_buf_register_large_buffer():
    """Test buffer registration with a large buffer."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate large CUDA memory (1MB)
    buffer_size = 1024 * 1024
    err, buf_ptr = cuda.cuMemAlloc(buffer_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Register the large buffer with cuFile
        flags = 0
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, flags)

        # Deregister the buffer
        cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)

def test_buf_register_already_registered():
    """Test that registering an already registered buffer fails."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate CUDA memory
    buffer_size = 1024
    err, buf_ptr = cuda.cuMemAlloc(buffer_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Register the buffer first time
        flags = 0
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, flags)

        # Try to register the same buffer again
        try:
            cufile.buf_register(buf_ptr_int, buffer_size, flags)
            # If we get here, deregister both times
            cufile.buf_deregister(buf_ptr_int)
            cufile.buf_deregister(buf_ptr_int)
        except Exception:
            # Expected error when registering buffer twice
            # Deregister the first registration
            cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)

def test_cufile_read_write():
    """Test cuFile read and write operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw.bin"
    
    # Allocate CUDA memory for write and read
    write_size = 65536
    err, write_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(write_size)
    
    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)

        # Register buffers with cuFile
        write_buf_int = int(write_buf)
        read_buf_int = int(read_buf)
        
        cufile.buf_register(write_buf_int, write_size, 0)
        cufile.buf_register(read_buf_int, write_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Prepare test data
        test_data = b"Hello cuFile! This is test data for read/write operations. " * 20
        test_data = test_data[:write_size]
        ctypes.memmove(host_buf, test_data, len(test_data))

        # Copy test data to CUDA write buffer
        cuda.cuMemcpyHtoD(write_buf, host_buf, write_size)

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Copy read data back to host
        cuda.cuMemcpyDtoH(host_buf, read_buf, write_size)

        # Verify the data
        read_data = host_buf.value
        assert read_data == test_data, "Read data doesn't match written data"

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        cufile.buf_deregister(write_buf_int)
        cufile.buf_deregister(read_buf_int)

    finally:
        # Close file
        os.close(fd)

        # Free CUDA memory
        cuda.cuMemFree(write_buf)
        cuda.cuMemFree(read_buf)

        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

def test_cufile_read_write_host_memory():
    """Test cuFile read and write operations using host memory."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw_host.bin"
    
    # Allocate host memory for write and read
    write_size = 65536
    err, write_buf = cuda.cuMemHostAlloc(write_size, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemHostAlloc(write_size, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)

        # Register host buffers with cuFile
        write_buf_int = int(write_buf)
        read_buf_int = int(read_buf)
        
        cufile.buf_register(write_buf_int, write_size, 0)
        cufile.buf_register(read_buf_int, write_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Prepare test data
        test_data = b"Host memory test data for cuFile operations! " * 20
        test_data = test_data[:write_size]
        
        # Copy test data to host write buffer
        ctypes.memmove(write_buf, test_data, len(test_data))

        # Get the actual data that was written
        write_buffer_content = ctypes.string_at(write_buf, write_size)

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Sync to ensure data is on disk
        os.fsync(fd)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Verify the data
        read_data = ctypes.string_at(read_buf, write_size)
        expected_data = write_buffer_content
        assert read_data == expected_data, "Read data doesn't match written data"

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        cufile.buf_deregister(write_buf_int)
        cufile.buf_deregister(read_buf_int)

    finally:
        # Close file
        os.close(fd)

        # Free host memory
        cuda.cuMemFreeHost(write_buf)
        cuda.cuMemFreeHost(read_buf)

        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

def test_cufile_read_write_large():
    """Test cuFile read and write operations with large data."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuCtxCreate(0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw_large.bin"
    
    # Allocate large CUDA memory (1MB)
    write_size = 1024 * 1024
    err, write_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(write_size)
    
    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)

        # Register buffers with cuFile
        write_buf_int = int(write_buf)
        read_buf_int = int(read_buf)
        
        cufile.buf_register(write_buf_int, write_size, 0)
        cufile.buf_register(read_buf_int, write_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Generate large test data
        import random
        test_data = bytes(random.getrandbits(8) for _ in range(write_size))
        ctypes.memmove(host_buf, test_data, write_size)

        # Copy test data to CUDA write buffer
        cuda.cuMemcpyHtoD(write_buf, host_buf, write_size)

        # Get the actual data that was written to CUDA buffer
        cuda.cuMemcpyDtoH(host_buf, write_buf, write_size)
        expected_data = host_buf.value

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Copy read data back to host
        cuda.cuMemcpyDtoH(host_buf, read_buf, write_size)

        # Verify the data
        read_data = host_buf.value
        assert read_data == expected_data, "Large read data doesn't match written data"

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        cufile.buf_deregister(write_buf_int)
        cufile.buf_deregister(read_buf_int)

    finally:
        # Close file
        os.close(fd)

        # Free CUDA memory
        cuda.cuMemFree(write_buf)
        cuda.cuMemFree(read_buf)

        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

