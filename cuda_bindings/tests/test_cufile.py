# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import errno
import logging
import os
import pathlib
import platform
import tempfile
from contextlib import suppress
from functools import cache

import pytest

import cuda.bindings.driver as cuda

# Configure logging to show INFO level and above
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,  # Override any existing logging configuration
)

try:
    from cuda.bindings import cufile
except ImportError:
    cufile = None


def platform_is_wsl():
    """Check if running on Windows Subsystem for Linux (WSL)."""
    return platform.system() == "Linux" and "microsoft" in pathlib.Path("/proc/version").read_text().lower()


if cufile is None:
    pytest.skip("skipping tests on Windows", allow_module_level=True)

if platform_is_wsl():
    pytest.skip("skipping cuFile tests on WSL", allow_module_level=True)


@pytest.fixture
def cufile_env_json():
    """Set CUFILE_ENV_PATH_JSON environment variable for async tests."""
    original_value = os.environ.get("CUFILE_ENV_PATH_JSON")

    # Get absolute path to cufile.json in the same directory as this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, "cufile.json")
    logging.info(f"Using cuFile config: {config_path}")
    assert os.path.isfile(config_path)
    os.environ["CUFILE_ENV_PATH_JSON"] = config_path
    yield

    # Restore original value or remove if it wasn't set
    if original_value is not None:
        os.environ["CUFILE_ENV_PATH_JSON"] = original_value
    else:
        os.environ.pop("CUFILE_ENV_PATH_JSON", None)


@cache
def cufileLibraryAvailable():
    """Check if cuFile library is available on the system."""
    try:
        # Try to get cuFile library version - this will fail if library is not available
        version = cufile.get_version()
        logging.info(f"cuFile library available, version: {version}")
        return True
    except Exception as e:
        logging.warning(f"cuFile library not available: {e}")
        return False


@cache
def cufileVersionLessThan(target):
    """Check if cuFile library version is less than target version."""
    try:
        # Get cuFile library version
        version = cufile.get_version()
        logging.info(f"cuFile library version: {version}")
        # Check if version is less than target
        if version < target:
            logging.warning(f"cuFile library version {version} is less than required {target}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error checking cuFile version: {e}")
        return True  # Assume old version if any error occurs


@cache
def isSupportedFilesystem():
    """Check if the current filesystem is supported (ext4 or xfs)."""
    try:
        # Try to get filesystem type from /proc/mounts
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mount_point = parts[1]
                    fs_type = parts[2]

                    # Check if current directory is under this mount point
                    current_dir = os.path.abspath(".")
                    if current_dir.startswith(mount_point):
                        fs_type_lower = fs_type.lower()
                        logging.info(f"Current filesystem type: {fs_type_lower}")
                        return fs_type_lower in ["ext4", "xfs"]

        # If we get here, we couldn't determine the filesystem type
        logging.warning("Could not determine filesystem type from /proc/mounts")
        return False
    except Exception as e:
        logging.error(f"Error checking filesystem type: {e}")
        return False


# Global skip condition for all tests if cuFile library is not available
pytestmark = pytest.mark.skipif(not cufileLibraryAvailable(), reason="cuFile library not available on this system")


def safe_decode_string(raw_value):
    """Safely decode a string value from ctypes buffer."""
    # Find null terminator if present
    null_pos = raw_value.find(b"\x00")
    if null_pos != -1:
        raw_value = raw_value[:null_pos]
    # Decode with error handling
    try:
        return raw_value.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        # If UTF-8 fails, try to decode as bytes
        return str(raw_value)


def test_cufile_success_defined():
    """Check if CUFILE_SUCCESS is defined in OpError enum."""
    assert hasattr(cufile.OpError, "SUCCESS")


def test_driver_open():
    """Test cuFile driver initialization."""
    cufile.driver_open()
    cufile.driver_close()


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_handle_register():
    """Test file handle registration with cuFile."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_handle_register.bin"

    # Create file with POSIX operations
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR, 0o600)

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
        with suppress(OSError):
            os.unlink(file_path)
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_simple():
    """Simple test for buffer registration with cuFile."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate CUDA memory
    buffer_size = 4096  # 4KB, aligned to 4096 bytes
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

        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_host_memory():
    """Test buffer registration with host memory."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate host memory
    buffer_size = 4096  # 4KB, aligned to 4096 bytes
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

        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_multiple_buffers():
    """Test registering multiple buffers."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate multiple CUDA buffers
    buffer_sizes = [4096, 16384, 65536]  # All aligned to 4096 bytes
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

        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_invalid_flags():
    """Test buffer registration with invalid flags."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
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

        with suppress(Exception):
            cufile.buf_register(buf_ptr_int, buffer_size, invalid_flags)
            # If we get here, deregister to clean up
            cufile.buf_deregister(buf_ptr_int)

    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)

        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_large_buffer():
    """Test buffer registration with a large buffer."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate large CUDA memory (1MB, aligned to 4096 bytes)
    buffer_size = 1024 * 1024  # 1MB, aligned to 4096 bytes (1048576 % 4096 == 0)
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
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def test_buf_register_already_registered():
    """Test that registering an already registered buffer fails."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Allocate CUDA memory
    buffer_size = 4096  # 4KB, aligned to 4096 bytes
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
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_read_write():
    """Test cuFile read and write operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw.bin"

    # Allocate CUDA memory for write and read
    write_size = 65536  # 64KB, aligned to 4096 bytes (65536 % 4096 == 0)
    err, write_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(write_size)

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

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
        test_string = b"Hello cuFile! This is test data for read/write operations. "
        test_string_len = len(test_string)
        repetitions = write_size // test_string_len
        test_data = test_string * repetitions
        test_data = test_data[:write_size]  # Ensure it fits exactly in buffer
        host_buf = ctypes.create_string_buffer(test_data, write_size)

        # Copy test data to CUDA write buffer
        cuda.cuMemcpyHtoDAsync(write_buf, host_buf, write_size, 0)
        cuda.cuStreamSynchronize(0)

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Copy read data back to host
        cuda.cuMemcpyDtoHAsync(host_buf, read_buf, write_size, 0)
        cuda.cuStreamSynchronize(0)

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
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_read_write_host_memory():
    """Test cuFile read and write operations using host memory."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw_host.bin"

    # Allocate host memory for write and read
    write_size = 65536  # 64KB, aligned to 4096 bytes (65536 % 4096 == 0)
    err, write_buf = cuda.cuMemHostAlloc(write_size, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemHostAlloc(write_size, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

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
        test_string = b"Host memory test data for cuFile operations! "
        test_string_len = len(test_string)
        repetitions = write_size // test_string_len
        test_data = test_string * repetitions
        test_data = test_data[:write_size]  # Ensure it fits exactly in buffer

        # Copy test data to host write buffer
        host_buf = ctypes.create_string_buffer(test_data, write_size)
        write_buf_content = ctypes.string_at(write_buf, write_size)

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Sync to ensure data is on disk
        os.fsync(fd)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Verify the data
        read_data = ctypes.string_at(read_buf, write_size)
        expected_data = write_buf_content
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
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_read_write_large():
    """Test cuFile read and write operations with large data."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_rw_large.bin"

    # Allocate large CUDA memory (1MB, aligned to 4096 bytes)
    write_size = 1024 * 1024  # 1MB, aligned to 4096 bytes (1048576 % 4096 == 0)
    err, write_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, read_buf = cuda.cuMemAlloc(write_size)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(write_size)

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

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
        host_buf = ctypes.create_string_buffer(test_data, write_size)

        # Copy test data to CUDA write buffer
        cuda.cuMemcpyHtoDAsync(write_buf, host_buf, write_size, 0)
        cuda.cuStreamSynchronize(0)

        # Get the actual data that was written to CUDA buffer
        cuda.cuMemcpyDtoHAsync(host_buf, write_buf, write_size, 0)
        cuda.cuStreamSynchronize(0)
        expected_data = host_buf.value

        # Write data using cuFile
        bytes_written = cufile.write(handle, write_buf_int, write_size, 0, 0)

        # Read data back using cuFile
        bytes_read = cufile.read(handle, read_buf_int, write_size, 0, 0)

        # Copy read data back to host
        cuda.cuMemcpyDtoHAsync(host_buf, read_buf, write_size, 0)
        cuda.cuStreamSynchronize(0)

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
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_write_async(cufile_env_json):
    """Test cuFile asynchronous write operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_write_async.bin"
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

    try:
        # Register file handle
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0
        handle = cufile.handle_register(descr.ptr)

        # Allocate and register device buffer
        buf_size = 65536  # 64KB, aligned to 4096 bytes (65536 % 4096 == 0)
        err, buf_ptr = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        cufile.buf_register(int(buf_ptr), buf_size, 0)

        # Create CUDA stream
        err, stream = cuda.cuStreamCreate(0)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register stream with cuFile
        cufile.stream_register(int(stream), 0)

        # Prepare test data in device buffer
        test_string = b"Async write test data for cuFile!"
        test_string_len = len(test_string)
        repetitions = buf_size // test_string_len
        test_data = test_string * repetitions
        test_data = test_data[:buf_size]  # Ensure it fits exactly in buffer
        host_buf = ctypes.create_string_buffer(test_data, buf_size)
        cuda.cuMemcpyHtoDAsync(buf_ptr, host_buf, buf_size, 0)
        cuda.cuStreamSynchronize(0)

        # Create parameter arrays for async write
        size_p = ctypes.c_size_t(buf_size)
        file_offset_p = ctypes.c_int64(0)
        buf_ptr_offset_p = ctypes.c_int64(0)
        bytes_written_p = ctypes.c_ssize_t(0)

        # Perform async write
        cufile.write_async(
            int(handle),
            int(buf_ptr),
            ctypes.addressof(size_p),
            ctypes.addressof(file_offset_p),
            ctypes.addressof(buf_ptr_offset_p),
            ctypes.addressof(bytes_written_p),
            int(stream),
        )

        # Synchronize stream to wait for completion
        cuda.cuStreamSynchronize(stream)

        # Verify bytes written
        assert bytes_written_p.value == buf_size, f"Expected {buf_size} bytes written, got {bytes_written_p.value}"

        # Deregister stream
        cufile.stream_deregister(int(stream))

        # Deregister and cleanup
        cufile.buf_deregister(int(buf_ptr))
        cufile.handle_deregister(handle)
        cuda.cuStreamDestroy(stream)
        cuda.cuMemFree(buf_ptr)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_read_async(cufile_env_json):
    """Test cuFile asynchronous read operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_read_async.bin"

    # First create and write test data without O_DIRECT
    fd_temp = os.open(file_path, os.O_CREAT | os.O_RDWR, 0o600)
    # Create test data that's aligned to 4096 bytes
    test_string = b"Async read test data for cuFile!"
    test_string_len = len(test_string)
    buf_size = 65536  # 64KB, aligned to 4096 bytes
    repetitions = buf_size // test_string_len
    test_data = test_string * repetitions
    test_data = test_data[:buf_size]  # Ensure exact 64KB
    os.write(fd_temp, test_data)
    os.fsync(fd_temp)
    os.close(fd_temp)

    # Now open with O_DIRECT for cuFile operations
    fd = os.open(file_path, os.O_RDWR | os.O_DIRECT)

    try:
        # Register file handle
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0
        handle = cufile.handle_register(descr.ptr)

        # Allocate and register device buffer
        buf_size = 65536  # 64KB, aligned to 4096 bytes (65536 % 4096 == 0)
        err, buf_ptr = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        cufile.buf_register(int(buf_ptr), buf_size, 0)

        # Create CUDA stream
        err, stream = cuda.cuStreamCreate(0)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register stream with cuFile
        cufile.stream_register(int(stream), 0)

        # Create parameter arrays for async read
        size_p = ctypes.c_size_t(buf_size)
        file_offset_p = ctypes.c_int64(0)
        buf_ptr_offset_p = ctypes.c_int64(0)
        bytes_read_p = ctypes.c_ssize_t(0)

        # Perform async read
        cufile.read_async(
            int(handle),
            int(buf_ptr),
            ctypes.addressof(size_p),
            ctypes.addressof(file_offset_p),
            ctypes.addressof(buf_ptr_offset_p),
            ctypes.addressof(bytes_read_p),
            int(stream),
        )

        # Synchronize stream to wait for completion
        cuda.cuStreamSynchronize(stream)

        # Verify bytes read
        assert bytes_read_p.value > 0, f"Expected bytes read, got {bytes_read_p.value}"

        # Copy read data back to host and verify
        host_buf = ctypes.create_string_buffer(buf_size)
        cuda.cuMemcpyDtoHAsync(host_buf, buf_ptr, buf_size, 0)
        cuda.cuStreamSynchronize(0)
        read_data = host_buf.value[: bytes_read_p.value]
        expected_data = test_data[: bytes_read_p.value]
        assert read_data == expected_data, "Read data doesn't match written data"

        # Deregister stream
        cufile.stream_deregister(int(stream))

        # Deregister and cleanup
        cufile.buf_deregister(int(buf_ptr))
        cufile.handle_deregister(handle)
        cuda.cuStreamDestroy(stream)
        cuda.cuMemFree(buf_ptr)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_cufile_async_read_write(cufile_env_json):
    """Test cuFile asynchronous read and write operations in sequence."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_cufile_async_rw.bin"
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

    try:
        # Register file handle
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0
        handle = cufile.handle_register(descr.ptr)

        # Allocate and register device buffers
        buf_size = 65536  # 64KB, aligned to 4096 bytes (65536 % 4096 == 0)
        err, write_buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        cufile.buf_register(int(write_buf), buf_size, 0)

        err, read_buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        cufile.buf_register(int(read_buf), buf_size, 0)

        # Create CUDA stream
        err, stream = cuda.cuStreamCreate(0)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register stream with cuFile
        cufile.stream_register(int(stream), 0)

        # Prepare test data in write buffer
        test_string = b"Async RW test data for cuFile!"
        test_string_len = len(test_string)
        repetitions = buf_size // test_string_len
        test_data = test_string * repetitions
        test_data = test_data[:buf_size]  # Ensure it fits exactly in buffer
        host_buf = ctypes.create_string_buffer(test_data, buf_size)
        cuda.cuMemcpyHtoDAsync(write_buf, host_buf, buf_size, 0)
        cuda.cuStreamSynchronize(0)

        # Create parameter arrays for async write
        write_size_p = ctypes.c_size_t(buf_size)
        write_file_offset_p = ctypes.c_int64(0)
        write_buf_ptr_offset_p = ctypes.c_int64(0)
        bytes_written_p = ctypes.c_ssize_t(0)

        # Perform async write
        cufile.write_async(
            int(handle),
            int(write_buf),
            ctypes.addressof(write_size_p),
            ctypes.addressof(write_file_offset_p),
            ctypes.addressof(write_buf_ptr_offset_p),
            ctypes.addressof(bytes_written_p),
            int(stream),
        )

        # Synchronize stream to wait for write completion
        cuda.cuStreamSynchronize(stream)

        # Verify bytes written
        assert bytes_written_p.value == buf_size, f"Expected {buf_size} bytes written, got {bytes_written_p.value}"

        # Create parameter arrays for async read
        read_size_p = ctypes.c_size_t(buf_size)
        read_file_offset_p = ctypes.c_int64(0)
        read_buf_ptr_offset_p = ctypes.c_int64(0)
        bytes_read_p = ctypes.c_ssize_t(0)

        # Perform async read
        cufile.read_async(
            int(handle),
            int(read_buf),
            ctypes.addressof(read_size_p),
            ctypes.addressof(read_file_offset_p),
            ctypes.addressof(read_buf_ptr_offset_p),
            ctypes.addressof(bytes_read_p),
            int(stream),
        )

        # Synchronize stream to wait for read completion
        cuda.cuStreamSynchronize(stream)

        # Verify bytes read
        assert bytes_read_p.value == buf_size, f"Expected {buf_size} bytes read, got {bytes_read_p.value}"

        # Copy read data back to host and verify
        host_buf = ctypes.create_string_buffer(buf_size)
        cuda.cuMemcpyDtoHAsync(host_buf, read_buf, buf_size, 0)
        cuda.cuStreamSynchronize(0)
        read_data = host_buf.value
        assert read_data == test_data, "Read data doesn't match written data"

        # Deregister stream
        cufile.stream_deregister(int(stream))

        # Deregister and cleanup
        cufile.buf_deregister(int(write_buf))
        cufile.buf_deregister(int(read_buf))
        cufile.handle_deregister(handle)
        cuda.cuStreamDestroy(stream)
        cuda.cuMemFree(write_buf)
        cuda.cuMemFree(read_buf)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_batch_io_basic():
    """Test basic batch IO operations with multiple read/write operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_batch_io.bin"

    # Allocate CUDA memory for multiple operations
    buf_size = 65536  # 64KB
    num_operations = 4

    buffers = []
    read_buffers = []  # Initialize read_buffers to avoid UnboundLocalError

    for i in range(num_operations):
        err, buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buffers.append(buf)

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(buf_size)

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

        # Register buffers with cuFile
        for buf in buffers:
            buf_int = int(buf)
            cufile.buf_register(buf_int, buf_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Set up batch IO
        batch_handle = cufile.batch_io_set_up(num_operations)

        # Create IOParams array for batch operations
        io_params = cufile.IOParams(num_operations)
        io_events = cufile.IOEvents(num_operations)

        # Prepare test data for each operation
        test_strings = [
            b"Batch operation 1 data for testing cuFile! ",
            b"Batch operation 2 data for testing cuFile! ",
            b"Batch operation 3 data for testing cuFile! ",
            b"Batch operation 4 data for testing cuFile! ",
        ]

        # Set up write operations
        for i in range(num_operations):
            # Prepare test data
            test_string = test_strings[i]
            test_string_len = len(test_string)
            repetitions = buf_size // test_string_len
            test_data = test_string * repetitions
            test_data = test_data[:buf_size]  # Ensure it fits exactly in buffer
            host_buf = ctypes.create_string_buffer(test_data, buf_size)

            # Copy test data to CUDA buffer
            cuda.cuMemcpyHtoDAsync(buffers[i], host_buf, buf_size, 0)
            cuda.cuStreamSynchronize(0)

            # Set up IOParams for this operation
            io_params[i].mode = cufile.BatchMode.BATCH  # Batch mode
            io_params[i].fh = handle
            io_params[i].opcode = cufile.Opcode.WRITE  # Write opcode
            io_params[i].cookie = i  # Use index as cookie for identification
            io_params[i].u.batch.dev_ptr_base = int(buffers[i])
            io_params[i].u.batch.file_offset = i * buf_size  # Sequential file offsets
            io_params[i].u.batch.dev_ptr_offset = 0
            io_params[i].u.batch.size_ = buf_size

        # Submit batch write operations
        cufile.batch_io_submit(batch_handle, num_operations, io_params.ptr, 0)

        # Get batch status
        min_nr = num_operations  # Wait for all operations to complete
        nr_completed = ctypes.c_uint(num_operations)  # Initialize to max operations posted
        timeout = ctypes.c_int(5000)  # 5 second timeout

        cufile.batch_io_get_status(
            batch_handle, min_nr, ctypes.addressof(nr_completed), io_events.ptr, ctypes.addressof(timeout)
        )

        # Verify all operations completed successfully
        assert nr_completed.value == num_operations, f"Expected {num_operations} operations, got {nr_completed.value}"

        # Collect all returned cookies
        returned_cookies = set()
        for i in range(num_operations):
            assert io_events[i].status == cufile.Status.COMPLETE, (
                f"Operation {i} failed with status {io_events[i].status}"
            )
            assert io_events[i].ret == buf_size, f"Expected {buf_size} bytes, got {io_events[i].ret} for operation {i}"
            returned_cookies.add(io_events[i].cookie)

        # Verify all expected cookies are present
        expected_cookies = set(range(num_operations))  # cookies 0, 1, 2, 3
        assert returned_cookies == expected_cookies, (
            f"Cookie mismatch. Expected {expected_cookies}, got {returned_cookies}"
        )

        # Now test batch read operations
        read_buffers = []
        for i in range(num_operations):
            err, buf = cuda.cuMemAlloc(buf_size)
            assert err == cuda.CUresult.CUDA_SUCCESS
            read_buffers.append(buf)
            buf_int = int(buf)
            cufile.buf_register(buf_int, buf_size, 0)

        # Create fresh io_events array for read operations
        io_events_read = cufile.IOEvents(num_operations)

        # Set up read operations
        for i in range(num_operations):
            io_params[i].mode = cufile.BatchMode.BATCH  # Batch mode
            io_params[i].fh = handle
            io_params[i].opcode = cufile.Opcode.READ  # Read opcode
            io_params[i].cookie = i + 100  # Different cookie for reads
            io_params[i].u.batch.dev_ptr_base = int(read_buffers[i])
            io_params[i].u.batch.file_offset = i * buf_size
            io_params[i].u.batch.dev_ptr_offset = 0
            io_params[i].u.batch.size_ = buf_size

        # Submit batch read operations
        cufile.batch_io_submit(batch_handle, num_operations, io_params.ptr, 0)

        # Get batch status for reads
        cufile.batch_io_get_status(
            batch_handle, min_nr, ctypes.addressof(nr_completed), io_events_read.ptr, ctypes.addressof(timeout)
        )

        # Verify read operations completed successfully
        assert nr_completed.value == num_operations, (
            f"Expected {num_operations} read operations, got {nr_completed.value}"
        )

        # Collect all returned cookies for read operations
        returned_cookies_read = set()
        for i in range(num_operations):
            assert io_events_read[i].status == cufile.Status.COMPLETE, (
                f"Operation {i} failed with status {io_events_read[i].status}"
            )
            assert io_events_read[i].ret == buf_size, (
                f"Expected {buf_size} bytes read, got {io_events_read[i].ret} for operation {i}"
            )
            returned_cookies_read.add(io_events_read[i].cookie)

        # Verify all expected cookies are present
        expected_cookies_read = set(range(100, 100 + num_operations))  # cookies 100, 101, 102, 103
        assert returned_cookies_read == expected_cookies_read, (
            f"Cookie mismatch. Expected {expected_cookies_read}, got {returned_cookies_read}"
        )

        # Verify the read data matches the written data
        for i in range(num_operations):
            # Copy read data back to host
            cuda.cuMemcpyDtoHAsync(host_buf, read_buffers[i], buf_size, 0)
            cuda.cuStreamSynchronize(0)
            read_data = host_buf.value

            # Prepare expected data
            test_string = test_strings[i]
            test_string_len = len(test_string)
            repetitions = buf_size // test_string_len
            expected_data = (test_string * repetitions)[:buf_size]

            assert read_data == expected_data, f"Read data doesn't match written data for operation {i}"

        # Clean up batch IO
        cufile.batch_io_destroy(batch_handle)

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        for buf in buffers + read_buffers:
            buf_int = int(buf)
            cufile.buf_deregister(buf_int)

    finally:
        # Close file
        os.close(fd)
        # Free CUDA memory
        for buf in buffers + read_buffers:
            cuda.cuMemFree(buf)
        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_batch_io_cancel():
    """Test batch IO cancellation."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_batch_cancel.bin"

    # Allocate CUDA memory
    buf_size = 4096  # 4KB, aligned to 4096 bytes
    num_operations = 2

    buffers = []
    for i in range(num_operations):
        err, buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buffers.append(buf)

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

        # Register buffers with cuFile
        for buf in buffers:
            buf_int = int(buf)
            cufile.buf_register(buf_int, buf_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Set up batch IO
        batch_handle = cufile.batch_io_set_up(num_operations)

        # Create IOParams array for batch operations
        io_params = cufile.IOParams(num_operations)

        # Set up write operations
        for i in range(num_operations):
            io_params[i].mode = cufile.BatchMode.BATCH  # Batch mode
            io_params[i].fh = handle
            io_params[i].opcode = cufile.Opcode.WRITE  # Write opcode
            io_params[i].cookie = i
            io_params[i].u.batch.dev_ptr_base = int(buffers[i])
            io_params[i].u.batch.file_offset = i * buf_size
            io_params[i].u.batch.dev_ptr_offset = 0
            io_params[i].u.batch.size_ = buf_size

        # Submit batch operations
        cufile.batch_io_submit(batch_handle, num_operations, io_params.ptr, 0)

        # Cancel the batch operations
        cufile.batch_io_cancel(batch_handle)

        # Clean up batch IO
        cufile.batch_io_destroy(batch_handle)

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        for buf in buffers:
            buf_int = int(buf)
            cufile.buf_deregister(buf_int)

    finally:
        # Close file
        os.close(fd)
        # Free CUDA memory
        for buf in buffers:
            cuda.cuMemFree(buf)
        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
def test_batch_io_large_operations():
    """Test batch IO with large buffer operations."""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Open cuFile driver
    cufile.driver_open()

    # Create test file
    file_path = "test_batch_large.bin"

    # Allocate large CUDA memory (1MB, aligned to 4096 bytes)
    buf_size = 1024 * 1024  # 1MB, aligned to 4096 bytes
    num_operations = 2

    write_buffers = []
    read_buffers = []
    all_buffers = []  # Initialize all_buffers to avoid UnboundLocalError

    for i in range(num_operations):
        err, buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        write_buffers.append(buf)

        err, buf = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        read_buffers.append(buf)

    # Allocate host memory for data verification
    host_buf = ctypes.create_string_buffer(buf_size)

    try:
        # Create file with O_DIRECT
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

        # Register all buffers with cuFile
        all_buffers = write_buffers + read_buffers
        for buf in all_buffers:
            buf_int = int(buf)
            cufile.buf_register(buf_int, buf_size, 0)

        # Create file descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register file handle
        handle = cufile.handle_register(descr.ptr)

        # Set up batch IO
        batch_handle = cufile.batch_io_set_up(num_operations * 2)  # 2 writes + 2 reads

        # Create IOParams array for batch operations
        io_params = cufile.IOParams(num_operations * 2)
        io_events = cufile.IOEvents(num_operations * 2)

        # Prepare test data
        test_strings = [
            b"Large batch operation 1 data for testing cuFile with 1MB buffers! ",
            b"Large batch operation 2 data for testing cuFile with 1MB buffers! ",
        ]

        # Prepare write data
        for i in range(num_operations):
            test_string = test_strings[i]
            test_string_len = len(test_string)
            repetitions = buf_size // test_string_len
            test_data = test_string * repetitions
            test_data = test_data[:buf_size]
            host_buf = ctypes.create_string_buffer(test_data, buf_size)
            cuda.cuMemcpyHtoDAsync(write_buffers[i], host_buf, buf_size, 0)
            cuda.cuStreamSynchronize(0)

        # Set up write operations
        for i in range(num_operations):
            io_params[i].mode = cufile.BatchMode.BATCH  # Batch mode
            io_params[i].fh = handle
            io_params[i].opcode = cufile.Opcode.WRITE  # Write opcode
            io_params[i].cookie = i
            io_params[i].u.batch.dev_ptr_base = int(write_buffers[i])
            io_params[i].u.batch.file_offset = i * buf_size
            io_params[i].u.batch.dev_ptr_offset = 0
            io_params[i].u.batch.size_ = buf_size

        # Set up read operations
        for i in range(num_operations):
            idx = i + num_operations
            io_params[idx].mode = cufile.BatchMode.BATCH  # Batch mode
            io_params[idx].fh = handle
            io_params[idx].opcode = cufile.Opcode.READ  # Read opcode
            io_params[idx].cookie = i + 100
            io_params[idx].u.batch.dev_ptr_base = int(read_buffers[i])
            io_params[idx].u.batch.file_offset = i * buf_size
            io_params[idx].u.batch.dev_ptr_offset = 0
            io_params[idx].u.batch.size_ = buf_size

        # Submit batch operations
        cufile.batch_io_submit(batch_handle, num_operations * 2, io_params.ptr, 0)

        # Get batch status
        min_nr = num_operations * 2  # Wait for all operations to complete
        nr_completed = ctypes.c_uint(num_operations * 2)  # Initialize to max operations posted
        timeout = ctypes.c_int(10000)  # 10 second timeout for large operations

        cufile.batch_io_get_status(
            batch_handle, min_nr, ctypes.addressof(nr_completed), io_events.ptr, ctypes.addressof(timeout)
        )

        # Verify all operations completed successfully
        assert nr_completed.value == num_operations * 2, (
            f"Expected {num_operations * 2} operations, got {nr_completed.value}"
        )

        # Collect all returned cookies
        returned_cookies = set()
        for i in range(num_operations * 2):
            assert io_events[i].status == cufile.Status.COMPLETE, (
                f"Operation {i} failed with status {io_events[i].status}"
            )
            returned_cookies.add(io_events[i].cookie)

        # Verify all expected cookies are present
        expected_cookies = set(range(num_operations)) | set(
            range(100, 100 + num_operations)
        )  # write cookies 0,1 + read cookies 100,101
        assert returned_cookies == expected_cookies, (
            f"Cookie mismatch. Expected {expected_cookies}, got {returned_cookies}"
        )

        # Verify the read data matches the written data
        for i in range(num_operations):
            # Copy read data back to host
            cuda.cuMemcpyDtoHAsync(host_buf, read_buffers[i], buf_size, 0)
            cuda.cuStreamSynchronize(0)
            read_data = host_buf.value

            # Prepare expected data
            test_string = test_strings[i]
            test_string_len = len(test_string)
            repetitions = buf_size // test_string_len
            expected_data = (test_string * repetitions)[:buf_size]

            if read_data != expected_data:
                n = 100  # Show first n bytes
                raise RuntimeError(
                    f"Read data doesn't match written data for operation {i}: "
                    f"{len(read_data)=}, {len(expected_data)=}, "
                    f"first {n} bytes: read {read_data[:n]!r}, "
                    f"expected {expected_data[:n]!r}"
                )

        # Clean up batch IO
        cufile.batch_io_destroy(batch_handle)

        # Deregister file handle
        cufile.handle_deregister(handle)

        # Deregister buffers
        for buf in all_buffers:
            buf_int = int(buf)
            cufile.buf_deregister(buf_int)

    finally:
        # Close file
        os.close(fd)
        # Free CUDA memory
        for buf in all_buffers:
            cuda.cuMemFree(buf)
        # Clean up test file
        try:
            os.unlink(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        # Close cuFile driver
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
def test_set_get_parameter_size_t():
    """Test setting and getting size_t parameters with cuFile validation."""

    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Test setting and getting various size_t parameters

        # Test poll threshold size (in KB)
        poll_threshold_kb = 64  # 64KB threshold
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.POLLTHRESHOLD_SIZE_KB, poll_threshold_kb)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.POLLTHRESHOLD_SIZE_KB)
        assert retrieved_value == poll_threshold_kb, (
            f"Poll threshold mismatch: set {poll_threshold_kb}, got {retrieved_value}"
        )

        # Test max direct IO size (in KB)
        max_direct_io_kb = 1024  # 1MB max direct IO size
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_MAX_DIRECT_IO_SIZE_KB, max_direct_io_kb)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_MAX_DIRECT_IO_SIZE_KB)
        assert retrieved_value == max_direct_io_kb, (
            f"Max direct IO size mismatch: set {max_direct_io_kb}, got {retrieved_value}"
        )

        # Test max device cache size (in KB)
        max_cache_kb = 512  # 512KB max cache size
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, max_cache_kb)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)
        assert retrieved_value == max_cache_kb, f"Max cache size mismatch: set {max_cache_kb}, got {retrieved_value}"

        # Test per buffer cache size (in KB)
        per_buffer_cache_kb = 128  # 128KB per buffer cache
        cufile.set_parameter_size_t(
            cufile.SizeTConfigParameter.PROPERTIES_PER_BUFFER_CACHE_SIZE_KB, per_buffer_cache_kb
        )
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_PER_BUFFER_CACHE_SIZE_KB)
        assert retrieved_value == per_buffer_cache_kb, (
            f"Per buffer cache size mismatch: set {per_buffer_cache_kb}, got {retrieved_value}"
        )

        # Test max device pinned memory size (in KB)
        max_pinned_kb = 2048  # 2MB max pinned memory
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, max_pinned_kb)
        retrieved_value = cufile.get_parameter_size_t(
            cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
        )
        assert retrieved_value == max_pinned_kb, (
            f"Max pinned memory size mismatch: set {max_pinned_kb}, got {retrieved_value}"
        )

        # Test IO batch size
        batch_size = 16  # 16 operations per batch
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_IO_BATCHSIZE, batch_size)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_IO_BATCHSIZE)
        assert retrieved_value == batch_size, f"IO batch size mismatch: set {batch_size}, got {retrieved_value}"

        # Test batch IO timeout (in milliseconds)
        timeout_ms = 5000  # 5 second timeout
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_BATCH_IO_TIMEOUT_MS, timeout_ms)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.PROPERTIES_BATCH_IO_TIMEOUT_MS)
        assert retrieved_value == timeout_ms, f"Batch IO timeout mismatch: set {timeout_ms}, got {retrieved_value}"

        # Test execution parameters
        max_io_queue_depth = 32  # Max 32 operations in queue
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MAX_IO_QUEUE_DEPTH, max_io_queue_depth)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MAX_IO_QUEUE_DEPTH)
        assert retrieved_value == max_io_queue_depth, (
            f"Max IO queue depth mismatch: set {max_io_queue_depth}, got {retrieved_value}"
        )

        max_io_threads = 8  # Max 8 IO threads
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MAX_IO_THREADS, max_io_threads)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MAX_IO_THREADS)
        assert retrieved_value == max_io_threads, (
            f"Max IO threads mismatch: set {max_io_threads}, got {retrieved_value}"
        )

        min_io_threshold_kb = 4  # 4KB minimum IO threshold
        cufile.set_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MIN_IO_THRESHOLD_SIZE_KB, min_io_threshold_kb)
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MIN_IO_THRESHOLD_SIZE_KB)
        assert retrieved_value == min_io_threshold_kb, (
            f"Min IO threshold mismatch: set {min_io_threshold_kb}, got {retrieved_value}"
        )

        max_request_parallelism = 4  # Max 4 parallel requests
        cufile.set_parameter_size_t(
            cufile.SizeTConfigParameter.EXECUTION_MAX_REQUEST_PARALLELISM, max_request_parallelism
        )
        retrieved_value = cufile.get_parameter_size_t(cufile.SizeTConfigParameter.EXECUTION_MAX_REQUEST_PARALLELISM)
        assert retrieved_value == max_request_parallelism, (
            f"Max request parallelism mismatch: set {max_request_parallelism}, got {retrieved_value}"
        )

    finally:
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
def test_set_get_parameter_bool():
    """Test setting and getting boolean parameters with cuFile validation."""

    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Test setting and getting various boolean parameters

        # Test poll mode
        cufile.set_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_USE_POLL_MODE, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_USE_POLL_MODE)
        assert retrieved_value is True, f"Poll mode mismatch: set True, got {retrieved_value}"

        # Test compatibility mode
        cufile.set_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_ALLOW_COMPAT_MODE, False)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_ALLOW_COMPAT_MODE)
        assert retrieved_value is False, f"Compatibility mode mismatch: set False, got {retrieved_value}"

        # Test force compatibility mode
        cufile.set_parameter_bool(cufile.BoolConfigParameter.FORCE_COMPAT_MODE, False)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.FORCE_COMPAT_MODE)
        assert retrieved_value is False, f"Force compatibility mode mismatch: set False, got {retrieved_value}"

        # Test aggressive API check
        cufile.set_parameter_bool(cufile.BoolConfigParameter.FS_MISC_API_CHECK_AGGRESSIVE, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.FS_MISC_API_CHECK_AGGRESSIVE)
        assert retrieved_value is True, f"Aggressive API check mismatch: set True, got {retrieved_value}"

        # Test parallel IO
        cufile.set_parameter_bool(cufile.BoolConfigParameter.EXECUTION_PARALLEL_IO, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.EXECUTION_PARALLEL_IO)
        assert retrieved_value is True, f"Parallel IO mismatch: set True, got {retrieved_value}"

        # Test NVTX profiling
        cufile.set_parameter_bool(cufile.BoolConfigParameter.PROFILE_NVTX, False)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.PROFILE_NVTX)
        assert retrieved_value is False, f"NVTX profiling mismatch: set False, got {retrieved_value}"

        # Test system memory allowance
        cufile.set_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_ALLOW_SYSTEM_MEMORY, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.PROPERTIES_ALLOW_SYSTEM_MEMORY)
        assert retrieved_value is True, f"System memory allowance mismatch: set True, got {retrieved_value}"

        # Test PCI P2P DMA
        cufile.set_parameter_bool(cufile.BoolConfigParameter.USE_PCIP2PDMA, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.USE_PCIP2PDMA)
        assert retrieved_value is True, f"PCI P2P DMA mismatch: set True, got {retrieved_value}"

        # Test IO uring preference
        cufile.set_parameter_bool(cufile.BoolConfigParameter.PREFER_IO_URING, False)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.PREFER_IO_URING)
        assert retrieved_value is False, f"IO uring preference mismatch: set False, got {retrieved_value}"

        # Test force O_DIRECT mode
        cufile.set_parameter_bool(cufile.BoolConfigParameter.FORCE_ODIRECT_MODE, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.FORCE_ODIRECT_MODE)
        assert retrieved_value is True, f"Force O_DIRECT mode mismatch: set True, got {retrieved_value}"

        # Test topology detection skip
        cufile.set_parameter_bool(cufile.BoolConfigParameter.SKIP_TOPOLOGY_DETECTION, False)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.SKIP_TOPOLOGY_DETECTION)
        assert retrieved_value is False, f"Topology detection skip mismatch: set False, got {retrieved_value}"

        # Test stream memops bypass
        cufile.set_parameter_bool(cufile.BoolConfigParameter.STREAM_MEMOPS_BYPASS, True)
        retrieved_value = cufile.get_parameter_bool(cufile.BoolConfigParameter.STREAM_MEMOPS_BYPASS)
        assert retrieved_value is True, f"Stream memops bypass mismatch: set True, got {retrieved_value}"

    finally:
        cuda.cuDevicePrimaryCtxRelease(device)


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
def test_set_get_parameter_string():
    """Test setting and getting string parameters with cuFile validation."""

    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    try:
        # Test setting and getting various string parameters
        # Note: String parameter tests may have issues with the current implementation

        # Test logging level
        logging_level = "INFO"
        try:
            # Convert Python string to null-terminated C string
            logging_level_bytes = logging_level.encode("utf-8") + b"\x00"
            logging_level_buffer = ctypes.create_string_buffer(logging_level_bytes)
            cufile.set_parameter_string(
                cufile.StringConfigParameter.LOGGING_LEVEL, int(ctypes.addressof(logging_level_buffer))
            )
            retrieved_value_raw = cufile.get_parameter_string(cufile.StringConfigParameter.LOGGING_LEVEL, 256)
            # Use safe_decode_string to handle null terminators and padding
            retrieved_value = safe_decode_string(retrieved_value_raw.encode("utf-8"))
            logging.info(f"Logging level test: set {logging_level}, got {retrieved_value}")
            # The retrieved value should be a string, so we can compare directly
            assert retrieved_value == logging_level, (
                f"Logging level mismatch: set {logging_level}, got {retrieved_value}"
            )
        except Exception as e:
            logging.error(f"Logging level test failed: {e}")
            # Re-raise the exception to make the test fail
            raise

        # Test environment log file path
        logfile_path = tempfile.gettempdir() + "/cufile.log"
        try:
            # Convert Python string to null-terminated C string
            logfile_path_bytes = logfile_path.encode("utf-8") + b"\x00"
            logfile_buffer = ctypes.create_string_buffer(logfile_path_bytes)
            cufile.set_parameter_string(
                cufile.StringConfigParameter.ENV_LOGFILE_PATH, int(ctypes.addressof(logfile_buffer))
            )
            retrieved_value_raw = cufile.get_parameter_string(cufile.StringConfigParameter.ENV_LOGFILE_PATH, 256)
            # Use safe_decode_string to handle null terminators and padding
            retrieved_value = safe_decode_string(retrieved_value_raw.encode("utf-8"))
            logging.info(f"Log file path test: set {logfile_path}, got {retrieved_value}")
            # The retrieved value should be a string, so we can compare directly
            assert retrieved_value == logfile_path, f"Log file path mismatch: set {logfile_path}, got {retrieved_value}"
        except Exception as e:
            logging.error(f"Log file path test failed: {e}")
            # Re-raise the exception to make the test fail
            raise

        # Test log directory
        log_dir = tempfile.gettempdir() + "/cufile_logs"
        try:
            # Convert Python string to null-terminated C string
            log_dir_bytes = log_dir.encode("utf-8") + b"\x00"
            log_dir_buffer = ctypes.create_string_buffer(log_dir_bytes)
            cufile.set_parameter_string(cufile.StringConfigParameter.LOG_DIR, int(ctypes.addressof(log_dir_buffer)))
            retrieved_value_raw = cufile.get_parameter_string(cufile.StringConfigParameter.LOG_DIR, 256)
            # Use safe_decode_string to handle null terminators and padding
            retrieved_value = safe_decode_string(retrieved_value_raw.encode("utf-8"))
            logging.info(f"Log directory test: set {log_dir}, got {retrieved_value}")
            # The retrieved value should be a string, so we can compare directly
            assert retrieved_value == log_dir, f"Log directory mismatch: set {log_dir}, got {retrieved_value}"
        except Exception as e:
            logging.error(f"Log directory test failed: {e}")
            # Re-raise the exception to make the test fail
            raise

    finally:
        cuda.cuDevicePrimaryCtxRelease(device)
