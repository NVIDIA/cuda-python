# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import errno
import logging
import os
import pathlib
import platform
import subprocess
import tempfile
from contextlib import suppress
from functools import cache

import cuda.bindings.driver as cuda
import pytest

cufile = pytest.importorskip("cuda.bindings.cufile")

# Configure logging to show INFO level and above
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,  # Override any existing logging configuration
)

cufile = pytest.importorskip("cuda.bindings.cufile", reason="skipping tests on Windows")


@pytest.fixture
def cufile_env_json(monkeypatch):
    """Set CUFILE_ENV_PATH_JSON environment variable for async tests."""
    # Get absolute path to cufile.json in the same directory as this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, "cufile.json")
    assert os.path.isfile(config_path)
    monkeypatch.setenv("CUFILE_ENV_PATH_JSON", config_path)
    logging.info(f"Using cuFile config: {config_path}")


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
    """Check if the current filesystem is supported (ext4 or xfs).

    This uses `findmnt` so the kernel's mount table logic owns the decoding of the filesystem type.
    """
    fs_type = subprocess.check_output(["findmnt", "-no", "FSTYPE", "-T", os.getcwd()], text=True).strip()  # noqa: S603, S607
    logging.info(f"Current filesystem type (findmnt): {fs_type}")
    return fs_type in ("ext4", "xfs")


# Global skip condition for all tests if cuFile library is not available
pytestmark = [
    pytest.mark.skipif(not cufileLibraryAvailable(), reason="cuFile library not available on this system"),
    pytest.mark.skipif(
        platform.system() == "Linux" and "microsoft" in pathlib.Path("/proc/version").read_text().lower(),
        reason="skipping cuFile tests on WSL",
    ),
    pytest.mark.skipif(pathlib.Path("/etc/nv_tegra_release").exists(), reason="skipping cuFile tests on Tegra Linux"),
]

xfail_handle_register = pytest.mark.xfail(
    condition=isSupportedFilesystem() and os.environ.get("CI") is not None,
    raises=cufile.cuFileError,
    reason="handle_register call fails in CI for unknown reasons",
)


def test_cufile_success_defined():
    """Check if CUFILE_SUCCESS is defined in OpError enum."""
    assert hasattr(cufile.OpError, "SUCCESS")


@pytest.fixture
def ctx():
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    yield

    cuda.cuDevicePrimaryCtxRelease(device)


@pytest.fixture
def driver(ctx):
    cufile.driver_open()
    yield
    cufile.driver_close()


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_handle_register():
    """Test file handle registration with cuFile."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_simple():
    """Simple test for buffer registration with cuFile."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_host_memory():
    """Test buffer registration with host memory."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_multiple_buffers():
    """Test registering multiple buffers."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_invalid_flags():
    """Test buffer registration with invalid flags."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_large_buffer():
    """Test buffer registration with a large buffer."""
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


@pytest.mark.usefixtures("driver")
def test_buf_register_already_registered():
    """Test that registering an already registered buffer fails."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_cufile_read_write():
    """Test cuFile read and write operations."""
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

        # Verify bytes written equals bytes read
        assert bytes_written == write_size, f"Expected to write {write_size} bytes, but wrote {bytes_written}"
        assert bytes_read == write_size, f"Expected to read {write_size} bytes, but read {bytes_read}"
        assert bytes_written == bytes_read, f"Bytes written ({bytes_written}) doesn't match bytes read ({bytes_read})"

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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_cufile_read_write_host_memory():
    """Test cuFile read and write operations using host memory."""
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

        # Verify bytes written equals bytes read
        assert bytes_written == write_size, f"Expected to write {write_size} bytes, but wrote {bytes_written}"
        assert bytes_read == write_size, f"Expected to read {write_size} bytes, but read {bytes_read}"
        assert bytes_written == bytes_read, f"Bytes written ({bytes_written}) doesn't match bytes read ({bytes_read})"

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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_cufile_read_write_large():
    """Test cuFile read and write operations with large data."""
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

        # Verify bytes written equals bytes read
        assert bytes_written == write_size, f"Expected to write {write_size} bytes, but wrote {bytes_written}"
        assert bytes_read == write_size, f"Expected to read {write_size} bytes, but read {bytes_read}"
        assert bytes_written == bytes_read, f"Bytes written ({bytes_written}) doesn't match bytes read ({bytes_read})"

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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("ctx", "cufile_env_json", "driver")
@xfail_handle_register
def test_cufile_write_async():
    """Test cuFile asynchronous write operations."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("ctx", "cufile_env_json", "driver")
@xfail_handle_register
def test_cufile_read_async():
    """Test cuFile asynchronous read operations."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@xfail_handle_register
@pytest.mark.usefixtures("ctx", "cufile_env_json", "driver")
def test_cufile_async_read_write():
    """Test cuFile asynchronous read and write operations in sequence."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_batch_io_basic():
    """Test basic batch IO operations with multiple read/write operations."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_batch_io_cancel():
    """Test batch IO cancellation."""
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


@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("driver")
@xfail_handle_register
def test_batch_io_large_operations():
    """Test batch IO with large buffer operations."""
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
        batch_handle = cufile.batch_io_set_up(num_operations)  # Only for writes

        # Create IOParams array for batch operations
        io_params = cufile.IOParams(num_operations)
        io_events = cufile.IOEvents(num_operations)

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

        # Submit writes
        cufile.batch_io_submit(batch_handle, num_operations, io_params.ptr, 0)

        # Wait for writes to complete
        nr_completed_writes = ctypes.c_uint(num_operations)
        timeout = ctypes.c_int(10000)
        cufile.batch_io_get_status(
            batch_handle,
            num_operations,
            ctypes.addressof(nr_completed_writes),
            io_events.ptr,
            ctypes.addressof(timeout),
        )

        # Clean up write batch
        cufile.batch_io_destroy(batch_handle)

        # Now submit reads separately
        read_batch_handle = cufile.batch_io_set_up(num_operations)
        read_io_params = cufile.IOParams(num_operations)
        read_io_events = cufile.IOEvents(num_operations)

        # Set up read operations
        for i in range(num_operations):
            read_io_params[i].mode = cufile.BatchMode.BATCH
            read_io_params[i].fh = handle
            read_io_params[i].opcode = cufile.Opcode.READ
            read_io_params[i].cookie = i + 100
            read_io_params[i].u.batch.dev_ptr_base = int(read_buffers[i])
            read_io_params[i].u.batch.file_offset = i * buf_size
            read_io_params[i].u.batch.dev_ptr_offset = 0
            read_io_params[i].u.batch.size_ = buf_size

        # Submit reads
        cufile.batch_io_submit(read_batch_handle, num_operations, read_io_params.ptr, 0)

        # Wait for reads
        nr_completed = ctypes.c_uint(num_operations)
        cufile.batch_io_get_status(
            read_batch_handle,
            num_operations,
            ctypes.addressof(nr_completed),
            read_io_events.ptr,
            ctypes.addressof(timeout),
        )

        # Verify all operations completed successfully
        assert nr_completed.value == num_operations, f"Expected {num_operations} operations, got {nr_completed.value}"

        # Collect all returned cookies
        returned_cookies = set()
        for i in range(num_operations):
            assert read_io_events[i].status == cufile.Status.COMPLETE, (
                f"Operation {i} failed with status {read_io_events[i].status}"
            )
            returned_cookies.add(read_io_events[i].cookie)

        # Verify all expected cookies are present
        expected_cookies = set(range(100, 100 + num_operations))
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
        cufile.batch_io_destroy(read_batch_handle)

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


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
@pytest.mark.usefixtures("ctx")
def test_set_get_parameter_size_t():
    """Test setting and getting size_t parameters with cuFile validation."""
    param_val_pairs = (
        (cufile.SizeTConfigParameter.POLLTHRESHOLD_SIZE_KB, 64),  # 64KB threshold
        (cufile.SizeTConfigParameter.PROPERTIES_MAX_DIRECT_IO_SIZE_KB, 1024),  # 1MB max direct IO size
        (cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, 512),  # 512KB max cache size
        (cufile.SizeTConfigParameter.PROPERTIES_PER_BUFFER_CACHE_SIZE_KB, 128),  # 128KB per buffer cache
        (cufile.SizeTConfigParameter.PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, 2048),  # 2MB max pinned memory
        (cufile.SizeTConfigParameter.PROPERTIES_IO_BATCHSIZE, 16),  # 16 operations per batch
        (cufile.SizeTConfigParameter.PROPERTIES_BATCH_IO_TIMEOUT_MS, 5000),  # 5 second timeout
        (cufile.SizeTConfigParameter.EXECUTION_MAX_IO_QUEUE_DEPTH, 32),  # Max 32 operations in queue
        (cufile.SizeTConfigParameter.EXECUTION_MAX_IO_THREADS, 8),  # Max 8 IO threads
        (cufile.SizeTConfigParameter.EXECUTION_MIN_IO_THRESHOLD_SIZE_KB, 4),  # 4KB minimum IO threshold
        (cufile.SizeTConfigParameter.EXECUTION_MAX_REQUEST_PARALLELISM, 4),  # Max 4 parallel requests
    )

    def test_param(param, val):
        orig_val = cufile.get_parameter_size_t(param)
        cufile.set_parameter_size_t(param, val)
        retrieved_val = cufile.get_parameter_size_t(param)
        assert retrieved_val == val
        cufile.set_parameter_size_t(param, orig_val)

    # Test setting and getting various size_t parameters
    for param, val in param_val_pairs:
        test_param(param, val)


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
@pytest.mark.usefixtures("ctx")
def test_set_get_parameter_bool():
    """Test setting and getting boolean parameters with cuFile validation."""
    param_val_pairs = (
        (cufile.BoolConfigParameter.PROPERTIES_USE_POLL_MODE, True),
        (cufile.BoolConfigParameter.PROPERTIES_ALLOW_COMPAT_MODE, False),
        (cufile.BoolConfigParameter.FORCE_COMPAT_MODE, False),
        (cufile.BoolConfigParameter.FS_MISC_API_CHECK_AGGRESSIVE, True),
        (cufile.BoolConfigParameter.EXECUTION_PARALLEL_IO, True),
        (cufile.BoolConfigParameter.PROFILE_NVTX, False),
        (cufile.BoolConfigParameter.PROPERTIES_ALLOW_SYSTEM_MEMORY, True),
        (cufile.BoolConfigParameter.USE_PCIP2PDMA, True),
        (cufile.BoolConfigParameter.PREFER_IO_URING, False),
        (cufile.BoolConfigParameter.FORCE_ODIRECT_MODE, True),
        (cufile.BoolConfigParameter.SKIP_TOPOLOGY_DETECTION, False),
        (cufile.BoolConfigParameter.STREAM_MEMOPS_BYPASS, True),
    )

    def test_param(param, val):
        orig_val = cufile.get_parameter_bool(param)
        cufile.set_parameter_bool(param, val)
        retrieved_val = cufile.get_parameter_bool(param)
        assert retrieved_val is val
        cufile.set_parameter_bool(param, orig_val)

    try:
        # Test setting and getting various boolean parameters
        for param, val in param_val_pairs:
            test_param(param, val)
    except cufile.cuFileError:
        if cufile.get_version() < 1160:
            raise
        assert param is cufile.BoolConfigParameter.PROFILE_NVTX  # Deprecated in CTK 13.1.0


@pytest.mark.skipif(
    cufileVersionLessThan(1140), reason="cuFile parameter APIs require cuFile library version 1.14.0 or later"
)
@pytest.mark.usefixtures("ctx")
def test_set_get_parameter_string(tmp_path):
    """Test setting and getting string parameters with cuFile validation."""
    temp_dir = tempfile.gettempdir()
    # must be set to avoid getter error when testing ENV_LOGFILE_PATH...
    os.environ["CUFILE_LOGFILE_PATH"] = ""

    param_val_pairs = (
        (cufile.StringConfigParameter.LOGGING_LEVEL, "INFO", "DEBUG"),  # Test logging level
        (
            cufile.StringConfigParameter.ENV_LOGFILE_PATH,
            os.path.join(temp_dir, "cufile.log"),
            str(tmp_path / "cufile.log"),
        ),  # Test environment log file path
        (
            cufile.StringConfigParameter.LOG_DIR,
            os.path.join(temp_dir, "cufile_logs"),
            str(tmp_path),
        ),  # Test log directory
    )

    def test_param(param, val, default_val):
        orig_val = cufile.get_parameter_string(param, 256)

        val_b = val.encode("utf-8")
        val_buf = ctypes.create_string_buffer(val_b)
        default_val_b = default_val.encode("utf-8")
        defualt_val_buf = ctypes.create_string_buffer(default_val_b)
        orig_val_b = orig_val.encode("utf-8")
        orig_val_buf = ctypes.create_string_buffer(orig_val_b)

        # Round-trip test
        cufile.set_parameter_string(param, int(ctypes.addressof(val_buf)))
        retrieved_val = cufile.get_parameter_string(param, 256)
        assert retrieved_val == val

        # Restore
        try:
            # Currently this line will raise, see below.
            cufile.set_parameter_string(param, int(ctypes.addressof(orig_val_buf)))
        except:
            # This block will always be reached because cuFILE could start with garbage default (empty string)
            # that cannot be restored. In other words, cuFILE does honor the common sense that getter/setter
            # should be round-tripable.
            cufile.set_parameter_string(param, int(ctypes.addressof(defualt_val_buf)))

    try:
        # Test setting and getting various string parameters
        # Note: String parameter tests may have issues with the current implementation
        for param, val, default_val in param_val_pairs:
            test_param(param, val, default_val)
    finally:
        del os.environ["CUFILE_LOGFILE_PATH"]


@pytest.fixture
def stats(driver):
    old_level = cufile.get_stats_level()
    yield
    # Reset cuFile statistics to clear all counters
    cufile.stats_reset()
    cufile.set_stats_level(old_level)


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.usefixtures("stats")
def test_set_stats_level():
    """Test cuFile statistics level configuration."""
    # Test setting different statistics levels
    valid_levels = [0, 1, 2, 3]  # 0=disabled, 1=basic, 2=detailed, 3=verbose

    for level in valid_levels:
        cufile.set_stats_level(level)

        # Verify the level was set correctly
        current_level = cufile.get_stats_level()
        assert current_level == level, f"Expected stats level {level}, but got {current_level}"

        logging.info(f"Successfully set and verified stats level {level}")

    # Test invalid level (should raise an error)
    try:
        assert cufile.set_stats_level(-1)  # Invalid negative level
    except Exception as e:
        logging.info(f"Correctly caught error for invalid stats level: {e}")

    try:
        assert cufile.set_stats_level(4)  # Invalid level > 3
    except Exception as e:
        logging.info(f"Correctly caught error for invalid stats level: {e}")


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.usefixtures("driver")
def test_get_parameter_min_max_value():
    """Test getting minimum and maximum values for size_t parameters."""
    # Test with poll threshold parameter
    param = cufile.SizeTConfigParameter.POLLTHRESHOLD_SIZE_KB

    # Get min/max values
    min_value, max_value = cufile.get_parameter_min_max_value(param)

    # Verify that min <= max and both are reasonable values
    assert min_value >= 0, f"Invalid min value: {min_value}"
    assert max_value >= min_value, f"Max value {max_value} < min value {min_value}"
    assert max_value > 0, f"Invalid max value: {max_value}"

    logging.info(f"POLLTHRESHOLD_SIZE_KB: min={min_value}, max={max_value}")


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.usefixtures("stats")
def test_stats_start_stop():
    """Test cuFile statistics collection stop."""
    # Set statistics level first (required before starting stats)
    cufile.set_stats_level(1)  # Level 1 = basic statistics
    # Start collecting cuFile statistics first
    cufile.stats_start()

    # Stop collecting cuFile statistics
    cufile.stats_stop()


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("stats")
@xfail_handle_register
def test_get_stats_l1():
    """Test cuFile L1 statistics retrieval with file operations."""
    # Create test file directly with O_DIRECT
    file_path = "test_stats_l1.bin"
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

    try:
        cufile.set_stats_level(1)  # L1 = basic operation counts
        # Start collecting cuFile statistics
        cufile.stats_start()

        # Create and initialize the descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register the handle
        handle = cufile.handle_register(descr.ptr)

        # Allocate CUDA memory
        buffer_size = 4096  # 4KB, aligned to 4096 bytes
        err, buf_ptr = cuda.cuMemAlloc(buffer_size)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register the buffer with cuFile
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, 0)

        # Prepare test data and copy to GPU buffer
        test_data = b"cuFile L1 stats test data" * 100  # Fill buffer
        test_data = test_data[:buffer_size]
        host_buf = ctypes.create_string_buffer(test_data, buffer_size)
        cuda.cuMemcpyHtoD(buf_ptr, host_buf, len(test_data))

        # Perform cuFile operations to generate L1 statistics
        cufile.write(handle, buf_ptr_int, buffer_size, 0, 0)
        cufile.read(handle, buf_ptr_int, buffer_size, 0, 0)

        # Use the exposed StatsLevel1 class from cufile module
        stats = cufile.StatsLevel1()

        # Get L1 statistics (basic operation counts)
        cufile.get_stats_l1(stats.ptr)

        # Verify actual field values using OpCounter class for cleaner access
        read_ops = cufile.OpCounter.from_data(stats.read_ops)
        write_ops = cufile.OpCounter.from_data(stats.write_ops)
        read_bytes = int(stats.read_bytes)
        write_bytes = int(stats.write_bytes)

        assert read_ops.ok > 0, f"Expected read operations, got {read_ops.ok}"
        assert write_ops.ok > 0, f"Expected write operations, got {write_ops.ok}"
        assert read_bytes > 0, f"Expected read bytes, got {read_bytes}"
        assert write_bytes > 0, f"Expected write bytes, got {write_bytes}"

        logging.info(
            f"Stats: reads={read_ops.ok}, writes={write_ops.ok}, read_bytes={read_bytes}, write_bytes={write_bytes}"
        )

        # Stop statistics collection
        cufile.stats_stop()

        # Clean up cuFile resources
        cufile.buf_deregister(buf_ptr_int)
        cufile.handle_deregister(handle)
        cuda.cuMemFree(buf_ptr)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("stats")
@xfail_handle_register
def test_get_stats_l2():
    """Test cuFile L2 statistics retrieval with file operations."""
    # Create test file directly with O_DIRECT
    file_path = "test_stats_l2.bin"
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

    try:
        cufile.set_stats_level(2)  # L2 = detailed performance metrics

        # Start collecting cuFile statistics
        cufile.stats_start()

        # Create and initialize the descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register the handle
        handle = cufile.handle_register(descr.ptr)

        # Allocate CUDA memory
        buffer_size = 8192  # 8KB for more detailed stats
        err, buf_ptr = cuda.cuMemAlloc(buffer_size)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register the buffer with cuFile
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, 0)

        # Prepare test data and copy to GPU buffer
        test_data = b"cuFile L2 detailed stats test data" * 150  # Fill buffer
        test_data = test_data[:buffer_size]
        host_buf = ctypes.create_string_buffer(test_data, buffer_size)
        cuda.cuMemcpyHtoD(buf_ptr, host_buf, len(test_data))

        # Perform multiple cuFile operations to generate detailed L2 statistics
        cufile.write(handle, buf_ptr_int, buffer_size, 0, 0)
        cufile.read(handle, buf_ptr_int, buffer_size, 0, 0)
        cufile.write(handle, buf_ptr_int, buffer_size, buffer_size, 0)  # Different offset
        cufile.read(handle, buf_ptr_int, buffer_size, buffer_size, 0)

        # Use the exposed StatsLevel2 class from cufile module
        stats = cufile.StatsLevel2()

        # Get L2 statistics (detailed performance metrics)
        cufile.get_stats_l2(stats.ptr)

        # Verify L2 histogram fields contain data
        # Access numpy array fields: histograms are numpy arrays
        read_hist_total = int(stats.read_size_kb_hist.sum())
        write_hist_total = int(stats.write_size_kb_hist.sum())
        assert read_hist_total > 0 or write_hist_total > 0, "Expected L2 histogram data"

        # L2 also contains L1 basic stats - verify using OpCounter class
        basic_stats = cufile.StatsLevel1.from_data(stats.basic)
        read_ops = cufile.OpCounter.from_data(basic_stats.read_ops)
        write_ops = cufile.OpCounter.from_data(basic_stats.write_ops)

        logging.info(
            f"L2 Stats: read_hist_total={read_hist_total}, write_hist_total={write_hist_total}, "
            f"basic_reads={read_ops.ok}, basic_writes={write_ops.ok}"
        )

        # Stop statistics collection
        cufile.stats_stop()

        # Clean up cuFile resources
        cufile.buf_deregister(buf_ptr_int)
        cufile.handle_deregister(handle)
        cuda.cuMemFree(buf_ptr)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.skipif(not isSupportedFilesystem(), reason="cuFile handle_register requires ext4 or xfs filesystem")
@pytest.mark.usefixtures("stats")
@xfail_handle_register
def test_get_stats_l3():
    """Test cuFile L3 statistics retrieval with file operations."""
    # Create test file directly with O_DIRECT
    file_path = "test_stats_l3.bin"
    fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)

    try:
        cufile.set_stats_level(3)  # L3 = comprehensive diagnostic data

        # Start collecting cuFile statistics
        cufile.stats_start()

        # Create and initialize the descriptor
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = 0

        # Register the handle
        handle = cufile.handle_register(descr.ptr)

        # Allocate CUDA memory
        buffer_size = 16384  # 16KB for comprehensive stats testing
        err, buf_ptr = cuda.cuMemAlloc(buffer_size)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Register the buffer with cuFile
        buf_ptr_int = int(buf_ptr)
        cufile.buf_register(buf_ptr_int, buffer_size, 0)

        # Prepare test data and copy to GPU buffer
        test_data = b"cuFile L3 comprehensive stats test data" * 200  # Fill buffer
        test_data = test_data[:buffer_size]
        host_buf = ctypes.create_string_buffer(test_data, buffer_size)
        cuda.cuMemcpyHtoD(buf_ptr, host_buf, len(test_data))

        # Perform comprehensive cuFile operations to generate L3 statistics
        # Multiple writes and reads at different offsets to generate rich stats
        cufile.write(handle, buf_ptr_int, buffer_size, 0, 0)
        cufile.read(handle, buf_ptr_int, buffer_size, 0, 0)
        cufile.write(handle, buf_ptr_int, buffer_size, buffer_size, 0)  # Different offset
        cufile.read(handle, buf_ptr_int, buffer_size, buffer_size, 0)
        cufile.write(handle, buf_ptr_int, buffer_size // 2, buffer_size * 2, 0)  # Partial write
        cufile.read(handle, buf_ptr_int, buffer_size // 2, buffer_size * 2, 0)  # Partial read

        # Use the exposed StatsLevel3 class from cufile module
        stats = cufile.StatsLevel3()

        # Get L3 statistics (comprehensive diagnostic data)
        cufile.get_stats_l3(stats.ptr)

        # Verify L3-specific fields
        num_gpus = int(stats.num_gpus)
        assert num_gpus >= 0, f"Expected valid GPU count, got {num_gpus}"

        # Check if we have at least one GPU with stats using PerGpuStats class
        gpu_with_data = False
        for i in range(min(num_gpus, 16)):
            # Access per-GPU stats using PerGpuStats class
            # stats.per_gpu_stats has shape (1, 16), we need to get [0] first to get the (16,) array
            # then slice [i:i+1] to get a 1-d array view (required by from_data)
            gpu_stats = stats.per_gpu_stats[i]  # Get the (16,) array
            if gpu_stats.n_total_reads > 0 or gpu_stats.read_bytes > 0:
                gpu_with_data = True
                break

        # L3 also contains L2 detailed stats (which includes L1 basic stats)
        detailed_stats = cufile.StatsLevel2.from_data(stats.detailed)
        read_hist_total = int(detailed_stats.read_size_kb_hist.sum())

        logging.info(
            f"L3 Stats: num_gpus={num_gpus}, gpu_with_data={gpu_with_data}, detailed_read_hist={read_hist_total}"
        )

        # Stop statistics collection
        cufile.stats_stop()

        # Clean up cuFile resources
        cufile.buf_deregister(buf_ptr_int)
        cufile.handle_deregister(handle)
        cuda.cuMemFree(buf_ptr)

    finally:
        os.close(fd)
        with suppress(OSError):
            os.unlink(file_path)


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.usefixtures("driver")
def test_get_bar_size_in_kb():
    """Test cuFile BAR (Base Address Register) size retrieval."""
    # Get BAR size in kilobytes
    bar_size_kb = cufile.get_bar_size_in_kb(0)

    # Verify BAR size is a reasonable value
    assert isinstance(bar_size_kb, int), "BAR size should be an integer"
    assert bar_size_kb > 0, "BAR size should be positive"

    logging.info(f"GPU BAR size: {bar_size_kb} KB ({bar_size_kb / 1024 / 1024:.2f} GB)")


@pytest.fixture(scope="module")
def slab_sizes():
    """Define slab sizes for POSIX I/O pool (common I/O buffer sizes) - BEFORE driver open"""
    return [
        4096,  # 4KB - small files
        65536,  # 64KB - medium files
        1048576,  # 1MB - large files
        16777216,  # 16MB - very large files
    ]


@pytest.fixture(scope="module")
def slab_counts():
    """Define counts for each slab size (number of buffers)"""
    return [
        10,  # 10 buffers of 4KB
        5,  # 5 buffers of 64KB
        3,  # 3 buffers of 1MB
        2,  # 2 buffers of 16MB
    ]


@pytest.fixture
def driver_config(slab_sizes, slab_counts):
    # Convert to ctypes arrays
    size_array_type = ctypes.c_size_t * len(slab_sizes)
    count_array_type = ctypes.c_size_t * len(slab_counts)
    size_array = size_array_type(*slab_sizes)
    count_array = count_array_type(*slab_counts)

    # Set POSIX pool slab array configuration BEFORE opening driver
    cufile.set_parameter_posix_pool_slab_array(
        ctypes.addressof(size_array), ctypes.addressof(count_array), len(slab_sizes)
    )


@pytest.mark.skipif(
    cufileVersionLessThan(1150), reason="cuFile parameter APIs require cuFile library version 13.0 or later"
)
@pytest.mark.usefixtures("ctx")
def test_set_parameter_posix_pool_slab_array(slab_sizes, slab_counts, driver_config):
    """Test cuFile POSIX pool slab array configuration."""
    # After setting parameters, retrieve them back to verify
    n_slab_sizes = len(slab_sizes)
    retrieved_sizes = (ctypes.c_size_t * n_slab_sizes)()
    retrieved_counts = (ctypes.c_size_t * len(slab_counts))()

    retrieved_sizes_addr = ctypes.addressof(retrieved_sizes)
    retrieved_counts_addr = ctypes.addressof(retrieved_counts)

    # Open cuFile driver AFTER setting parameters
    cufile.driver_open()
    try:
        cufile.get_parameter_posix_pool_slab_array(retrieved_sizes_addr, retrieved_counts_addr, n_slab_sizes)
    finally:
        cufile.driver_close()

    # Verify they match what we set
    assert list(retrieved_sizes) == slab_sizes
    assert list(retrieved_counts) == slab_counts
