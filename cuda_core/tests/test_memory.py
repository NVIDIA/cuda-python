# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import array
import ctypes
import multiprocessing
import platform
import traceback

import pytest

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, MemoryResource
from cuda.core.experimental._memory import DLDeviceType
from cuda.core.experimental._utils.cuda_utils import get_binding_version, handle_return

if platform.system() == "Linux":
    from socket import AF_UNIX, CMSG_LEN, SCM_RIGHTS, SOCK_DGRAM, SOL_SOCKET, socketpair


class DummyDeviceMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def __bool__(self):
        return True

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFree(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return 0


class DummyHostMemoryResource(MemoryResource):
    def __init__(self):
        pass

    def __bool__(self):
        return True

    def allocate(self, size, stream=None) -> Buffer:
        # Allocate a ctypes buffer of size `size`
        ptr = (ctypes.c_byte * size)()
        self._ptr = ptr
        return Buffer.from_handle(ptr=ctypes.addressof(ptr), size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        del self._ptr

    @property
    def is_device_accessible(self) -> bool:
        return False

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")


class DummyUnifiedMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def __bool__(self):
        return True

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocManaged(size, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFree(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        return 0


class DummyPinnedMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def __bool__(self):
        return True

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocHost(size))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFreeHost(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")


class NullMemoryResource(DummyHostMemoryResource):
    @property
    def is_host_accessible(self) -> bool:
        return False


def buffer_initialization(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=1024)
    assert buffer.handle != 0
    assert buffer.size == 1024
    assert buffer.memory_resource == dummy_mr
    assert buffer.is_device_accessible == dummy_mr.is_device_accessible
    assert buffer.is_host_accessible == dummy_mr.is_host_accessible
    buffer.close()


def test_buffer_initialization():
    device = Device()
    device.set_current()
    buffer_initialization(DummyDeviceMemoryResource(device))
    buffer_initialization(DummyHostMemoryResource())
    buffer_initialization(DummyUnifiedMemoryResource(device))
    buffer_initialization(DummyPinnedMemoryResource(device))


def buffer_copy_to(dummy_mr: MemoryResource, device: Device, check=False):
    src_buffer = dummy_mr.allocate(size=1024)
    dst_buffer = dummy_mr.allocate(size=1024)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(1024):
            src_ptr[i] = ctypes.c_byte(i)

    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()

    if check:
        dst_ptr = ctypes.cast(dst_buffer.handle, ctypes.POINTER(ctypes.c_byte))

        for i in range(10):
            assert dst_ptr[i] == src_ptr[i]

    dst_buffer.close()
    src_buffer.close()


def test_buffer_copy_to():
    device = Device()
    device.set_current()
    buffer_copy_to(DummyDeviceMemoryResource(device), device)
    buffer_copy_to(DummyUnifiedMemoryResource(device), device)
    buffer_copy_to(DummyPinnedMemoryResource(device), device, check=True)


def buffer_copy_from(dummy_mr: MemoryResource, device, check=False):
    src_buffer = dummy_mr.allocate(size=1024)
    dst_buffer = dummy_mr.allocate(size=1024)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(1024):
            src_ptr[i] = ctypes.c_byte(i)

    dst_buffer.copy_from(src_buffer, stream=stream)
    device.sync()

    if check:
        dst_ptr = ctypes.cast(dst_buffer.handle, ctypes.POINTER(ctypes.c_byte))

        for i in range(10):
            assert dst_ptr[i] == src_ptr[i]

    dst_buffer.close()
    src_buffer.close()


def test_buffer_copy_from():
    device = Device()
    device.set_current()
    buffer_copy_from(DummyDeviceMemoryResource(device), device)
    buffer_copy_from(DummyUnifiedMemoryResource(device), device)
    buffer_copy_from(DummyPinnedMemoryResource(device), device, check=True)


def buffer_close(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=1024)
    buffer.close()
    assert buffer.handle is None
    assert buffer.memory_resource is None


def test_buffer_close():
    device = Device()
    device.set_current()
    buffer_close(DummyDeviceMemoryResource(device))
    buffer_close(DummyHostMemoryResource())
    buffer_close(DummyUnifiedMemoryResource(device))
    buffer_close(DummyPinnedMemoryResource(device))


def test_buffer_dunder_dlpack():
    device = Device()
    device.set_current()
    dummy_mr = DummyDeviceMemoryResource(device)
    buffer = dummy_mr.allocate(size=1024)
    capsule = buffer.__dlpack__()
    assert "dltensor" in repr(capsule)
    capsule = buffer.__dlpack__(max_version=(1, 0))
    assert "dltensor" in repr(capsule)
    with pytest.raises(BufferError, match=r"^Sorry, not supported: dl_device other than None$"):
        buffer.__dlpack__(dl_device=())
    with pytest.raises(BufferError, match=r"^Sorry, not supported: copy=True$"):
        buffer.__dlpack__(copy=True)
    with pytest.raises(BufferError, match=r"^Expected max_version tuple\[int, int\], got \(\)$"):
        buffer.__dlpack__(max_version=())
    with pytest.raises(BufferError, match=r"^Expected max_version tuple\[int, int\], got \(9, 8, 7\)$"):
        buffer.__dlpack__(max_version=(9, 8, 7))


@pytest.mark.parametrize(
    ("DummyMR", "expected"),
    [
        (DummyDeviceMemoryResource, (DLDeviceType.kDLCUDA, 0)),
        (DummyHostMemoryResource, (DLDeviceType.kDLCPU, 0)),
        (DummyUnifiedMemoryResource, (DLDeviceType.kDLCUDAHost, 0)),
        (DummyPinnedMemoryResource, (DLDeviceType.kDLCUDAHost, 0)),
    ],
)
def test_buffer_dunder_dlpack_device_success(DummyMR, expected):
    device = Device()
    device.set_current()
    dummy_mr = DummyMR() if DummyMR is DummyHostMemoryResource else DummyMR(device)
    buffer = dummy_mr.allocate(size=1024)
    assert buffer.__dlpack_device__() == expected


def test_buffer_dunder_dlpack_device_failure():
    dummy_mr = NullMemoryResource()
    buffer = dummy_mr.allocate(size=1024)
    with pytest.raises(BufferError, match=r"^buffer is neither device-accessible nor host-accessible$"):
        buffer.__dlpack_device__()


def test_device_memory_resource_initialization():
    """Test that DeviceMemoryResource can be initialized successfully.

    This test verifies that the DeviceMemoryResource initializes properly,
    including the release threshold configuration for performance optimization.
    """
    device = Device()
    if not device.properties.memory_pools_supported:
        pytest.skip("memory pools not supported")
    device.set_current()

    # This should succeed and configure the memory pool release threshold.
    # The resource can be constructed from either a device or device ordinal.
    for device_arg in [device, device.device_id]:
        mr = DeviceMemoryResource(device_arg)

        # Verify basic properties
        assert mr.device_id == device.device_id
        assert mr.is_device_accessible
        assert not mr.is_host_accessible
        assert not mr.is_ipc_enabled

        # Test allocation/deallocation works
        buffer = mr.allocate(1024)
        assert buffer.size == 1024
        assert buffer.device_id == device.device_id
        buffer.close()


def test_mempool():
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    pool_size = 2097152  # 2MB size

    # Test basic pool creation
    mr = DeviceMemoryResource(device, dict(max_size=pool_size, ipc_enabled=False))
    assert mr.device_id == device.device_id
    assert mr.is_device_accessible
    assert not mr.is_host_accessible
    assert not mr.is_ipc_enabled

    # Test allocation and deallocation
    buffer1 = mr.allocate(1024)
    assert buffer1.handle != 0
    assert buffer1.size == 1024
    assert buffer1.memory_resource == mr
    buffer1.close()

    # Test multiple allocations
    buffer1 = mr.allocate(1024)
    buffer2 = mr.allocate(2048)
    assert buffer1.handle != buffer2.handle
    assert buffer1.size == 1024
    assert buffer2.size == 2048
    buffer1.close()
    buffer2.close()

    # Test stream-based allocation
    stream = device.create_stream()
    buffer = mr.allocate(1024, stream=stream)
    assert buffer.handle != 0
    buffer.close()

    # Test memory copying between buffers from same pool
    src_buffer = mr.allocate(64)
    dst_buffer = mr.allocate(64)
    stream = device.create_stream()
    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()
    dst_buffer.close()
    src_buffer.close()

    # Test error cases
    # Test IPC operations are disabled
    buffer = mr.allocate(64)
    ipc_error_msg = "Memory resource is not IPC-enabled"

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        mr.get_allocation_handle()

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        mr.export_buffer(buffer)

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        mr.import_buffer(None)

    buffer.close()


@pytest.mark.parametrize(
    "property_name,expected_type",
    [
        ("reuse_follow_event_dependencies", bool),
        ("reuse_allow_opportunistic", bool),
        ("reuse_allow_internal_dependencies", bool),
        ("release_threshold", int),
        ("reserved_mem_current", int),
        ("reserved_mem_high", int),
        ("used_mem_current", int),
        ("used_mem_high", int),
    ],
)
def test_mempool_properties(property_name, expected_type):
    """Test all properties of the DeviceMemoryResource class."""
    # skip test if cuda version is less than 12
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")

    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    pool_size = 2097152  # 2MB size
    for ipc_enabled in [True, False]:
        mr = DeviceMemoryResource(device, dict(max_size=pool_size, ipc_enabled=ipc_enabled))
        assert mr.is_ipc_enabled == ipc_enabled

        try:
            # Get the property value
            value = getattr(mr, property_name)

            # Test type
            assert isinstance(value, expected_type), f"{property_name} should return {expected_type}, got {type(value)}"

            # Test value constraints
            if expected_type is int:
                assert value >= 0, f"{property_name} should be non-negative"

            # Test memory usage properties with actual allocations
            if property_name in ["reserved_mem_current", "used_mem_current"]:
                # Allocate some memory and check if values increase
                initial_value = value
                buffer = None
                try:
                    buffer = mr.allocate(1024)
                    new_value = getattr(mr, property_name)
                    assert new_value >= initial_value, f"{property_name} should increase or stay same after allocation"
                finally:
                    if buffer is not None:
                        buffer.close()

            # Test high watermark properties
            if property_name in ["reserved_mem_high", "used_mem_high"]:
                # High watermark should never be less than current
                current_prop = property_name.replace("_high", "_current")
                current_value = getattr(mr, current_prop)
                assert value >= current_value, f"{property_name} should be >= {current_prop}"

        finally:
            # Ensure we allocate and deallocate a small buffer to flush any pending operations
            flush_buffer = mr.allocate(64)
            flush_buffer.close()


def mempool_child_process(importer, queue):
    try:
        device = Device()
        device.set_current()
        stream = device.create_stream()

        # Get the allocation handle differently based on platform
        if platform.system() == "Windows":
            alloc_handle = queue.get()  # On Windows, we pass the handle through the queue
        else:
            # Unix socket handle transfer
            fds = array.array("i")
            _, ancdata, _, _ = importer.recvmsg(0, CMSG_LEN(fds.itemsize))
            assert len(ancdata) == 1
            cmsg_level, cmsg_type, cmsg_data = ancdata[0]
            assert cmsg_level == SOL_SOCKET and cmsg_type == SCM_RIGHTS
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            alloc_handle = int(fds[0])

        mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
        ipc_buffer = queue.get()  # Get exported buffer data
        buffer = mr.import_buffer(ipc_buffer)

        # Create a new buffer to verify data using unified memory
        unified_mr = DummyUnifiedMemoryResource(device)
        verify_buffer = unified_mr.allocate(64)

        # Copy data from imported buffer to verify contents
        verify_buffer.copy_from(buffer, stream=stream)
        device.sync()

        # Verify the data matches expected pattern
        verify_ptr = ctypes.cast(int(verify_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        for i in range(64):
            assert ctypes.c_byte(verify_ptr[i]).value == ctypes.c_byte(i).value, f"Data mismatch at index {i}"

        # Write new pattern to the buffer using unified memory
        src_buffer = unified_mr.allocate(64)
        src_ptr = ctypes.cast(int(src_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        for i in range(64):
            src_ptr[i] = ctypes.c_byte(255 - i)  # Write inverted pattern

        # Copy new pattern to the IPC buffer
        buffer.copy_from(src_buffer, stream=stream)
        device.sync()

        verify_buffer.close()
        src_buffer.close()
        buffer.close()

        queue.put(True)
    except Exception as e:
        # Capture the full traceback
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        queue.put((e, tb_str))


def test_ipc_mempool():
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")

    # Check if IPC is supported on this platform/device
    device = Device()
    device.set_current()
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    if platform.system() == "Windows":
        pytest.skip("IPC is not supported on Windows")

    # Set multiprocessing start method before creating any multiprocessing objects
    multiprocessing.set_start_method("spawn", force=True)

    stream = device.create_stream()
    pool_size = 2097152  # 2MB size
    mr = DeviceMemoryResource(device, dict(max_size=pool_size, ipc_enabled=True))
    assert mr.is_ipc_enabled

    # Create socket pair for handle transfer (only on Unix systems)
    exporter = None
    importer = None
    if platform.system() == "Linux":
        exporter, importer = socketpair(AF_UNIX, SOCK_DGRAM)

    queue = multiprocessing.Queue()
    process = None

    try:
        alloc_handle = mr.get_allocation_handle()

        # Allocate and export memory
        buffer = mr.allocate(64)

        try:
            # Fill buffer with test pattern using unified memory
            unified_mr = DummyUnifiedMemoryResource(device)
            src_buffer = unified_mr.allocate(64)
            try:
                src_ptr = ctypes.cast(int(src_buffer.handle), ctypes.POINTER(ctypes.c_byte))
                for i in range(64):
                    src_ptr[i] = ctypes.c_byte(i)

                buffer.copy_from(src_buffer, stream=stream)
                device.sync()
            finally:
                src_buffer.close()

            # Export buffer for IPC
            ipc_buffer = mr.export_buffer(buffer)

            # Start child process
            process = multiprocessing.Process(
                target=mempool_child_process, args=(importer if platform.system() == "Linux" else None, queue)
            )
            process.start()

            # Send handles to child process
            if platform.system() == "Windows":
                queue.put(alloc_handle)  # Send handle through queue on Windows
            else:
                # Use Unix socket for handle transfer
                exporter.sendmsg([], [(SOL_SOCKET, SCM_RIGHTS, array.array("i", [alloc_handle]))])

            queue.put(ipc_buffer)

            # Wait for child process
            process.join(timeout=10)
            assert process.exitcode == 0

            # Check for exceptions
            if not queue.empty():
                result = queue.get()
                if isinstance(result, tuple):
                    exception, traceback_str = result
                    print("\nException in child process:")
                    print(traceback_str)
                    raise exception
                assert result is True

            # Verify child process wrote the inverted pattern using unified memory
            verify_buffer = unified_mr.allocate(64)
            try:
                verify_buffer.copy_from(buffer, stream=stream)
                device.sync()

                verify_ptr = ctypes.cast(int(verify_buffer.handle), ctypes.POINTER(ctypes.c_byte))
                for i in range(64):
                    assert ctypes.c_byte(verify_ptr[i]).value == ctypes.c_byte(255 - i).value, (
                        f"Child process data not reflected in parent at index {i}"
                    )
            finally:
                verify_buffer.close()

        finally:
            mr.close_allocation_handle(alloc_handle)
            buffer.close()

    finally:
        # Clean up all resources
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1)
        queue.close()
        queue.join_thread()
        if exporter is not None:
            exporter.close()
        if importer is not None:
            importer.close()
        # Flush any pending operations
        flush_buffer = mr.allocate(64)
        flush_buffer.close()
