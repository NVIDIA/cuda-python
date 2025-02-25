# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import traceback

import pytest

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import array
import ctypes
import multiprocessing
from socket import AF_UNIX, CMSG_LEN, SCM_RIGHTS, SOCK_DGRAM, SOL_SOCKET, socketpair

from cuda.core.experimental import Device
from cuda.core.experimental._memory import AsyncMempool, Buffer, MemoryResource
from cuda.core.experimental._utils import get_binding_version, handle_return


class DummyDeviceMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer(ptr=ptr, size=size, mr=self)

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

    def allocate(self, size, stream=None) -> Buffer:
        ptr = (ctypes.c_byte * size)()
        return Buffer(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        pass

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

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocManaged(size, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value))
        return Buffer(ptr=ptr, size=size, mr=self)

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

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocHost(size))
        return Buffer(ptr=ptr, size=size, mr=self)

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


# Basic Buffer Tests
def buffer_initialization(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=64)
    assert buffer.handle != 0
    assert buffer.size == 64
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
    src_buffer = dummy_mr.allocate(size=64)
    dst_buffer = dummy_mr.allocate(size=64)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(64):
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
    src_buffer = dummy_mr.allocate(size=64)
    dst_buffer = dummy_mr.allocate(size=64)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(64):
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
    buffer = dummy_mr.allocate(size=64)
    buffer.close()
    assert buffer.handle == 0
    assert buffer.memory_resource is None


def test_buffer_close():
    device = Device()
    device.set_current()
    buffer_close(DummyDeviceMemoryResource(device))
    buffer_close(DummyHostMemoryResource())
    buffer_close(DummyUnifiedMemoryResource(device))
    buffer_close(DummyPinnedMemoryResource(device))


def test_mempool():
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")
    device = Device()
    device.set_current()
    pool_size = 2097152  # 2MB size

    # Test basic pool creation
    mr = AsyncMempool.create(device.device_id, pool_size, ipc_enabled=False)
    assert mr.device_id == device.device_id
    assert mr.is_device_accessible
    assert not mr.is_host_accessible

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
    with pytest.raises(NotImplementedError, match="directly creating an AsyncMempool object is not supported"):
        AsyncMempool()

    with pytest.raises(ValueError, match="max_size must be provided when creating a new memory pool"):
        AsyncMempool.create(device.device_id, None)

    # Test IPC operations are disabled
    buffer = mr.allocate(64)

    with pytest.raises(RuntimeError, match="This memory pool was not created with IPC support enabled"):
        mr.get_shareable_handle()

    with pytest.raises(RuntimeError, match="This memory pool was not created with IPC support enabled"):
        mr.export_buffer(buffer)

    with pytest.raises(RuntimeError, match="This memory pool was not created with IPC support enabled"):
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
    """Test all properties of the AsyncMempool class."""
    # skip test if cuda version is less than 12
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")

    device = Device()
    device.set_current()
    pool_size = 2097152  # 2MB size
    mr = AsyncMempool.create(device.device_id, pool_size, ipc_enabled=False)

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

        # Receive the handle via socket
        fds = array.array("i")
        _, ancdata, _, _ = importer.recvmsg(0, CMSG_LEN(fds.itemsize))
        assert len(ancdata) == 1
        cmsg_level, cmsg_type, cmsg_data = ancdata[0]
        assert cmsg_level == SOL_SOCKET and cmsg_type == SCM_RIGHTS
        fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
        shared_handle = int(fds[0])

        mr = AsyncMempool.from_shared_handle(device.device_id, shared_handle)
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
    # Set multiprocessing start method before creating any multiprocessing objects
    multiprocessing.set_start_method("spawn", force=True)

    device = Device()
    device.set_current()
    stream = device.create_stream()
    pool_size = 2097152  # 2MB size
    mr = AsyncMempool.create(device.device_id, pool_size, ipc_enabled=True)

    # Create socket pair for handle transfer
    exporter, importer = socketpair(AF_UNIX, SOCK_DGRAM)
    queue = multiprocessing.Queue()
    process = None

    try:
        shareable_handle = mr.get_shareable_handle()

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
            process = multiprocessing.Process(target=mempool_child_process, args=(importer, queue))
            process.start()

            # Send handles to child process
            exporter.sendmsg([], [(SOL_SOCKET, SCM_RIGHTS, array.array("i", [shareable_handle]))])
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
                    assert (
                        ctypes.c_byte(verify_ptr[i]).value == ctypes.c_byte(255 - i).value
                    ), f"Child process data not reflected in parent at index {i}"
            finally:
                verify_buffer.close()

        finally:
            buffer.close()

    finally:
        # Clean up all resources
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1)
        queue.close()
        queue.join_thread()  # Ensure the queue's background thread is cleaned up
        exporter.close()
        importer.close()
        # Flush any pending operations
        flush_buffer = mr.allocate(64)
        flush_buffer.close()
