# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes
import multiprocessing

from cuda.core.experimental import Device
from cuda.core.experimental._memory import Buffer, MemoryResource, ShareableAllocator, SharedMempool
from cuda.core.experimental._utils import handle_return


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
        # Allocate a ctypes buffer of size `size`
        ptr = (ctypes.c_byte * size)()
        return Buffer(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        # the memory is deallocated per the ctypes deallocation at garbage collection time
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


def child_process(shared_handle, queue):
    """Child process function that writes data to shared memory."""
    try:
        from cuda.core.experimental import Device

        device = Device()
        device.set_current()

        # Import the shared memory pool
        print("creating a shared memory pool from a handle")
        mr = SharedMempool(0, shared_handle=shared_handle)

        # Allocate and write to buffer
        buffer = mr.allocate(64)
        ptr = ctypes.cast(buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(64):
            ptr[i] = ctypes.c_byte(i % 256)

        # Signal parent process that data is ready
        queue.put("Data written")

        # Wait for parent to read
        assert queue.get() == "Data read"

        buffer.close()

    except Exception as e:
        queue.put(e)
        raise


# def parent_process(device_id, shared_handle, queue):
#     """Parent process function that reads and verifies data from shared memory."""
#     try:
#         # Import the shared memory pool
#         mr = SharedMempool(device_id, shared_handle=shared_handle)

#         # Wait for child to write data
#         assert queue.get() == "Data written"

#         # Read and verify data
#         buffer = mr.allocate(64)
#         ptr = ctypes.cast(buffer.handle, ctypes.POINTER(ctypes.c_byte))
#         for i in range(64):
#             assert ptr[i] == ctypes.c_byte(i % 256), f"Mismatch at index {i}"

#         # Signal child that we've read the data
#         queue.put("Data read")

#         buffer.close()

#     except Exception as e:
#         queue.put(e)
#         raise


def test_shared_memory_resource():
    """Test shared memory pool functionality across processes."""

    # Initialize device
    device = Device()
    device.set_current()

    # Create shared memory pool
    pool_size = 64 * 64  # 1MB
    mr = SharedMempool(0, max_size=pool_size)

    # Test basic allocation
    # buffer = mr.allocate(64)
    # assert buffer.handle != 0
    # assert buffer.size == 64
    # assert buffer.memory_resource == mr
    # assert buffer.is_device_accessible
    # assert not buffer.is_host_accessible
    # buffer.close()

    # Get shareable handle
    shareable_handle = mr.get_shareable_handle()
    print("shareable handle: ", shareable_handle)
    assert shareable_handle != 0

    # try importing on the same process
    imported_pool = SharedMempool(0, shared_handle=shareable_handle)
    print("imported pool: ", imported_pool)

    # Test cross-process sharing
    multiprocessing.set_start_method("spawn", force=True)
    queue = multiprocessing.Queue()

    # Create child process
    process = multiprocessing.Process(target=child_process, args=(shareable_handle, queue))
    process.start()

    # Wait for child process to complete
    process.join(timeout=10)
    assert process.exitcode == 0, "Child process failed"
    print("child process done")

    # Run parent process logic
    print("creating a shared mempool for a sharable handle within the same parent proc")
    # parent_process(0, shareable_handle, queue)

    # Check for any exceptions from the child process
    if not queue.empty():
        exception = queue.get()
        if isinstance(exception, Exception):
            raise exception


def child_process_allocator(size, handle, queue):
    try:
        # Create context with flags=0 for default stream behavior
        driver.cuCtxCreate(0, 0)

        # Create allocator and import buffer
        alloc = ShareableAllocator(0)
        imported_buffer = alloc.import_shareable_allocation(size, handle)

        # Verify imported buffer properties
        assert imported_buffer.handle != 0
        assert imported_buffer.size == size
        assert imported_buffer.memory_resource == alloc
        assert imported_buffer.is_device_accessible
        assert not imported_buffer.is_host_accessible
        assert imported_buffer.device_id == 0

        # Clean up
        imported_buffer.close()
        queue.put(None)  # Signal success

    except Exception as e:
        queue.put(e)


def test_sharable_allocator():
    print("\nTesting ShareableAllocator...")

    # Initialize device
    print("Initializing device...")

    # Create context with flags=0 for default stream behavior
    driver.cuCtxCreate(0, 0)

    print(f"Using device {0}")

    # Create allocator and get sharable allocation
    print("Creating ShareableAllocator...")
    alloc = ShareableAllocator(0)
    size = 2097152
    print(f"Getting shareable allocation of size {size} bytes...")
    buffer, handle = alloc.get_shareable_allocation(size)
    print(f"Got buffer with handle {buffer.handle} and shareable handle {handle}")

    # Verify original buffer properties
    print("Verifying buffer properties...")
    assert buffer.handle != 0
    assert buffer.size == size
    assert buffer.memory_resource == alloc
    assert buffer.is_device_accessible
    assert not buffer.is_host_accessible
    assert buffer.device_id == 0
    print("Buffer properties verified successfully")

    print("mock sharing on same process")
    alloc = ShareableAllocator(0)
    imported_buffer = alloc.import_shareable_allocation(size, handle)
    print(imported_buffer)
    print("done mock sharing on same process")
    # Test cross-process sharing
    print("Testing cross-process sharing...")
    queue = multiprocessing.Queue()
    print("Creating child process...")
    process = multiprocessing.Process(target=child_process_allocator, args=(size, handle, queue))
    print("Starting child process...")
    process.start()
    print("Waiting for child process to complete...")
    process.join(timeout=10)

    # Check child process results
    print("Checking child process results...")
    assert process.exitcode == 0, "Child process failed"
    if not queue.empty():
        exception = queue.get()
        if isinstance(exception, Exception):
            raise exception
    print("Child process completed successfully")

    # Clean up
    print("Cleaning up...")
    buffer.close()
    print("ShareableAllocator tests passed")
