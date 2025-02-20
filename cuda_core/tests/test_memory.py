# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

try:
    from cuda.bindings import driver, runtime
except ImportError:
    from cuda import cuda as driver
    from cuda import cudart as runtime

import array
import ctypes
import multiprocessing
from socket import AF_UNIX, CMSG_LEN, SCM_RIGHTS, SOCK_DGRAM, SOL_SOCKET, socketpair

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


def child_process(importer, queue):
    try:
        device = Device()
        device.set_current()

        # Receive the handle via socket
        fds = array.array("i")
        _, ancdata, _, _ = importer.recvmsg(0, CMSG_LEN(fds.itemsize))
        assert len(ancdata) == 1
        cmsg_level, cmsg_type, cmsg_data = ancdata[0]
        assert cmsg_level == SOL_SOCKET and cmsg_type == SCM_RIGHTS
        fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
        shared_handle = int(fds[0])

        mr = SharedMempool.from_shared_handle(device.device_id, shared_handle)
        # Get the exported pointer data from the queue
        export_data = queue.get()
        buffer = mr.import_pointer(export_data)

        # # Verify we can read the data
        # data = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_byte))
        # for i in range(64):
        #     assert data[i] == i % 256

        queue.put(True)
        buffer.close()
    except Exception as e:
        queue.put(e)


def test_shared_memory_resource():
    device = Device()
    device.set_current()
    pool_size = 2097152  # Keep consistent 2MB size
    mr = SharedMempool.create(device.device_id, pool_size)
    shareable_handle = mr.get_shareable_handle()

    # Allocate and initialize memory
    buffer = mr.allocate(64)
    # ptr = ctypes.cast(buffer.handle, ctypes.POINTER(ctypes.c_byte))
    # for i in range(64):
    #     ptr[i] = ctypes.c_byte(i % 256)
    # device.sync()

    # Export the pointer
    shareable_buffer = mr.export_pointer(buffer.handle)

    # Create socket pair for handle transfer
    exporter, importer = socketpair(AF_UNIX, SOCK_DGRAM)

    multiprocessing.set_start_method("spawn", force=True)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_process, args=(importer, queue))
    process.start()

    # Send the handle via socket
    exporter.sendmsg([], [(SOL_SOCKET, SCM_RIGHTS, array.array("i", [shareable_handle]))])

    # Send the exported pointer data through the queue
    queue.put(shareable_buffer)

    process.join(timeout=10)
    assert process.exitcode == 0

    if not queue.empty():
        exception = queue.get()
        if isinstance(exception, Exception):
            raise exception


def child_process_allocator(size, importer, queue):
    device = Device(0)
    device.set_current()

    handle_return(runtime.cudaGetLastError())

    alloc = ShareableAllocator(device.device_id)
    handle_return(runtime.cudaGetLastError())

    fds = array.array("i")
    _, ancdata, _, _ = importer.recvmsg(0, CMSG_LEN(fds.itemsize))
    assert len(ancdata) == 1
    cmsg_level, cmsg_type, cmsg_data = ancdata[0]
    assert cmsg_level == SOL_SOCKET and cmsg_type == SCM_RIGHTS
    fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
    handle = int(fds[0])

    try:
        buffer = alloc.import_shareable_allocation(size, handle)
        assert buffer.handle != 0
        assert buffer.size == size
        assert buffer.memory_resource == alloc
        assert buffer.is_device_accessible
        assert not buffer.is_host_accessible
        assert buffer.device_id == device.device_id
        buffer.close()
        queue.put(True)
    except Exception as e:
        queue.put(e)


def test_shareable_allocator():
    device = Device(0)
    device.set_current()
    alloc = ShareableAllocator(device.device_id)
    size = 2097152
    buffer, handle = alloc.get_shareable_allocation(size)
    assert buffer.handle != 0
    assert buffer.size == size
    assert buffer.memory_resource == alloc
    assert buffer.is_device_accessible
    assert not buffer.is_host_accessible
    assert buffer.device_id == device.device_id

    exporter, importer = socketpair(AF_UNIX, SOCK_DGRAM)

    multiprocessing.set_start_method("spawn", force=True)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_process_allocator, args=(size, importer, queue))
    process.start()

    exporter.sendmsg([], [(SOL_SOCKET, SCM_RIGHTS, array.array("i", [handle]))])

    process.join(timeout=10)
    assert process.exitcode == 0

    if not queue.empty():
        exception = queue.get()
        if isinstance(exception, Exception):
            raise exception

    buffer.close()


# def child_process_pointer(importer, queue):
#     try:
#         device = Device()
#         device.set_current()
#         stream = device.create_stream()

#         # Receive the pool handle via socket
#         fds = array.array("i")
#         _, ancdata, _, _ = importer.recvmsg(0, CMSG_LEN(fds.itemsize))
#         assert len(ancdata) == 1
#         cmsg_level, cmsg_type, cmsg_data = ancdata[0]
#         assert cmsg_level == SOL_SOCKET and cmsg_type == SCM_RIGHTS
#         fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
#         shared_handle = int(fds[0])

#         # Import the pool
#         mr = SharedMempool.from_shared_handle(device.device_id, shared_handle)

#         # Receive the pointer export data
#         export_data = queue.get()
#         assert not isinstance(export_data, Exception)

#         # Import the pointer
#         ptr = mr.import_pointer(export_data)
#         assert ptr != 0

#         # Verify we can read the data
#         data = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_byte))
#         for i in range(64):
#             assert data[i] == i % 256

#         # Signal success
#         queue.put(True)

#         # Clean up
#         handle_return(driver.cuMemFreeAsync(ptr, stream.handle))
#         device.sync()

#     except Exception as e:
#         queue.put(e)


# def test_shared_memory_pointer():
#     device = Device()
#     device.set_current()
#     stream = device.create_stream()
#     pool_size = 2097152  # 2MB size
#     mr = SharedMempool.create(device.device_id, pool_size)

#     # Allocate and initialize memory
#     buffer = mr.allocate(64, stream=stream)
#     ptr = ctypes.cast(buffer.handle, ctypes.POINTER(ctypes.c_byte))
#     for i in range(64):
#         ptr[i] = ctypes.c_byte(i % 256)
#     device.sync()

#     # Export the pointer
#     export_data = mr.export_pointer(buffer.handle)

#     # Get shareable handle for the pool
#     shareable_handle = mr.get_shareable_handle()

#     # Create socket pair for handle transfer
#     exporter, importer = socketpair(AF_UNIX, SOCK_DGRAM)

#     # Start child process
#     multiprocessing.set_start_method("spawn", force=True)
#     queue = multiprocessing.Queue()
#     process = multiprocessing.Process(target=child_process_pointer, args=(importer, queue))
#     process.start()

#     # Send the pool handle via socket
#     exporter.sendmsg([], [(SOL_SOCKET, SCM_RIGHTS, array.array("i", [shareable_handle]))])

#     # Send the pointer export data via queue
#     queue.put(export_data)

#     # Wait for child process to finish
#     process.join(timeout=10)
#     assert process.exitcode == 0

#     # Check for exceptions from child process
#     if not queue.empty():
#         result = queue.get()
#         if isinstance(result, Exception):
#             raise result
#         assert result is True  # Success flag

#     # Clean up
#     buffer.close()
#     device.sync()
