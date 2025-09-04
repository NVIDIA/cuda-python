try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes
import multiprocessing
import platform
import pytest

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, IPCChannel, MemoryResource
from cuda.core.experimental._memory import DLDeviceType
from cuda.core.experimental._utils.cuda_utils import get_binding_version, handle_return

POOL_SIZE = 2097152  # 2MB size
NBYTES = 64

@pytest.fixture(scope="function")
def ipc_device():
    """Obtains a device suitable for IPC-enabled mempool tests, or skips."""
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")

    if platform.system() == "Windows":
        pytest.skip("IPC is not supported on Windows")

    # Check if IPC is supported on this platform/device
    device = Device()
    device.set_current()
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")
    return device

def test_ipc_mempool(ipc_device):
    # Set up the IPC-enabled memory pool and share it.
    mr = DeviceMemoryResource(ipc_device, dict(max_size=POOL_SIZE, ipc_enabled=True))
    assert mr.is_ipc_enabled
    channel = IPCChannel()
    mr.share_to_channel(channel)
    try:
        # Start the child process.
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=child_main, args=(channel, queue))
        process.start()

        # Allocate and fill memory.
        buffer = mr.allocate(NBYTES)
        protocol = IPCBufferTestProtocol(ipc_device, buffer)
        protocol.fill_buffer(flipped=False)

        # Export buffer via IPC.
        handle = mr.export_buffer(buffer)
        queue.put(handle)

        # Wait for the child process.
        process.join(timeout=10)
        assert process.exitcode == 0

        # Verify that the buffer was modified.
        protocol.verify_buffer(flipped=True)
    finally:
        if locals().get('buffer') is not None:
            buffer.close()
        if locals().get('process') is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if locals().get('queue') is not None:
            queue.close()
            queue.join_thread()
        mr.allocate(NBYTES).close() # Flush any pending operations

def child_main(channel, queue):
    device = Device()
    device.set_current()
    try:
        mr = DeviceMemoryResource.from_shared_channel(device, channel)
        handle = queue.get()  # Get exported buffer data
        buffer = mr.import_buffer(handle)

        protocol = IPCBufferTestProtocol(device, buffer)
        protocol.verify_buffer(flipped=False)
        protocol.fill_buffer(flipped=True)
    finally:
        if locals().get('buffer') is not None:
            buffer.close()

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
        return self.device

class IPCBufferTestProtocol:
    """The protocol for verifying IPC.

    Provides methods to fill a buffer with one of two test patterns and verify
    the expected values.
    """

    def __init__(self, device, buffer, nbytes=NBYTES, stream=None):
        self.device = device
        self.buffer = buffer
        self.nbytes = nbytes
        self.stream = stream if stream is not None else device.create_stream()
        self.scratch_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.nbytes, stream=self.stream)

    def fill_buffer(self, flipped=False):
        """Fill a device buffer with test pattern using unified memory."""
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.nbytes):
            ptr[i] = ctypes.c_byte(op(i))
        self.buffer.copy_from(self.scratch_buffer, stream=self.stream)

    def verify_buffer(self, flipped=False):
        """Verify the buffer contents."""
        self.scratch_buffer.copy_from(self.buffer, stream=self.stream)
        self.device.sync()
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.nbytes):
            assert ctypes.c_byte(ptr[i]).value == ctypes.c_byte(op(i)).value, (
                f"Buffer contains incorrect data at index {i}"
            )

