# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing

import pytest
from utility import IPCBufferTestHelper

from cuda.core.experimental import Buffer, DeviceMemoryResource

CHILD_TIMEOUT_SEC = 10
NBYTES = 64
POOL_SIZE = 2097152


class TestObjectSerialization:
    @pytest.mark.parametrize("use_alloc_handle", [True, False])
    def test_main(self, use_alloc_handle, device, ipc_memory_resource):
        """Test sending IPC memory objects to a child through a queue."""
        mr = ipc_memory_resource

        # Start the child process.
        pipe = [multiprocessing.Queue() for _ in range(2)]
        process = multiprocessing.Process(target=self.child_main, args=(pipe, use_alloc_handle))
        process.start()

        # Send a device description.
        pipe[0].put(device)
        device_id = pipe[1].get()
        assert device_id == device.device_id

        # Send a memory resource directly or by allocation handle.
        # Note: there is no apparent way to check the ID between processes.
        if use_alloc_handle:
            # Send MR by a handle.
            alloc_handle = mr.get_allocation_handle()
            pipe[0].put(alloc_handle)
        else:
            # Send MR directly.
            pipe[0].put(mr)

        # Send a buffer.
        buffer = mr.allocate(NBYTES)
        helper = IPCBufferTestHelper(device, buffer)
        helper.fill_buffer(flipped=False)
        pipe[0].put(buffer)
        pipe[1].get()  # signal done
        helper.verify_buffer(flipped=True)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

    def child_main(self, pipe, use_alloc_handle):
        # Device.
        device = pipe[0].get()
        pipe[1].put(device.device_id)

        # Memory resource.
        if use_alloc_handle:
            alloc_handle = pipe[0].get()
            mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
        else:
            mr = pipe[0].get()

        # Buffer.
        buffer = pipe[0].get()
        assert buffer.memory_resource.handle == mr.handle
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=False)
        helper.fill_buffer(flipped=True)
        pipe[1].put(None)


def test_object_passing(device, ipc_memory_resource):
    """Test sending objects as arguments when starting a process."""
    # Define the objects.
    mr = ipc_memory_resource
    alloc_handle = mr.get_allocation_handle()
    buffer = mr.allocate(NBYTES)
    buffer_desc = buffer.export()

    helper = IPCBufferTestHelper(device, buffer)
    helper.fill_buffer(flipped=False)

    # Start the child process.
    process = multiprocessing.Process(target=child_main, args=(device, alloc_handle, mr, buffer_desc, buffer))
    process.start()
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    helper.verify_buffer(flipped=True)


def child_main(device, alloc_handle, mr1, buffer_desc, buffer1):
    mr2 = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)

    # OK to build the buffer from either mr and descriptor.
    # These all point to the same buffer.
    buffer2 = Buffer.import_(mr1, buffer_desc)
    buffer3 = Buffer.import_(mr2, buffer_desc)

    helper1 = IPCBufferTestHelper(device, buffer1)
    helper2 = IPCBufferTestHelper(device, buffer2)
    helper3 = IPCBufferTestHelper(device, buffer3)

    helper1.verify_buffer(flipped=False)
    helper2.verify_buffer(flipped=False)
    helper3.verify_buffer(flipped=False)

    helper1.fill_buffer(flipped=True)

    helper1.verify_buffer(flipped=True)
    helper2.verify_buffer(flipped=True)
    helper3.verify_buffer(flipped=True)

    helper2.fill_buffer(flipped=False)

    helper1.verify_buffer(flipped=False)
    helper2.verify_buffer(flipped=False)
    helper3.verify_buffer(flipped=False)

    helper3.fill_buffer(flipped=True)

    helper1.verify_buffer(flipped=True)
    helper2.verify_buffer(flipped=True)
    helper3.verify_buffer(flipped=True)
