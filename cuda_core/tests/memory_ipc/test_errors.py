# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing

from cuda.core.experimental import Buffer, DeviceMemoryResource
from cuda.core.experimental._utils.cuda_utils import CUDAError

CHILD_TIMEOUT_SEC = 10
NBYTES = 64
POOL_SIZE = 2097152


class ChildErrorHarness:
    """Test harness for checking errors in child processes. Subclasses override
    PARENT_ACTION, CHILD_ACTION, and ASSERT (see below for examples)."""

    def test_main(self, device, ipc_memory_resource):
        """Parent process that checks child errors."""
        # Attach fixtures to this object for convenience. These can be accessed
        # from PARENT_ACTION.
        self.device = device
        self.mr = ipc_memory_resource

        # Start a child process to generate error info.
        pipe = [multiprocessing.Queue() for _ in range(2)]
        process = multiprocessing.Process(target=self.child_main, args=(pipe,))
        process.start()

        # Interact.
        self.PARENT_ACTION(pipe[0])

        # Check the error.
        exc_type, exc_msg = pipe[1].get(timeout=CHILD_TIMEOUT_SEC)
        self.ASSERT(exc_type, exc_msg)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

    def child_main(self, pipe):
        """Child process that pushes IPC errors to a shared pipe for testing."""
        try:
            self.CHILD_ACTION(pipe[0])
        except Exception as e:
            exc_info = type(e), str(e)
        else:
            exc_info = None, None
        pipe[1].put(exc_info)


class TestAllocFromImportedMr(ChildErrorHarness):
    """Error when attempting to allocate from an import memory resource."""

    def PARENT_ACTION(self, queue):
        queue.put(self.mr)

    def CHILD_ACTION(self, queue):
        mr = queue.get()
        mr.allocate(NBYTES)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is TypeError
        assert exc_msg == "Cannot allocate from shared memory pool imported via IPC"


class TestImportWrongMR(ChildErrorHarness):
    """Error when importing a buffer from the wrong memory resource."""

    def PARENT_ACTION(self, queue):
        mr2 = DeviceMemoryResource(self.device, dict(max_size=POOL_SIZE, ipc_enabled=True))
        buffer = mr2.allocate(NBYTES)
        queue.put([self.mr, buffer.export()])  # Note: mr does not own this buffer

    def CHILD_ACTION(self, queue):
        mr, buffer_desc = queue.get()
        Buffer.import_(mr, buffer_desc)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is CUDAError
        assert "CUDA_ERROR_INVALID_VALUE" in exc_msg
