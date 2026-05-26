# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import pickle
import re

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.core import Buffer, Device, DeviceMemoryResource, DeviceMemoryResourceOptions
from cuda.core._utils.cuda_utils import CUDAError

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64
POOL_SIZE = 2097152


def test_outer_timeout_marker_is_applied(request):
    """Verify that memory_ipc/conftest.py applies the outer pytest-timeout marker.

    If this test fails, the per-directory conftest is not being loaded, or its
    pytest_collection_modifyitems hook is not adding the marker. Without this
    marker, the only thing protecting the GHA runner from a wedged IPC test is
    the in-test cleanup -- which we want to keep as defense in depth, not as
    the sole guard.
    """
    expected = child_timeout_sec() + 30
    marker = request.node.get_closest_marker("timeout")
    assert marker is not None, "memory_ipc/conftest.py did not apply a timeout marker"
    assert marker.args == (expected,), f"unexpected timeout value: {marker.args!r}"


class ChildErrorHarness:
    """Test harness for checking errors in child processes. Subclasses override
    PARENT_ACTION, CHILD_ACTION, and ASSERT (see below for examples)."""

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        """Parent process that checks child errors."""
        # Attach fixtures to this object for convenience. These can be accessed
        # from PARENT_ACTION.
        self.device = ipc_device
        self.mr = ipc_memory_resource
        self._extra_mrs = []

        try:
            # Start a child process to generate error info.
            pipe = [multiprocessing.Queue() for _ in range(2)]
            process = multiprocessing.Process(target=self.child_main, args=(pipe, self.device, self.mr))
            process.start()

            # Interact.
            self.PARENT_ACTION(pipe[0])

            # Check the error.
            exc_type, exc_msg = pipe[1].get(timeout=CHILD_TIMEOUT_SEC)
            self.ASSERT(exc_type, exc_msg)

            # Wait for the child process.
            process.join(timeout=CHILD_TIMEOUT_SEC)
            survivors = kill_subprocesses(process)
            assert not survivors, "child did not exit within timeout"
            assert process.exitcode == 0
        finally:
            for mr in self._extra_mrs:
                mr.close()

    def child_main(self, pipe, device, mr):
        """Child process that pushes IPC errors to a shared pipe for testing."""
        self.device = device
        self.device.set_current()
        self.mr = mr
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
        mr = queue.get(timeout=CHILD_TIMEOUT_SEC)
        mr.allocate(NBYTES, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is TypeError
        assert exc_msg == "Cannot allocate from a mapped IPC-enabled memory resource"


class TestImportWrongMR(ChildErrorHarness):
    """Error when importing a buffer from the wrong memory resource."""

    def PARENT_ACTION(self, queue):
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr2 = DeviceMemoryResource(self.device, options=options)
        self._extra_mrs.append(mr2)
        buffer = mr2.allocate(NBYTES, stream=self.device.default_stream)
        queue.put([self.mr, buffer.ipc_descriptor])  # Note: mr does not own this buffer

    def CHILD_ACTION(self, queue):
        mr, buffer_desc = queue.get(timeout=CHILD_TIMEOUT_SEC)
        Buffer.from_ipc_descriptor(mr, buffer_desc, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is CUDAError
        assert "CUDA_ERROR_INVALID_VALUE" in exc_msg


class TestImportBuffer(ChildErrorHarness):
    """Error when using a buffer as a buffer descriptor."""

    def PARENT_ACTION(self, queue):
        # Note: if the buffer is not attached to something to prolong its life,
        # CUDA_ERROR_INVALID_CONTEXT is raised from Buffer.__del__
        self.buffer = self.mr.allocate(NBYTES, stream=self.device.default_stream)
        queue.put(self.buffer)

    def CHILD_ACTION(self, queue):
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        Buffer.from_ipc_descriptor(self.mr, buffer, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is TypeError
        assert exc_msg.startswith("Argument 'ipc_descriptor' has incorrect type")


class TestDanglingBuffer(ChildErrorHarness):
    """
    Error when importing a buffer object without registering its memory
    resource.
    """

    def PARENT_ACTION(self, queue):
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr2 = DeviceMemoryResource(self.device, options=options)
        self._extra_mrs.append(mr2)
        self.buffer = mr2.allocate(NBYTES, stream=self.device.default_stream)
        buffer_s = pickle.dumps(self.buffer)
        queue.put(buffer_s)  # Note: mr2 not sent

    def CHILD_ACTION(self, queue):
        Device().set_current()
        buffer_s = queue.get(timeout=CHILD_TIMEOUT_SEC)
        pickle.loads(buffer_s)  # noqa: S301

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is RuntimeError
        assert re.match(r"Memory resource [a-z0-9-]+ was not found", exc_msg)
