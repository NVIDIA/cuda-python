# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

import pytest
from cuda.core import Device, DeviceMemoryResource, DeviceMemoryResourceOptions
from cuda.core._utils.cuda_utils import CUDAError
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
POOL_SIZE = 2097152


class TestPeerAccessNotPreservedOnImport:
    """
    Verify that peer access settings are not preserved when a memory resource
    is sent to another process via IPC, and that peer access can be set after import.
    """

    @pytest.mark.flaky(reruns=2)
    def test_main(self, mempool_device_x2):
        dev0, dev1 = mempool_device_x2

        # Parent Process - Create and Configure MR
        dev1.set_current()
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr = DeviceMemoryResource(dev1, options=options)
        mr.peer_accessible_by = [dev0]
        assert mr.peer_accessible_by == (0,)

        # Spawn child process
        process = mp.Process(target=self.child_main, args=(mr,))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Verify parent's MR still has peer access set (independent state)
        assert mr.peer_accessible_by == (0,)
        mr.close()

    def child_main(self, mr):
        Device(1).set_current()
        assert mr.is_mapped is True
        assert mr.device_id == 1
        assert mr.peer_accessible_by == ()
        mr.peer_accessible_by = [0]
        assert mr.peer_accessible_by == (0,)
        mr.peer_accessible_by = []
        assert mr.peer_accessible_by == ()
        mr.close()


class TestBufferPeerAccessAfterImport:
    """
    Verify that buffers imported via IPC can be accessed from peer devices after
    setting peer access on the imported memory resource, and that access can be revoked.
    """

    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("grant_access_in_parent", [True, False])
    def test_main(self, mempool_device_x2, grant_access_in_parent):
        dev0, dev1 = mempool_device_x2

        # Parent Process - Create MR and Buffer
        dev1.set_current()
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr = DeviceMemoryResource(dev1, options=options)
        if grant_access_in_parent:
            mr.peer_accessible_by = [dev0]
            assert mr.peer_accessible_by == (0,)
        else:
            assert mr.peer_accessible_by == ()
        buffer = mr.allocate(NBYTES)
        pgen = PatternGen(dev1, NBYTES)
        pgen.fill_buffer(buffer, seed=False)

        # Spawn child process
        process = mp.Process(target=self.child_main, args=(mr, buffer))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        buffer.close()
        mr.close()

    def child_main(self, mr, buffer):
        # Verify MR and buffer are mapped
        Device(1).set_current()
        assert mr.is_mapped is True
        assert buffer.is_mapped is True
        assert mr.device_id == 1
        assert buffer.device_id == 1

        # Test 1: Buffer accessible from resident device (dev1) - should always work
        dev1 = Device(1)
        dev1.set_current()
        PatternGen(dev1, NBYTES).verify_buffer(buffer, seed=False)

        # Test 2: Buffer NOT accessible from dev0 initially (peer access not preserved)
        dev0 = Device(0)
        dev0.set_current()
        with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
            PatternGen(dev0, NBYTES).verify_buffer(buffer, seed=False)

        # Test 3: Set peer access and verify buffer becomes accessible
        dev1.set_current()
        mr.peer_accessible_by = [0]
        assert mr.peer_accessible_by == (0,)
        dev0.set_current()
        PatternGen(dev0, NBYTES).verify_buffer(buffer, seed=False)

        # Test 4: Revoke peer access and verify buffer becomes inaccessible
        dev1.set_current()
        mr.peer_accessible_by = []
        assert mr.peer_accessible_by == ()
        dev0.set_current()
        with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
            PatternGen(dev0, NBYTES).verify_buffer(buffer, seed=False)

        buffer.close()
        mr.close()
