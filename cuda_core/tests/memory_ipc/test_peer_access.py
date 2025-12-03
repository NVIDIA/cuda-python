# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

from cuda.core.experimental import Device, DeviceMemoryResource, DeviceMemoryResourceOptions

CHILD_TIMEOUT_SEC = 20
POOL_SIZE = 2097152


class TestPeerAccessNotPreservedOnImport:
    """
    Verify that peer access settings are not preserved when a memory resource
    is sent to another process via IPC, and that peer access can be set after import.
    """

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
