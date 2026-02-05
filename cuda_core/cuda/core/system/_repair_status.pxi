# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class RepairStatus:
    """
    Repair status for TPC/Channel repair.
    """
    cdef object _repair_status

    def __init__(self, handle: int):
        self._repair_status = nvml.device_get_repair_status(handle)

    @property
    def channel_repair_pending(self) -> bool:
        """
        `True` if a channel repair is pending.
        """
        return bool(self._repair_status.b_channel_repair_pending)

    @property
    def tpc_repair_pending(self) -> bool:
        """
        `True` if a TPC repair is pending.
        """
        return bool(self._repair_status.b_tpc_repair_pending)
