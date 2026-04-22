# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class ProcessInfo:
    """
    Information about running compute processes on the GPU.
    """
    def __init__(self, device: "Device", process_info: nvml.ProcessInfo):
        self._device = device
        self._process_info = process_info

    @property
    def pid(self) -> int:
        """
        The PID of the process.
        """
        return self._process_info.pid

    @property
    def used_gpu_memory(self) -> int:
        """
        The amount of GPU memory (in bytes) used by the process.
        """
        return self._process_info.used_gpu_memory

    @property
    def gpu_instance_id(self) -> int:
        """
        The GPU instance ID for MIG devices.

        Only valid for processes running on MIG devices.
        """
        if not self._device.mig.is_mig_device:
            raise nvml.NotSupportedError(
                "GPU instance ID is only valid for processes running on MIG devices."
            )
        return self._process_info.gpu_instance_id

    @property
    def compute_instance_id(self) -> int:
        """
        The Compute instance ID for MIG devices.

        Only valid for processes running on MIG devices.
        """
        if not self._device.mig.is_mig_device:
            raise nvml.NotSupportedError(
                "Compute instance ID is only valid for processes running on MIG devices."
            )
        return self._process_info.compute_instance_id
