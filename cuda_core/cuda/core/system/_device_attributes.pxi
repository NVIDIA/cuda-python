# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class DeviceAttributes:
    """
    Various device attributes.
    """
    def __init__(self, attributes: nvml.DeviceAttributes):
        self._attributes = attributes

    @property
    def multiprocessor_count(self) -> int:
        """
        The streaming multiprocessor count
        """
        return self._attributes.multiprocessor_count

    @property
    def shared_copy_engine_count(self) -> int:
        """
        The shared copy engine count
        """
        return self._attributes.shared_copy_engine_count

    @property
    def shared_decoder_count(self) -> int:
        """
        The shared decoder engine count
        """
        return self._attributes.shared_decoder_count

    @property
    def shared_encoder_count(self) -> int:
        """
        The shared encoder engine count
        """
        return self._attributes.shared_encoder_count

    @property
    def shared_jpeg_count(self) -> int:
        """
        The shared JPEG engine count
        """
        return self._attributes.shared_jpeg_count

    @property
    def shared_ofa_count(self) -> int:
        """
        The shared optical flow accelerator (OFA) engine count
        """
        return self._attributes.shared_ofa_count

    @property
    def gpu_instance_slice_count(self) -> int:
        """
        The GPU instance slice count
        """
        return self._attributes.gpu_instance_slice_count

    @property
    def compute_instance_slice_count(self) -> int:
        """
        The compute instance slice count
        """
        return self._attributes.compute_instance_slice_count

    @property
    def memory_size_mb(self) -> int:
        """
        Device memory size in MiB
        """
        return self._attributes.memory_size_mb
