# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class InforomInfo:
    cdef Device _device

    def __init__(self, device: Device):
        self._device = device

    def get_version(self, inforom: InforomObject) -> str:
        """
        Retrieves the InfoROM version for a given InfoROM object.

        For all products with an InfoROM.

        Fermi™ and higher parts have non-volatile on-board memory for persisting
        device info, such as aggregate ECC counts.

        Parameters
        ----------
        inforom: :class:`InforomObject`
            The InfoROM object to query.

        Returns
        -------
        str
            The InfoROM version.
        """
        return nvml.device_get_inforom_version(self._device._handle, inforom)

    @property
    def image_version(self) -> str:
        """
        Retrieves the global InfoROM image version.

        For all products with an InfoROM.

        Image version just like VBIOS version uniquely describes the exact
        version of the InfoROM flashed on the board in contrast to InfoROM
        object version which is only an indicator of supported features.

        Returns
        -------
        str
            The InfoROM image version.
        """
        return nvml.device_get_inforom_image_version(self._device._handle)

    @property
    def configuration_checksum(self) -> int:
        """
        Retrieves the checksum of the configuration stored in the device's InfoROM.

        For all products with an InfoROM.

        Can be used to make sure that two GPUs have the exact same
        configuration.  Current checksum takes into account configuration stored
        in PWR and ECC InfoROM objects.  Checksum can change between driver
        releases or when user changes configuration (e.g. disable/enable ECC)

        Returns
        -------
        int
            The InfoROM checksum.
        """
        return nvml.device_get_inforom_configuration_checksum(self._device._handle)

    def validate(self) -> None:
        """
        Reads the InfoROM from the flash and verifies the checksums.

        For all products with an InfoROM.

        Raises
        ------
        :class:`cuda.core.system.CorruptedInforomError`
            If the device's InfoROM is corrupted.
        """
        nvml.device_validate_inforom(self._device._handle)

    @property
    def bbx_flush_time(self) -> int:
        """
        Retrieves the timestamp and duration of the last flush of the BBX
        (bloackbox) InfoROM object during the current run.

        For all products with an InfoROM.

        Returns
        -------
        tuple[int, int]
            - timestamp: The start timestamp of the last BBX flush
            - duration_us: The duration (in μs) of the last BBX flush
        """
        return nvml.device_get_last_bbx_flush_time(self._device._handle)
