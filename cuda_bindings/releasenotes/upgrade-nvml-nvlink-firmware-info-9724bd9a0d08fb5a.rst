``nvml.NvlinkFirmwareInfo``: The ``firmware_version`` member now returns only
valid entries.  The ``num_valid_entries`` member is removed, as its design made
it possible to access invalid memory. (#2953)
