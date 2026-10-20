:attr:`system.Device.uuid_without_prefix` preserves UUIDs that NVML already
reports without a prefix, such as on Orin. This also allows
:meth:`system.Device.to_cuda_device` to match those devices correctly. (#2947)
