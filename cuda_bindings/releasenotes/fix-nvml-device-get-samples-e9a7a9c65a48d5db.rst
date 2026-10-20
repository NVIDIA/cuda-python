``nvml.device_get_samples``: On a zero-sample result, the old implementation
returned a bare ``Sample`` instance instead of the documented
``(sample_val_type, samples)`` tuple. It now always returns the correct 2-tuple. (#2862)
