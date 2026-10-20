:class:`CUDAError` and :class:`NVRTCError` are now importable from the
top-level ``cuda.core`` namespace, alongside :class:`CUDAWarning`. Previously
they were only available from a private module. (#2759)
