# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Neutral leaf module: declares the CopyOptions-to-CUmemcpyAttributes converter
# and the 13.2 availability gate so both _buffer and _copy_ops can cimport them
# without either depending on the other.

from cuda.bindings cimport cydriver
from cuda.core._utils.version cimport cy_binding_version, cy_driver_version  # no-cython-lint


IF CUDA_CORE_BUILD_MAJOR >= 13:
    from cuda.core._resource_handles cimport has_memcpy_with_attributes_async

    cdef inline bint _with_attributes_available():
        # has_memcpy_with_attributes_async() says whether the installed
        # cuda-bindings actually exports cuMemcpyWithAttributesAsync (13.2+);
        # the version checks alone are not sufficient, since cuda.core's build
        # can be paired with a cuda-bindings install older than what it built
        # against (see https://github.com/NVIDIA/cuda-python/issues/2063).
        return (
            has_memcpy_with_attributes_async()
            and cy_driver_version() >= (13, 2, 0)
            and cy_binding_version() >= (13, 2, 0)
        )
ELSE:
    cdef inline bint _with_attributes_available():
        return False

cdef cydriver.CUmemcpyAttributes _to_cu_memcpy_attributes(object attr)
