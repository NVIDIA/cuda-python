# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Neutral leaf module: declares the CopyOptions-to-CUmemcpyAttributes converter
# and the 13.2 availability gate so both _buffer and _copy_ops can cimport them
# without either depending on the other.

from cuda.bindings cimport cydriver
from cuda.core._utils.version cimport cy_driver_version  # no-cython-lint


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline bint _with_attributes_available():
        # cuMemcpyWithAttributesAsync is a 13.2 driver API; cuda-bindings 13.4+
        # (the floor) always exports it, so only the driver can lack it.
        return cy_driver_version() >= (13, 2, 0)
ELSE:
    cdef inline bint _with_attributes_available():
        return False

cdef cydriver.CUmemcpyAttributes _to_cu_memcpy_attributes(object attr)
