# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Neutral leaf module: declares the CopyOptions-to-CUmemcpyAttributes converter
# and the 13.2 availability gate so both _buffer and _copy_ops can cimport them
# without either depending on the other.

from cuda.bindings cimport cydriver


cdef bint _with_attributes_available()
cdef cydriver.CUmemcpyAttributes _to_cu_memcpy_attributes(object attr)
