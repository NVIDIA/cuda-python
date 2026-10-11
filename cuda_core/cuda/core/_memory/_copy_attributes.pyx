# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._memory._location cimport to_cumemlocation

from cuda.core._memory._managed_location import _coerce_location


cdef cydriver.CUmemcpyAttributes _to_cu_memcpy_attributes(object attr):
    """Convert a CopyOptions to a cydriver.CUmemcpyAttributes struct."""
    cdef cydriver.CUmemcpyAttributes cu_attr
    memset(&cu_attr, 0, sizeof(cydriver.CUmemcpyAttributes))
    cu_attr.srcAccessOrder = <cydriver.CUmemcpySrcAccessOrder>(<int>attr._to_driver_enum())
    cu_attr.flags = <unsigned int>(<int>attr._to_driver_flags())

    cdef object src_loc = _coerce_location(attr.src_location_hint, allow_none=True)
    cdef object dst_loc = _coerce_location(attr.dst_location_hint, allow_none=True)

    if src_loc is not None:
        cu_attr.srcLocHint = to_cumemlocation(src_loc.kind, src_loc.id)
    if dst_loc is not None:
        cu_attr.dstLocHint = to_cumemlocation(dst_loc.kind, dst_loc.id)

    return cu_attr
