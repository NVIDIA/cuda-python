# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Conversion helpers for the driver's ``CUmemLocation`` struct.
#
# Header-only so both the managed-memory ops and the batched copy path can
# cimport it without either module depending on the other. ``CUmemLocation``
# is only populated on a CUDA 13 build; the CUDA 12 stub exists so callers
# compiled there still resolve the symbol.
#
# Both helpers use field assignment rather than Cython struct literals
# (``CUmemLocation(type=..., id=...)``) so this source keeps compiling if a
# future generated ``cydriver.pxd`` adds a sibling member to the struct's
# anonymous union (e.g. CUDA 13.4's ``localized`` arm): Cython's struct-literal
# coercion warns "Not all members given for struct" whenever a call site does
# not name every declared member, and cuda_core promotes that warning to a
# build error.

from cuda.bindings cimport cydriver


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline cydriver.CUmemLocation cumemlocation_from_type(
        cydriver.CUmemLocationType loc_type, int loc_id
    ):
        """Build a ``CUmemLocation`` from an already-known ``CUmemLocationType``.

        For call sites that already carry a ``CUmemLocationType`` value
        (e.g. from a pool-configuration parameter), rather than the
        ``kind`` string used by :func:`to_cumemlocation`.
        """
        cdef cydriver.CUmemLocation cu_loc
        cu_loc.type = loc_type
        cu_loc.id = loc_id
        return cu_loc

    cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id):
        if kind == "device":
            return cumemlocation_from_type(
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, loc_id)
        elif kind == "host":
            return cumemlocation_from_type(
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST, 0)
        elif kind == "host_numa":
            return cumemlocation_from_type(
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, loc_id)
        elif kind == "host_numa_current":
            return cumemlocation_from_type(
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT, 0)
        else:
            raise ValueError(f"unknown location kind: {kind!r}")
ELSE:
    cdef inline cydriver.CUmemLocation cumemlocation_from_type(
        cydriver.CUmemLocationType loc_type, int loc_id
    ):
        cdef cydriver.CUmemLocation cu_loc
        cu_loc.type = loc_type
        cu_loc.id = loc_id
        return cu_loc

    cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id):
        raise NotImplementedError(
            "CUmemLocation requires cuda.core built against CUDA 13 headers"
        )
