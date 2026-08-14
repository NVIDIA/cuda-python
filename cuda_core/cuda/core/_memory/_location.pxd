# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Conversion from the internal ``_LocSpec`` record produced by
# ``_managed_location._coerce_location`` to the driver's ``CUmemLocation``.
#
# Header-only so both the managed-memory ops and the batched copy path can
# cimport it without either module depending on the other. ``CUmemLocation``
# is only populated on a CUDA 13 build; the CUDA 12 stub exists so callers
# compiled there still resolve the symbol.

from cuda.bindings cimport cydriver


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id):
        cdef cydriver.CUmemLocation cu_loc
        if kind == "device":
            cu_loc.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            cu_loc.id = loc_id
        elif kind == "host":
            cu_loc.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            cu_loc.id = 0
        elif kind == "host_numa":
            cu_loc.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            cu_loc.id = loc_id
        elif kind == "host_numa_current":
            cu_loc.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT
            cu_loc.id = 0
        else:
            raise ValueError(f"unknown location kind: {kind!r}")
        return cu_loc
ELSE:
    cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id):
        raise NotImplementedError(
            "CUmemLocation requires cuda.core built against CUDA 13 headers"
        )
