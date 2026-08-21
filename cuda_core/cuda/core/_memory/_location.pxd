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
#
# Construction uses field assignment rather than Cython struct literals so
# the same source compiles against both the CUDA 13.3 two-member declaration
# and the CUDA 13.4 declaration that adds the ``localized`` union arm.
# ``CUDA_CORE_HAS_LOCALIZED_LOCATION`` selects a helper signature with an
# optional ``localized`` argument (13.4+) or without it (13.3). Passing
# ``localized=...`` is therefore a Cython compile-time error on 13.3.

from cuda.bindings cimport cydriver


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline void _fill_id_location(
        cydriver.CUmemLocation* cu_loc, str kind, int loc_id
    ) except *:
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

    IF CUDA_CORE_HAS_LOCALIZED_LOCATION:
        cdef inline cydriver.CUmemLocation to_cumemlocation(
            str kind, int loc_id=0, tuple localized=None
        ):
            cdef cydriver.CUmemLocation cu_loc
            if kind == "device_locality_domain":
                if localized is None:
                    raise ValueError(
                        "kind='device_locality_domain' requires "
                        "localized=(device_id, locality_domain_id)"
                    )
                cu_loc.type = (
                    cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
                )
                cu_loc.localized.deviceId = localized[0]
                cu_loc.localized.localityDomainId = localized[1]
                return cu_loc
            _fill_id_location(&cu_loc, kind, loc_id)
            return cu_loc
    ELSE:
        cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id=0):
            cdef cydriver.CUmemLocation cu_loc
            _fill_id_location(&cu_loc, kind, loc_id)
            return cu_loc
ELSE:
    cdef inline cydriver.CUmemLocation to_cumemlocation(str kind, int loc_id=0):
        raise NotImplementedError(
            "CUmemLocation requires cuda.core built against CUDA 13 headers"
        )
