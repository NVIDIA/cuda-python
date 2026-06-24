# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport ContextHandle, GreenCtxHandle


cdef class SMResource:
    cdef:
        cydriver.CUdevResource _resource
        unsigned int _sm_count
        unsigned int _min_partition_size
        unsigned int _coscheduled_alignment
        unsigned int _flags
        bint _is_usable
        object __weakref__

    @staticmethod
    cdef SMResource _from_dev_resource(cydriver.CUdevResource res, int device_id)

    @staticmethod
    cdef SMResource _from_split_resource(cydriver.CUdevResource res, SMResource parent, bint is_usable)


cdef class WorkqueueResource:
    cdef:
        cydriver.CUdevResource _wq_config_resource
        cydriver.CUdevResource _wq_resource
        object __weakref__

    @staticmethod
    cdef WorkqueueResource _from_dev_resources(
        cydriver.CUdevResource wq_config,
        cydriver.CUdevResource wq,
    )


cdef class DeviceResources:
    cdef:
        int _device_id
        ContextHandle _h_context  # NULL for device-level queries
        object __weakref__

    @staticmethod
    cdef DeviceResources _init(int device_id)

    @staticmethod
    cdef DeviceResources _init_from_ctx(ContextHandle h_context, int device_id)

    cdef inline int _query_sm(self, cydriver.CUdevResource* res) except?-1 nogil
