# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver


cdef class Event:

    cdef:
        cydriver.CUevent _handle
        bint _timing_disabled
        bint _busy_waited
        int _device_id
        object _ctx_handle

    cpdef close(self)
