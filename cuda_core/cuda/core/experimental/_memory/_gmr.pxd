# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport MemoryResource
from cuda.core.experimental._memory._dmr cimport DeviceMemoryResource
from cuda.core.experimental._memory._ipc cimport IPCData


cdef class cyGraphMemoryResource(MemoryResource):
    cdef:
        int _dev_id
