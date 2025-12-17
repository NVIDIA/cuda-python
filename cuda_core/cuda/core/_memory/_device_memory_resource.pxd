# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memory._memory_pool cimport _MemPool
from cuda.core._memory._ipc cimport IPCDataForMR


cdef class DeviceMemoryResource(_MemPool):
    pass


cpdef DMR_mempool_get_access(DeviceMemoryResource, int)
