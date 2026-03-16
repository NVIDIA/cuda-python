# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memory._memory_pool cimport _MemPool


cdef class ManagedMemoryResource(_MemPool):
    cdef:
        str _pref_loc_type
        int _pref_loc_id
