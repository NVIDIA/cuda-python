# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memory._memory_pool cimport _MemPool


cdef class PinnedMemoryResource(_MemPool):
    cdef int _numa_id
    # False only for an imported pool whose node is unknown (see numa_id).
    cdef bint _numa_id_known
