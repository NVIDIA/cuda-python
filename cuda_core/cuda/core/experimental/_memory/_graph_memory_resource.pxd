# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental._memory._buffer cimport MemoryResource


cdef class cyGraphMemoryResource(MemoryResource):
    cdef:
        int _dev_id
