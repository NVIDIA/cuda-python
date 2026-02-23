# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver


cdef class TensorMapDescriptor:
    cdef cydriver.CUtensorMap _tensor_map
    cdef object _source_ref

    cdef void* _get_data_ptr(self)
