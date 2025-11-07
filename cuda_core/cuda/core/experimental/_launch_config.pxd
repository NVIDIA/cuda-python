# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Public API declarations for _launch_config module

cdef bint _inited
cdef bint _use_ex

cdef void _lazy_init() except *

cdef class LaunchConfig:
    """Customizable launch options."""
    cdef public tuple grid
    cdef public tuple cluster
    cdef public tuple block
    cdef public int shmem_size
    cdef public bint cooperative_launch

cpdef object _to_native_launch_config(LaunchConfig config)
