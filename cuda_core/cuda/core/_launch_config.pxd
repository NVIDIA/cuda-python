# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver


cdef class LaunchConfig:
    """Customizable launch options."""
    cdef:
        readonly tuple grid
        readonly tuple cluster
        readonly tuple block
        readonly int shmem_size
        readonly bint is_cooperative

        vector[cydriver.CUlaunchAttribute] _attrs
        cydriver.CUlaunchConfig _cached_drv_cfg
        bint _cache_valid
        object __weakref__

    cdef cydriver.CUlaunchConfig _to_native_launch_config(self)


cpdef object _to_native_launch_config(LaunchConfig config)
