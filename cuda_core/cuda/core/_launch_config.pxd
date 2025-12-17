# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver


cdef class LaunchConfig:
    """Customizable launch options."""
    cdef:
        public tuple grid
        public tuple cluster
        public tuple block
        public int shmem_size
        public bint cooperative_launch

        vector[cydriver.CUlaunchAttribute] _attrs

    cdef cydriver.CUlaunchConfig _to_native_launch_config(self)


cpdef object _to_native_launch_config(LaunchConfig config)
