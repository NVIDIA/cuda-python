# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver
from cuda.core._event cimport Event


cdef class LaunchConfig:
    """Customizable launch options."""
    cdef:
        public tuple grid
        public tuple cluster
        public tuple block
        public int shmem_size
        public bint is_cooperative
        public bint programmatic_stream_serialization
        public str cluster_scheduling_policy_preference
        public int priority
        public Event programmatic_event
        public bint programmatic_event_trigger_at_block_start

        vector[cydriver.CUlaunchAttribute] _attrs
        object __weakref__

    cdef Event _accept_programmatic_event(self)
    cdef cydriver.CUlaunchConfig _to_native_launch_config(self)


cpdef object _to_native_launch_config(LaunchConfig config)
