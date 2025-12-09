# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t
from libcpp.memory cimport shared_ptr

from cuda.bindings cimport cydriver

# Declare the C++ namespace and types
cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    # Handle type - shared_ptr to const CUcontext
    ctypedef shared_ptr[const cydriver.CUcontext] ContextHandle

    # Function to create a non-owning context handle (references existing context)
    # This is nogil-safe (pure C++, no Python dependencies)
    ContextHandle create_context_handle_ref(cydriver.CUcontext ctx) nogil

    # Context acquisition functions (pure C++, nogil-safe with thread-local caching)
    ContextHandle get_primary_context(int dev_id) nogil
    ContextHandle get_current_context() nogil

    # ========================================================================
    # Helper functions to extract raw resources from handles
    # Defined in C++ to support overloading when additional handle types are added
    # ========================================================================

    # native() - extract the raw CUDA handle (nogil-safe)
    cydriver.CUcontext native(ContextHandle h) nogil

    # intptr() - extract handle as uintptr_t (nogil-safe)
    uintptr_t intptr(ContextHandle h) nogil

    # py() - convert handle to Python driver wrapper object (requires GIL)
    object py(ContextHandle h)
