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


# ============================================================================
# Helper functions to extract raw resources from handles
# ============================================================================

cdef inline cydriver.CUcontext native(ContextHandle h) nogil:
    """Extract the native C type (cydriver.CUcontext) from the handle.

    This is for use with cydriver API calls that expect the raw C type.
    """
    return h.get()[0]


# Python conversion function (implemented in .pyx due to Python module dependency)
cdef object py(ContextHandle h)


cdef inline uintptr_t intptr(ContextHandle h) nogil:
    """Extract the handle as a uintptr_t integer address.

    This is for use with internal APIs that expect integer addresses.
    """
    return <uintptr_t>(h.get()[0])
