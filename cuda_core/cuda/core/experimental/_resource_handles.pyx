# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module exists to compile _cpp/resource_handles.cpp into a shared library.
# The helper functions (native, intptr, py) are implemented as inline C++ functions
# in _cpp/resource_handles.hpp and declared as extern in _resource_handles.pxd.

from cpython.pycapsule cimport PyCapsule_New

from cuda.core.experimental._resource_handles_cxx_api cimport (
    ResourceHandlesCxxApiV1,
    get_resource_handles_cxx_api_v1,
)


cdef const char* _CXX_API_NAME = b"cuda.core.experimental._resource_handles._CXX_API"

# Export the C++ handles dispatch table as a PyCapsule.
# Consumers use PyCapsule_Import(_CXX_API_NAME, 0) to retrieve it.
cdef const ResourceHandlesCxxApiV1* _handles_table = get_resource_handles_cxx_api_v1()
if _handles_table == NULL:
    raise RuntimeError("Failed to initialize resource handles C++ API table")

_CXX_API = <object>PyCapsule_New(<void*>_handles_table, _CXX_API_NAME, NULL)
if _CXX_API is None:
    raise RuntimeError("Failed to create _CXX_API capsule")
