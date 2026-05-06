# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 1.5.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycudla cimport *




###############################################################################
# Types
###############################################################################

ctypedef cudlaDevHandle DevHandle
ctypedef cudlaModule Module


###############################################################################
# Enum
###############################################################################

ctypedef cudlaStatus _Status
ctypedef cudlaMode _Mode
ctypedef cudlaModuleAttributeType _ModuleAttributeType
ctypedef cudlaFenceType _FenceType
ctypedef cudlaModuleLoadFlags _ModuleLoadFlags
ctypedef cudlaSubmissionFlags _SubmissionFlags
ctypedef cudlaAccessPermissionFlags _AccessPermissionFlags
ctypedef cudlaDevAttributeType _DevAttributeType


###############################################################################
# Functions
###############################################################################

cpdef uint64_t get_version() except? -1
cpdef uint64_t device_get_count() except? -1
cpdef intptr_t create_device(uint64_t device, uint32_t flags) except *
cpdef intptr_t mem_register(intptr_t dev_handle, intptr_t ptr, size_t size, uint32_t flags) except *
cpdef intptr_t module_load_from_memory(intptr_t dev_handle, p_module, size_t module_size, uint32_t flags) except *
cpdef module_unload(intptr_t h_module, uint32_t flags)
cpdef submit_task(intptr_t dev_handle, intptr_t ptr_to_tasks, uint32_t num_tasks, intptr_t stream, uint32_t flags)
cpdef object device_get_attribute(intptr_t dev_handle, int attrib) except *
cpdef mem_unregister(intptr_t dev_handle, intptr_t dev_ptr)
cpdef int get_last_error(intptr_t dev_handle) except? 0
cpdef destroy_device(intptr_t dev_handle)
cpdef set_task_timeout_in_ms(intptr_t dev_handle, uint32_t timeout)

cpdef module_get_attributes(intptr_t h_module, int attr_type) except *
