# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cpdef int _check_driver_error(error) except?-1
cpdef int _check_runtime_error(error) except?-1
cpdef int _check_nvrtc_error(error) except?-1
cpdef check_or_create_options(type cls, options, str options_description=*, bint keep_none=*)
