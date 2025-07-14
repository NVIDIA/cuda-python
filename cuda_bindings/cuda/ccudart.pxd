# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings.cyruntime cimport *

cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.ccudart module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cyruntime module instead." )
    #else
    #warning The cuda.ccudart module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.cyruntime module instead.
    #endif
    """
