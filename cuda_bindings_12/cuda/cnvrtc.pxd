# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cynvrtc cimport *

cdef extern from *:
    """
    #ifdef _MSC_VER
    #pragma message ( "The cuda.cnvrtc module is deprecated and will be removed in a future release, " \
                      "please switch to use the cuda.bindings.cynvrtc module instead." )
    #else
    #warning The cuda.cnvrtc module is deprecated and will be removed in a future release, \
             please switch to use the cuda.bindings.cynvrtc module instead.
    #endif
    """
