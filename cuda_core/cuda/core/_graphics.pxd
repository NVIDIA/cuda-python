# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._resource_handles cimport GraphicsResourceHandle


cdef class GraphicsResource:

    cdef:
        GraphicsResourceHandle _handle
        bint _mapped

    cpdef close(self)
