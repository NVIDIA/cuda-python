# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._resource_handles cimport GraphicsResourceHandle
from cuda.core._memory._buffer cimport Buffer


cdef class GraphicsResource(Buffer):

    cdef:
        GraphicsResourceHandle _handle
        bint _mapped
        object _map_stream

    cpdef close(self, stream=*)
