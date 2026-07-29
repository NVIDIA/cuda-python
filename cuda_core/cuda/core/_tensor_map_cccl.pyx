# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional CCCL <cuda/tma> helper for TensorMapDescriptor.

This extension is imported only after ``load_nvidia_dynamic_lib("cudart")`` so
CCCL's references to ``cudaGetErrorString`` resolve from the globally loaded
shared cudart. It must not be imported at ``cuda.core`` package import time.

``get_make_tma_descriptor_tiled`` is exported via ``__pyx_capi__`` for
soft-linking from ``_tensor_map.pyx`` (returns real fn or NULL).
"""

from cuda.core._tensor_map_cccl cimport make_tma_descriptor_tiled_t

cdef extern from "_cpp/tensor_map_cccl.h":
    make_tma_descriptor_tiled_t cuda_core_cccl_make_tma_descriptor_tiled


cdef make_tma_descriptor_tiled_t get_make_tma_descriptor_tiled() noexcept nogil:
    return cuda_core_cccl_make_tma_descriptor_tiled
