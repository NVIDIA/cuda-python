# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings.cyruntime cimport *
cimport cuda.bindings._bindings.cydriver as cydriver

# These are hard-coded helper function from the initial reimplementation of CUDA Runtime
# and will be removed as part of https://github.com/NVIDIA/cuda-python/issues/488
cdef cudaError_t getDriverEglFrame(cydriver.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
