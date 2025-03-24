# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda.bindings.cyruntime cimport *
cimport cuda.bindings._bindings.cydriver as cydriver

# These are hard-coded helper function from the initial reimplementation of CUDA Runtime
# and will be removed as part of https://github.com/NVIDIA/cuda-python/issues/488
cdef cudaError_t getDriverEglFrame(cydriver.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
