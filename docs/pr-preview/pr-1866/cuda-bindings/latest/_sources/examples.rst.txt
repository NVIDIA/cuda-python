.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Examples
========

This page links to the ``cuda.bindings`` examples shipped in the
`cuda-python repository <https://github.com/NVIDIA/cuda-python/tree/main/cuda_bindings/examples>`_.
Use it as a quick index when you want a runnable sample for a specific API area
or CUDA feature.

Introduction
------------

- `clock_nvrtc_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/clock_nvrtc_test.py>`_
  uses NVRTC-compiled CUDA code and the device clock to time a reduction
  kernel.
- `simpleCubemapTexture_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/simpleCubemapTexture_test.py>`_
  demonstrates cubemap texture sampling and transformation.
- `simpleP2P_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/simpleP2P_test.py>`_
  shows peer-to-peer memory access and transfers between multiple GPUs.
- `simpleZeroCopy_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/simpleZeroCopy_test.py>`_
  uses zero-copy mapped host memory for vector addition.
- `systemWideAtomics_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/systemWideAtomics_test.py>`_
  demonstrates system-wide atomic operations on managed memory.
- `vectorAddDrv_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/vectorAddDrv_test.py>`_
  uses the CUDA Driver API and unified virtual addressing for vector addition.
- `vectorAddMMAP_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/0_Introduction/vectorAddMMAP_test.py>`_
  uses virtual memory management APIs such as ``cuMemCreate`` and
  ``cuMemMap`` for vector addition.

Concepts and techniques
-----------------------

- `streamOrderedAllocation_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/2_Concepts_and_Techniques/streamOrderedAllocation_test.py>`_
  demonstrates ``cudaMallocAsync`` and ``cudaFreeAsync`` together with
  memory-pool release thresholds.

CUDA features
-------------

- `globalToShmemAsyncCopy_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/3_CUDA_Features/globalToShmemAsyncCopy_test.py>`_
  compares asynchronous global-to-shared-memory copy strategies in matrix
  multiplication kernels.
- `simpleCudaGraphs_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/3_CUDA_Features/simpleCudaGraphs_test.py>`_
  shows both manual CUDA graph construction and stream-capture-based replay.

Libraries and tools
-------------------

- `conjugateGradientMultiBlockCG_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/4_CUDA_Libraries/conjugateGradientMultiBlockCG_test.py>`_
  implements a conjugate-gradient solver with cooperative groups and
  multi-block synchronization.
- `nvidia_smi.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/4_CUDA_Libraries/nvidia_smi.py>`_
  uses NVML to implement a Python subset of ``nvidia-smi``.

Advanced and interoperability
-----------------------------

- `isoFDModelling_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/extra/isoFDModelling_test.py>`_
  runs isotropic finite-difference wave propagation across multiple GPUs with
  peer-to-peer halo exchange.
- `jit_program_test.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/extra/jit_program_test.py>`_
  JIT-compiles a SAXPY kernel with NVRTC and launches it through the Driver
  API.
- `numba_emm_plugin.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples/extra/numba_emm_plugin.py>`_
  shows how to back Numba's EMM interface with the NVIDIA CUDA Python Driver
  API.
