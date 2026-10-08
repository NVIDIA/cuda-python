.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Examples
========

The canonical, runnable examples for ``cuda.bindings`` live under
:samples:`samples/cuda_bindings/ </>` in the cuda-python repository. Each
sample is a self-contained directory with
its own ``README.md``, ``requirements.txt``, and PEP 723 dependency block,
and every sample is exercised as part of the ``cuda.bindings`` test suite.

The cuda-bindings-flavored samples preserve the same category structure the
``cuda-samples`` repository uses, so they are grouped under
``samples/cuda_bindings/0_Introduction/``,
``samples/cuda_bindings/2_Concepts_and_Techniques/``,
``samples/cuda_bindings/3_CUDA_Features/``,
``samples/cuda_bindings/4_CUDA_Libraries/``, and
``samples/cuda_bindings/extra/``.

Introduction
------------

- :sample:`0_Introduction/clockNvrtc <0_Introduction/clockNvrtc/>`
  uses NVRTC-compiled CUDA code and the device clock intrinsic to time a
  reduction kernel; also demonstrates dynamic shared memory.
- :sample:`0_Introduction/simpleCubemapTexture <0_Introduction/simpleCubemapTexture/>`
  demonstrates cubemap texture sampling, ``cudaMemcpy3D``, and bindless
  texture objects.
- :sample:`0_Introduction/systemWideAtomics <0_Introduction/systemWideAtomics/>`
  exercises every ``atomic*_system`` intrinsic on managed memory.
- :sample:`0_Introduction/vectorAddDriverApi <0_Introduction/vectorAddDriverApi/>`
  is the "hello world" of the raw driver API in Python.
- :sample:`0_Introduction/vectorAddMmap <0_Introduction/vectorAddMmap/>`
  uses the Virtual Memory Management API (``cuMemCreate``, ``cuMemMap``,
  ``cuMemSetAccess``) to stripe an allocation across peer-capable devices.

Concepts and techniques
-----------------------

- :sample:`2_Concepts_and_Techniques/streamOrderedAllocation <2_Concepts_and_Techniques/streamOrderedAllocation/>`
  demonstrates ``cudaMallocAsync`` / ``cudaFreeAsync`` together with
  memory-pool release thresholds.

CUDA features
-------------

- :sample:`3_CUDA_Features/globalToShmemAsyncCopy <3_CUDA_Features/globalToShmemAsyncCopy/>`
  compares asynchronous global-to-shared-memory copy strategies
  (``cuda::memcpy_async`` + ``cuda::pipeline`` / ``cuda::barrier``) in
  matrix-multiplication kernels.
- :sample:`3_CUDA_Features/cudaGraphsManualNodes <3_CUDA_Features/cudaGraphsManualNodes/>`
  shows both manual CUDA-graph construction (``cudaGraphAdd*Node``) and
  stream-capture-based replay.

Libraries and tools
-------------------

- :sample:`4_CUDA_Libraries/conjugateGradientMultiBlockCG <4_CUDA_Libraries/conjugateGradientMultiBlockCG/>`
  implements a conjugate-gradient solver with cooperative groups and
  multi-block grid synchronization.
- :sample:`4_CUDA_Libraries/nvidiaSmi <4_CUDA_Libraries/nvidiaSmi/>`
  uses NVML to implement a Python subset of ``nvidia-smi``.

Advanced and interoperability
-----------------------------

- :sample:`extra/isoFdModelling <extra/isoFdModelling/>`
  runs isotropic finite-difference wave propagation across multiple GPUs
  with peer-to-peer halo exchange.
- :sample:`extra/jitProgram <extra/jitProgram/>`
  JIT-compiles a SAXPY kernel with NVRTC and launches it through the
  Driver API -- the low-level companion to
  :cuda-core-sample:`cuda.core's jitLtoLinking sample <jitLtoLinking/>`.
