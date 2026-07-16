.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Examples
========

This page links to the ``cuda.bindings`` examples shipped in the
:cuda-bindings-examples:`cuda-python repository </>`.
Use it as a quick index when you want a runnable sample for a specific API area
or CUDA feature.

Introduction
------------

- :cuda-bindings-example:`clock_nvrtc.py <0_Introduction/clock_nvrtc.py>`
  uses NVRTC-compiled CUDA code and the device clock to time a reduction
  kernel.
- :cuda-bindings-example:`simple_cubemap_texture.py <0_Introduction/simple_cubemap_texture.py>`
  demonstrates cubemap texture sampling and transformation.
- :cuda-bindings-example:`system_wide_atomics.py <0_Introduction/system_wide_atomics.py>`
  demonstrates system-wide atomic operations on managed memory.
- :cuda-bindings-example:`vector_add_drv.py <0_Introduction/vector_add_drv.py>`
  uses the CUDA Driver API and unified virtual addressing for vector addition.
- :cuda-bindings-example:`vector_add_mmap.py <0_Introduction/vector_add_mmap.py>`
  uses virtual memory management APIs such as ``cuMemCreate`` and
  ``cuMemMap`` for vector addition.

Peer-to-peer and zero-copy patterns (``simple_p2p.py`` and
``simple_zero_copy.py``) are covered by the higher-level
:samples:`samples/simpleP2P/ <simpleP2P/>` and
:samples:`samples/simpleZeroCopy/ <simpleZeroCopy/>` sample directories,
which use ``cuda.core``'s modern peer-access and pinned-memory APIs on top
of ``cuda.bindings``.

Concepts and techniques
-----------------------

- :cuda-bindings-example:`stream_ordered_allocation.py <2_Concepts_and_Techniques/stream_ordered_allocation.py>`
  demonstrates ``cudaMallocAsync`` and ``cudaFreeAsync`` together with
  memory-pool release thresholds.

CUDA features
-------------

- :cuda-bindings-example:`global_to_shmem_async_copy.py <3_CUDA_Features/global_to_shmem_async_copy.py>`
  compares asynchronous global-to-shared-memory copy strategies in matrix
  multiplication kernels.
- :cuda-bindings-example:`simple_cuda_graphs.py <3_CUDA_Features/simple_cuda_graphs.py>`
  shows both manual CUDA graph construction and stream-capture-based replay.

Libraries and tools
-------------------

- :cuda-bindings-example:`conjugate_gradient_multi_block_cg.py <4_CUDA_Libraries/conjugate_gradient_multi_block_cg.py>`
  implements a conjugate-gradient solver with cooperative groups and
  multi-block synchronization.
- :cuda-bindings-example:`nvidia_smi.py <4_CUDA_Libraries/nvidia_smi.py>`
  uses NVML to implement a Python subset of ``nvidia-smi``.

Advanced and interoperability
-----------------------------

- :cuda-bindings-example:`iso_fd_modelling.py <extra/iso_fd_modelling.py>`
  runs isotropic finite-difference wave propagation across multiple GPUs with
  peer-to-peer halo exchange.
- :cuda-bindings-example:`jit_program.py <extra/jit_program.py>`
  JIT-compiles a SAXPY kernel with NVRTC and launches it through the Driver
  API.
