.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Examples
========

This page links to the ``cuda.bindings`` examples shipped in the
`cuda-python repository <https://github.com/NVIDIA/cuda-python/tree/|cuda_bindings_github_ref|/cuda_bindings/examples>`_.
Use it as a quick index when you want a runnable sample for a specific API area
or CUDA feature.

Introduction
------------

- `clock_nvrtc.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/clock_nvrtc.py>`_
  uses NVRTC-compiled CUDA code and the device clock to time a reduction
  kernel.
- `simple_cubemap_texture.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/simple_cubemap_texture.py>`_
  demonstrates cubemap texture sampling and transformation.
- `simple_p2p.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/simple_p2p.py>`_
  shows peer-to-peer memory access and transfers between multiple GPUs.
- `simple_zero_copy.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/simple_zero_copy.py>`_
  uses zero-copy mapped host memory for vector addition.
- `system_wide_atomics.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/system_wide_atomics.py>`_
  demonstrates system-wide atomic operations on managed memory.
- `vector_add_drv.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/vector_add_drv.py>`_
  uses the CUDA Driver API and unified virtual addressing for vector addition.
- `vector_add_mmap.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/0_Introduction/vector_add_mmap.py>`_
  uses virtual memory management APIs such as ``cuMemCreate`` and
  ``cuMemMap`` for vector addition.

Concepts and techniques
-----------------------

- `stream_ordered_allocation.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/2_Concepts_and_Techniques/stream_ordered_allocation.py>`_
  demonstrates ``cudaMallocAsync`` and ``cudaFreeAsync`` together with
  memory-pool release thresholds.

CUDA features
-------------

- `global_to_shmem_async_copy.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/3_CUDA_Features/global_to_shmem_async_copy.py>`_
  compares asynchronous global-to-shared-memory copy strategies in matrix
  multiplication kernels.
- `simple_cuda_graphs.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/3_CUDA_Features/simple_cuda_graphs.py>`_
  shows both manual CUDA graph construction and stream-capture-based replay.

Libraries and tools
-------------------

- `conjugate_gradient_multi_block_cg.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/4_CUDA_Libraries/conjugate_gradient_multi_block_cg.py>`_
  implements a conjugate-gradient solver with cooperative groups and
  multi-block synchronization.
- `nvidia_smi.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/4_CUDA_Libraries/nvidia_smi.py>`_
  uses NVML to implement a Python subset of ``nvidia-smi``.

Advanced and interoperability
-----------------------------

- `iso_fd_modelling.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/extra/iso_fd_modelling.py>`_
  runs isotropic finite-difference wave propagation across multiple GPUs with
  peer-to-peer halo exchange.
- `jit_program.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/extra/jit_program.py>`_
  JIT-compiles a SAXPY kernel with NVRTC and launches it through the Driver
  API.
- `numba_emm_plugin.py <https://github.com/NVIDIA/cuda-python/blob/|cuda_bindings_github_ref|/cuda_bindings/examples/extra/numba_emm_plugin.py>`_
  shows how to back Numba's EMM interface with the NVIDIA CUDA Python Driver
  API.
