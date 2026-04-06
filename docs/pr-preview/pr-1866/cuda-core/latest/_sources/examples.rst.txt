.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Examples
========

This page links to the ``cuda.core`` examples shipped in the
`cuda-python repository <https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/examples>`_.
Use it as a quick index when you want a runnable starting point for a specific
workflow.

Compilation and kernel launch
-----------------------------

- `vector_add.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/vector_add.py>`_
  compiles and launches a simple vector-add kernel with CuPy arrays.
- `saxpy.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/saxpy.py>`_
  JIT-compiles a templated SAXPY kernel and launches both float and double
  instantiations.
- `pytorch_example.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/pytorch_example.py>`_
  launches a CUDA kernel with PyTorch tensors and a wrapped PyTorch stream.

Multi-device and advanced launch configuration
----------------------------------------------

- `simple_multi_gpu_example.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/simple_multi_gpu_example.py>`_
  compiles and launches kernels across multiple GPUs.
- `thread_block_cluster.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/thread_block_cluster.py>`_
  demonstrates thread block cluster launch configuration on Hopper-class GPUs.
- `tma_tensor_map.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/tma_tensor_map.py>`_
  demonstrates Tensor Memory Accelerator descriptors and TMA-based bulk copies.

Linking and graphs
------------------

- `jit_lto_fractal.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/jit_lto_fractal.py>`_
  uses JIT link-time optimization to link user-provided device code into a
  fractal workflow at runtime.
- `cuda_graphs.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/cuda_graphs.py>`_
  captures and replays a multi-kernel CUDA graph to reduce launch overhead.

Interoperability and memory access
----------------------------------

- `memory_ops.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/memory_ops.py>`_
  covers memory resources, pinned memory, device transfers, and DLPack interop.
- `strided_memory_view_cpu.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/strided_memory_view_cpu.py>`_
  uses ``StridedMemoryView`` with JIT-compiled CPU code via ``cffi``.
- `strided_memory_view_gpu.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/strided_memory_view_gpu.py>`_
  uses ``StridedMemoryView`` with JIT-compiled GPU code and foreign GPU buffers.
- `gl_interop_plasma.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/gl_interop_plasma.py>`_
  renders a CUDA-generated plasma effect through OpenGL interop without CPU
  copies.

System inspection
-----------------

- `show_device_properties.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/show_device_properties.py>`_
  prints a detailed report of the CUDA devices available on the system.
