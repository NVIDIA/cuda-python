.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Examples
========

This page links to the ``cuda.core`` examples shipped in the
:cuda-core-examples:`cuda-python repository </>`.
Use it as a quick index when you want a runnable starting point for a specific
workflow.

Compilation and kernel launch
-----------------------------

- :cuda-core-example:`vector_add.py`
  compiles and launches a simple vector-add kernel with CuPy arrays.
- :cuda-core-example:`saxpy.py`
  JIT-compiles a templated SAXPY kernel and launches both float and double
  instantiations.
- :cuda-core-example:`pytorch_example.py`
  launches a CUDA kernel with PyTorch tensors and a wrapped PyTorch stream.

Multi-device and advanced launch configuration
----------------------------------------------

- :cuda-core-example:`simple_multi_gpu_example.py`
  compiles and launches kernels across multiple GPUs.
- :cuda-core-example:`thread_block_cluster.py`
  demonstrates thread block cluster launch configuration on Hopper-class GPUs.
- :cuda-core-example:`tma_tensor_map.py`
  demonstrates Tensor Memory Accelerator descriptors and TMA-based bulk copies.

Linking and graphs
------------------

- :cuda-core-example:`jit_lto_fractal.py`
  uses JIT link-time optimization to link user-provided device code into a
  fractal workflow at runtime.
- :cuda-core-example:`cuda_graphs.py`
  captures and replays a multi-kernel CUDA graph to reduce launch overhead.

Interoperability and memory access
----------------------------------

- :cuda-core-example:`memory_ops.py`
  covers memory resources, pinned memory, device transfers, and DLPack interop.
- :cuda-core-example:`strided_memory_view_cpu.py`
  uses ``StridedMemoryView`` with JIT-compiled CPU code via ``cffi``.
- :cuda-core-example:`strided_memory_view_gpu.py`
  uses ``StridedMemoryView`` with JIT-compiled GPU code and foreign GPU buffers.
- :cuda-core-example:`gl_interop_plasma.py`
  renders a CUDA-generated plasma effect through OpenGL interop without CPU
  copies.

System inspection
-----------------

- :cuda-core-example:`show_device_properties.py`
  prints a detailed report of the CUDA devices available on the system.
