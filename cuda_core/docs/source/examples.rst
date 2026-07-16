.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Examples
========

The canonical, runnable examples for ``cuda.core`` live under the top-level
:samples:`samples/ directory </>` of the cuda-python repository. They are
self-contained scripts (each with a ``README.md`` and PEP 723 dependency
block) and are exercised as part of the ``cuda.core`` test suite.

Use the samples directory as your first stop when looking for a runnable
starting point for a specific workflow. The remaining entries below are the
few examples still hosted alongside the package that have not yet been
migrated to ``samples/``.

Not yet migrated to samples/
----------------------------

- :cuda-core-example:`simple_multi_gpu_example.py <simple_multi_gpu_example.py>`
  compiles and launches kernels across multiple GPUs.
- :cuda-core-example:`thread_block_cluster.py <thread_block_cluster.py>`
  demonstrates thread block cluster launch configuration on Hopper-class GPUs.
- :cuda-core-example:`strided_memory_view_constructors.py <strided_memory_view_constructors.py>`
  walks through the explicit ``StridedMemoryView.from_*`` constructors.
- :cuda-core-example:`strided_memory_view_cpu.py <strided_memory_view_cpu.py>`
  uses ``StridedMemoryView`` with JIT-compiled CPU code via ``cffi``.
- :cuda-core-example:`strided_memory_view_gpu.py <strided_memory_view_gpu.py>`
  uses ``StridedMemoryView`` with JIT-compiled GPU code and foreign GPU buffers.
- :cuda-core-example:`gl_interop_plasma.py <gl_interop_plasma.py>`
  renders a CUDA-generated plasma effect through OpenGL interop without CPU
  copies.
