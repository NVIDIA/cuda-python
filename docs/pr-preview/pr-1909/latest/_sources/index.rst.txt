.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core`_: Pythonic access to CUDA Runtime and other core functionality
- `cuda.bindings`_: Low-level Python bindings to CUDA C APIs
- `cuda.pathfinder`_: Utilities for locating CUDA components installed in the user's Python environment
- `cuda.coop`_: A Python module providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
- `cuda.compute`_: A Python module for easy access to CCCL's highly efficient and customizable parallel algorithms, like ``sort``, ``scan``, ``reduce``, ``transform``, etc. that are callable on the *host*
- `numba.cuda`_: A Python DSL that exposes CUDA **SIMT** programming model and compiles a restricted subset of Python code into CUDA kernels and device functions
- `cuda.tile`_: A new Python DSL that exposes CUDA **Tile** programming model and allows users to write NumPy-like code in CUDA kernels
- `nvmath-python`_: Pythonic access to NVIDIA CPU & GPU Math Libraries, with `host`_, `device`_, and `distributed`_ APIs. It also provides low-level Python bindings to host C APIs (`nvmath.bindings`_).
- `nvshmem4py`_: Pythonic interface to the NVSHMEM library, enabling Python applications to leverage NVSHMEM's high-performance PGAS (Partitioned Global Address Space) programming model for GPU-accelerated computing
- `Nsight Python`_: Python kernel profiling interface that automates performance analysis across multiple kernel configurations using NVIDIA Nsight Tools
- `CUPTI Python`_: Python APIs for creation of profiling tools that target CUDA Python applications via the CUDA Profiling Tools Interface (CUPTI)
- `Accelerated Computing Hub`_: Open-source learning materials related to GPU computing. You will find user guides, tutorials, and other works freely available for all learners interested in GPU computing.

.. _cuda.coop: https://nvidia.github.io/cccl/python/coop
.. _cuda.compute: https://nvidia.github.io/cccl/python/compute
.. _numba.cuda: https://nvidia.github.io/numba-cuda/
.. _cuda.tile: https://docs.nvidia.com/cuda/cutile-python/
.. _nvmath-python: https://docs.nvidia.com/cuda/nvmath-python/latest
.. _host: https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#host-apis
.. _device: https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#device-apis
.. _distributed: https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html
.. _nvmath.bindings: https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html
.. _nvshmem4py: https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html
.. _Nsight Python: https://docs.nvidia.com/nsight-python/index.html
.. _CUPTI Python: https://docs.nvidia.com/cupti-python/
.. _Accelerated Computing Hub: https://github.com/NVIDIA/accelerated-computing-hub

CUDA Python is currently undergoing an overhaul to improve existing and introduce new components.
All of the previously available functionality from the ``cuda-python`` package will continue to
be available, please refer to the `cuda.bindings`_ documentation for installation guide and further detail.

..
   The urls above can be auto-inserted by Sphinx (see rst_epilog in conf.py), but
   not for the urls below, which must be hard-coded due to Sphinx limitation...

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   release
   cuda.core <https://nvidia.github.io/cuda-python/cuda-core/latest>
   cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest>
   cuda.pathfinder <https://nvidia.github.io/cuda-python/cuda-pathfinder/latest>
   cuda.coop <https://nvidia.github.io/cccl/python/coop>
   cuda.compute <https://nvidia.github.io/cccl/python/compute>
   numba.cuda <https://nvidia.github.io/numba-cuda/>
   cuda.tile <https://docs.nvidia.com/cuda/cutile-python/>
   nvmath-python <https://docs.nvidia.com/cuda/nvmath-python/>
   nvshmem4py <https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html>
   Nsight Python <https://docs.nvidia.com/nsight-python/index.html>
   CUPTI Python <https://docs.nvidia.com/cupti-python/>
   Accelerated Computing Hub <https://github.com/NVIDIA/accelerated-computing-hub>
