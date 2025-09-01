.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core`_: Pythonic access to CUDA runtime and other core functionalities
- `cuda.bindings`_: Low-level Python bindings to CUDA C APIs
- `cuda.pathfinder`_: Utilities for locating CUDA components installed in the user's Python environment
- `cuda.cccl.cooperative`_: A Python module providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
- `cuda.cccl.parallel`_: A Python module for easy access to CCCL's highly efficient and customizable parallel algorithms, like ``sort``, ``scan``, ``reduce``, ``transform``, etc, that are callable on the *host*
- `numba.cuda`_: Numba's target for CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.
* `nvmath-python`_: Pythonic access to NVIDIA CPU & GPU Math Libraries, with both *host* and *device* (through `nvmath.device`_) APIs. It also provides low-level Python bindings to host C APIs (through `nvmath.bindings`_).

.. _nvmath-python: https://docs.nvidia.com/cuda/nvmath-python/latest
.. _nvmath.device: https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#device-apis
.. _nvmath.bindings: https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components.
All of the previously available functionalities from the ``cuda-python`` package will continue to
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
   cuda.cccl.cooperative <https://nvidia.github.io/cccl/python/cooperative>
   cuda.cccl.parallel <https://nvidia.github.io/cccl/python/parallel>
   numba.cuda <https://nvidia.github.io/numba-cuda/>
   nvmath-python <https://docs.nvidia.com/cuda/nvmath-python/>
