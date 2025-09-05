.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

**************************************************************
cuda-python: Metapackage collection of CUDA Python subpackages
**************************************************************

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of multiple components:

* `cuda.core <https://nvidia.github.io/cuda-python/cuda-core/latest>`_: Pythonic access to CUDA Runtime and other core functionalities
* `cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest>`_: Low-level Python bindings to CUDA C APIs
* `cuda.cooperative <https://nvidia.github.io/cccl/python/cooperative>`_: A Python package providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
* `cuda.parallel <https://nvidia.github.io/cccl/python/parallel>`_: A Python package for easy access to CCCL's highly efficient and customizable parallel algorithms, like ``sort``, ``scan``, ``reduce``, ``transform``, etc, that are callable on the *host*
* `numba.cuda <https://nvidia.github.io/numba-cuda/>`_: Numba's target for CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.

For access to NVIDIA CPU & GPU Math Libraries, please refer to `nvmath-python <https://docs.nvidia.com/cuda/nvmath-python/latest>`_.

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components. All of the previously available functionalities from the ``cuda-python`` package will continue to be available, please refer to the `cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest>`_ documentation for installation guide and further detail.

cuda-python as a metapackage
============================

``cuda-python`` is now a metapackage that contains a collection of subpackages. Each subpackage is versioned independently, allowing installation of each component as needed.

Subpackage: cuda.core
---------------------

The ``cuda.core`` package offers idiomatic, pythonic access to CUDA Runtime and other functionalities.

The goals are to

1. Provide **idiomatic ("pythonic")** access to CUDA Driver, Runtime, and JIT compiler toolchain
2. Focus on **developer productivity** by ensuring end-to-end CUDA development can be performed quickly and entirely in Python
3. **Avoid homegrown** Python abstractions for CUDA for new Python GPU libraries starting from scratch
4. **Ease** developer **burden of maintaining** and catching up with latest CUDA features
5. **Flatten the learning curve** for current and future generations of CUDA developers

Subpackage: cuda.bindings
-------------------------

The ``cuda.bindings`` package is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python.

The list of available interfaces are:

* CUDA Driver
* CUDA Runtime
* NVRTC
* nvJitLink
* NVVM
* cuFile
