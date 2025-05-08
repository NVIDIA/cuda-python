.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core`_: Pythonic access to CUDA runtime and other core functionalities
- `cuda.bindings`_: Low-level Python bindings to CUDA C APIs
- `cuda.cooperative`_: A Python package providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
- `cuda.parallel`_: A Python package for easy access to CCCL's highly efficient and customizable parallel algorithms, like ``sort``, ``scan``, ``reduce``, ``transform``, etc, that are callable on the *host*
- `numba.cuda`_: Numba's target for CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.

For access to NVIDIA CPU & GPU Math Libraries, please refer to `nvmath-python`_.

.. _nvmath-python: https://docs.nvidia.com/cuda/nvmath-python/latest

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components.
All of the previously available functionalities from the ``cuda-python`` package will continue to
be available, please refer to the `cuda.bindings`_ documentation for installation guide and further detail.

..
   The urls above can be auto-inserted by Sphinx (see rst_epilog in conf.py), but
   not for the urls below, which must be hard-coded due to Sphinx limitation...

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   release.md
   cuda.core <https://nvidia.github.io/cuda-python/cuda-core/latest>
   cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest>
   cuda.cooperative <https://nvidia.github.io/cccl/cuda_cooperative>
   cuda.parallel <https://nvidia.github.io/cccl/cuda_parallel>
   numba.cuda <https://nvidia.github.io/numba-cuda/>
