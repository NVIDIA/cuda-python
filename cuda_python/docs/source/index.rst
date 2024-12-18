CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core`_: Pythonic access to CUDA runtime and other core functionalities
- `cuda.bindings`_: Low-level Python bindings to CUDA C APIs
- `cuda.cooperative`_: Pythonic exposure of CUB cooperative algorithms
- `cuda.parallel`_: Pythonic exposure of Thrust parallel algorithms

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
   conduct.md
   contribute.md
