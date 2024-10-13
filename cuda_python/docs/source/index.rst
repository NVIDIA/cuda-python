CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core <https://nvidia.github.io/cuda-python/cuda-core>`_: Pythonic access to CUDA
  runtime and other core functionalities
- `cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings>`_: Low-level Python
  bindings to CUDA C APIs
- `cuda.cooperative <https://nvidia.github.io/cccl/cuda_cooperative/>`_: Pythonic exposure
  of CUB cooperative algorithms.
- `cuda.parallel <https://nvidia.github.io/cccl/cuda_parallel/>`_: Pythonic exposure of
  Thrust parallel algorithms.

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components.
All of the previously available functionalities from the ``cuda-python`` package will continue to
be available, please refer to the `cuda.bindings <https://nvidia.github.io/cuda-python/cuda-bindings>`_
documentation for more detail.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   release.md
   conduct.md
   contribute.md
