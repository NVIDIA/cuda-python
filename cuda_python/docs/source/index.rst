CUDA Python
===========

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of
multiple components:

- `cuda.core`_: Pythonic access to CUDA runtime and other core functionalities
- `cuda.bindings`_: Low-level Python bindings to CUDA C APIs
- `cuda.cooperative`_: Pythonic exposure of CUB cooperative algorithms
- `cuda.parallel`_: Pythonic exposure of Thrust parallel algorithms

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components.
All of the previously available functionalities from the ``cuda-python`` package will continue to
be available, please refer to the `cuda.bindings`_ documentation for installation guide and further detail.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   release.md
   conduct.md
   contribute.md
