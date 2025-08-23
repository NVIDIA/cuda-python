Installation
============

Runtime Requirements
--------------------

``cuda.core`` is supported on all platforms that CUDA is supported. Specific dependencies are as follows:

+------------------------------+---------------------------------------+---------------------------------------+
|                              | CUDA 11                               | CUDA 12                               |
+==============================+=======================================+=======================================+
| CUDA Toolkit  [1]_           | 11.2 - 11.8                           | 12.x                                  |
+------------------------------+---------------------------------------+---------------------------------------+
| Driver                       | 450.80.02+ (Linux), 452.39+ (Windows) | 525.60.13+ (Linux), 527.41+ (Windows) |
+------------------------------+---------------------------------------+---------------------------------------+

``cuda.core`` supports Python 3.9 - 3.13, on Linux (x86-64, arm64) and Windows (x86-64).

Installing from PyPI
--------------------

``cuda.core`` works with ``cuda.bindings`` (part of ``cuda-python``) 11 or 12. Test dependencies now use the ``cuda-toolkit`` metapackage for improved dependency resolution. For example with CUDA 12:

.. code:: console

   $ pip install cuda-core[cu12]

and likewise use ``[cu11]`` for CUDA 11, or ``[cu13]`` for CUDA 13.

Note that using ``cuda.core`` with NVRTC installed from PyPI via ``pip install`` requires ``cuda.bindings`` 12.8.0+ or 11.8.6+. Likewise, with nvJitLink it requires 12.8.0+.

Installing from Conda (conda-forge)
-----------------------------------

Same as above, ``cuda.core`` can be installed in a CUDA 11 or 12 environment. For example with CUDA 12:

.. code:: console

   $ conda install -c conda-forge cuda-core cuda-version=12

and likewise use ``cuda-version=11`` for CUDA 11.

Note that to use ``cuda.core`` with nvJitLink installed from conda-forge requires ``cuda.bindings`` 12.8.0+.

Installing from Source
----------------------

.. code:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_core
   $ pip install .

``cuda-bindings`` 11.x or 12.x is a required dependency.

.. [1]
   Including ``cuda-python``.
