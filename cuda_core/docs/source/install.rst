.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Installation
============

Runtime Requirements
--------------------

``cuda.core`` is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

.. list-table::
   :header-rows: 1

   * -
     - CUDA 12
     - CUDA 13
   * - CUDA Toolkit\ [#f1]_
     - 12.x
     - 13.x
   * - Driver
     - 525.60.13+ (Linux), 527.41+ (Windows)
     - 580.65+ (Linux), 580.88+ (Windows)

.. [#f1] Including ``cuda-python``.


``cuda.core`` supports Python 3.9 - 3.13, on Linux (x86-64, arm64) and Windows (x86-64).

Installing from PyPI
--------------------

``cuda.core`` works with ``cuda.bindings`` (part of ``cuda-python``) 12 or 13. Test dependencies now use the ``cuda-toolkit`` metapackage for improved dependency resolution. For example with CUDA 12:

.. code-block:: console

   $ pip install cuda-core[cu12]

and likewise use ``[cu13]`` for CUDA 13.

Note that using ``cuda.core`` with NVRTC installed from PyPI via ``pip install`` requires
``cuda.bindings`` 12.8.0+. Likewise, with nvJitLink it requires 12.8.0+.

Installing from Conda (conda-forge)
-----------------------------------

Same as above, ``cuda.core`` can be installed in a CUDA 12 or 13 environment. For example with CUDA 12:

.. code-block:: console

   $ conda install -c conda-forge cuda-core cuda-version=12

and likewise use ``cuda-version=13`` for CUDA 13.

Note that to use ``cuda.core`` with nvJitLink installed from conda-forge requires ``cuda.bindings`` 12.8.0+.

Installing from Source
----------------------

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_core
   $ pip install .

``cuda-bindings`` 12.x or 13.x is a required dependency.
