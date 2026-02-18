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


``cuda.core`` supports Python 3.10 - 3.14, on Linux (x86-64, arm64) and Windows (x86-64). **Experimental** free-threaded builds for Python 3.14 are also provided.


Free-threading Build Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of cuda-core 0.4.0, **experimental** packages for the `free-threaded interpreter`_ are shipped.

1. Support for these builds is best effort, due to heavy use of `built-in
   modules that are known to be thread-unsafe`_, such as ``ctypes``.
2. For now, you are responsible for making sure that calls into the ``cuda-core``
   library are thread-safe. This is subject to change.

.. _built-in modules that are known to be thread-unsafe: https://github.com/python/cpython/issues/116738
.. _free-threaded interpreter: https://docs.python.org/3/howto/free-threading-python.html

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


Installing with uv
------------------

`uv`_ is a fast Python package and project manager. To install ``cuda-core`` using ``uv``:

.. code-block:: console

   $ uv pip install cuda-core[cu12]

and likewise use ``[cu13]`` for CUDA 13.

``uv`` can also manage virtual environments automatically:

.. code-block:: console

   $ uv venv
   $ uv pip install cuda-core[cu12]

.. _uv: https://docs.astral.sh/uv/

Installing with pixi
--------------------

`pixi`_ is a cross-platform package manager built on top of the conda ecosystem. To install ``cuda-core`` in a pixi project:

.. code-block:: console

   $ pixi init my-cuda-project
   $ cd my-cuda-project
   $ pixi add cuda-core --channel conda-forge

Or add it to an existing ``pixi.toml``:

.. code-block:: toml

   [dependencies]
   cuda-core = "*"

.. note::

   Use the ``cuda-version`` package to pin the CUDA Toolkit version in your pixi environment:

.. code-block:: console

   $ pixi add cuda-version=12 --channel conda-forge

.. _pixi: https://pixi.sh/

Installing from Source
----------------------

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_core
   $ pip install .

``cuda-bindings`` 12.x or 13.x is a required dependency.
