.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Starting ``cuda-core`` 0.4.0, **experimental** packages for the `free-threaded interpreter`_ are shipped.

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


Development environment
-----------------------

The sections above cover end-user installation. The section below focuses on
a repeatable *development* workflow (editable installs and running tests).

Installing the latest nightly (top-of-tree builds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are useful for users looking to test new features or bug fixes prior to
their inclusion in a release.

CI publishes wheels as GitHub Actions artifacts on every push to ``main``. To
obtain the most recent build, use the following commands:

.. code-block:: console

   $ # Find the latest successful CI run on main:
   $ RUN_ID=$(gh run list -R NVIDIA/cuda-python -w ci.yml -b main -s success -L1 --json databaseId -q '.[0].databaseId')

   $ # Download the wheel (pick your Python version and platform):
   $ gh run download "$RUN_ID" -R NVIDIA/cuda-python -p "cuda-core-python312-linux-64-*"

   $ # Install the downloaded wheel:
   $ pip install cuda-core-python312-linux-64-*/cuda_core*.whl[cu13]

Replace ``python312`` with your Python version (e.g. ``python310``, ``python311``,
``python313``, ``python314``, ``python314t``). For aarch64, replace ``linux-64``
with ``linux-aarch64``; for Windows, use ``win-64``. Replace ``cu13`` with
``cu12`` for CUDA 12.x environments.

Development with uv
~~~~~~~~~~~~~~~~~~~

`uv`_ is a fast Python package and project manager. For example, to work on
``cuda-core`` against CUDA 13:

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_core
   $ uv venv
   $ source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   $ uv pip install -e .[cu13] --group test

Run tests:

.. code-block:: console

   $ python -m pytest tests

.. _uv: https://docs.astral.sh/uv/

Development with pixi
~~~~~~~~~~~~~~~~~~~~~

`pixi`_ provides a reproducible development environment across the repository.
From the repository root:

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python
   $ pixi run -e cu13 test-core

To run all repository tests (pathfinder → bindings → core):

.. code-block:: console

   $ pixi run -e cu13 test

Use ``-e cu12`` to test against CUDA 12 instead.

.. _pixi: https://pixi.sh/

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_core
   $ pip install .

``cuda-bindings`` 12.x or 13.x is a required dependency.
