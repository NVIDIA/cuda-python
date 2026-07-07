.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Installation
============

Runtime Requirements
--------------------

``cuda.pathfinder`` is a pure-Python package with no runtime dependencies:

* Linux (x86-64, arm64) and Windows (x86-64)
* Python 3.10 - 3.14

Installing from PyPI
--------------------

.. code-block:: console

   $ pip install -U cuda-pathfinder

Installing from Conda (conda-forge)
-----------------------------------

.. code-block:: console

   $ conda install -c conda-forge cuda-pathfinder

Development environment
-----------------------

The sections above cover end-user installation. The section below focuses on
a repeatable *development* workflow (editable installs and running tests).

Installing the latest nightly (top-of-tree builds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are useful for users looking to test new features or bug fixes prior to
their inclusion in a release.

CI publishes the wheel as a GitHub Actions artifact on every push to ``main``.
Because ``cuda.pathfinder`` is pure Python, a single wheel covers every
supported Python version and platform. To obtain the most recent build, use
the following commands:

.. code-block:: console

   $ # Find the latest successful CI run on main:
   $ RUN_ID=$(gh run list -R NVIDIA/cuda-python -w ci.yml -b main -s success -L1 --json databaseId -q '.[0].databaseId')

   $ # Download the wheel:
   $ gh run download "$RUN_ID" -R NVIDIA/cuda-python -p "cuda-pathfinder-wheel"

   $ # Install the downloaded wheel:
   $ pip install cuda-pathfinder-wheel/*.whl

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ git clone https://github.com/NVIDIA/cuda-python
   $ cd cuda-python/cuda_pathfinder
   $ pip install .

For an editable install (e.g. when developing ``cuda.pathfinder`` itself):

.. code-block:: console

   $ pip install -v -e .
