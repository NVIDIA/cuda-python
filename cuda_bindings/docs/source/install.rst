.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Installation
============

Runtime Requirements
--------------------

``cuda.bindings`` supports the same platforms as CUDA. Runtime dependencies are:

* Linux (x86-64, arm64) and Windows (x86-64)
* Python 3.10 - 3.14
* Driver: Linux (580.65.06 or later) Windows (580.88 or later)
* Optionally, NVRTC, nvJitLink, nvFatBin, NVVM, cuFile, and cuDLA from CUDA Toolkit 13.x

.. note::

   The optional CUDA Toolkit components are now installed via the ``cuda-toolkit`` metapackage from PyPI for improved dependency resolution. Components can also be installed via Conda, OS-specific package managers, or local installers (as described in the CUDA Toolkit `Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_ and `Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ Installation Guides).

Starting from v12.8.0, ``cuda-python`` becomes a meta package which currently depends only on ``cuda-bindings``; in the future more sub-packages will be added to ``cuda-python``. In the instructions below, we still use ``cuda-python`` as example to serve existing users, but everything is applicable to ``cuda-bindings`` as well.


Free-threading Build Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of cuda-bindings 13.0.2 and 12.9.3, **experimental** packages for the `free-threaded interpreter`_ are shipped.

1. Support for these builds is best effort, due to heavy use of `built-in
   modules that are known to be thread-unsafe`_, such as ``ctypes``.
2. For now, you are responsible for making sure that calls into the ``cuda-bindings``
   library are thread-safe. This is subject to change.

.. _built-in modules that are known to be thread-unsafe: https://github.com/python/cpython/issues/116738
.. _free-threaded interpreter: https://docs.python.org/3/howto/free-threading-python.html


Installing from PyPI
--------------------

.. code-block:: console

   $ pip install -U cuda-python

Install all optional dependencies with:

.. code-block:: console

   $ pip install -U cuda-python[all]

Where the optional dependencies include:

* ``nvidia-cuda-nvrtc`` (NVRTC runtime compilation library)
* ``nvidia-nvjitlink`` (nvJitLink library)
* ``nvidia-nvfatbin`` (nvFatBin library)
* ``nvidia-nvvm`` (NVVM library)
* ``nvidia-cufile`` (cuFile library, Linux only)
* ``nvidia-cudla`` (cuDLA library, Linux aarch64 only)

These are now installed through the ``cuda-toolkit`` metapackage, where available, for improved dependency resolution.

Installing from Conda
---------------------

.. code-block:: console

   $ conda install -c conda-forge cuda-python

.. note::

   When using conda, the ``cuda-version`` metapackage can be used to control the versions of CUDA Toolkit components that are installed to the conda environment.

For example:

.. code-block:: console

   $ conda install -c conda-forge cuda-python cuda-version=13

.. note::

   Tegra users can install the cuDLA conda package from conda-forge through ``conda install -c conda-forge libcudla cuda-version=13``, if it does not already exist on the system.

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
   $ gh run download "$RUN_ID" -R NVIDIA/cuda-python -p "cuda-bindings-python312-cuda13*-linux-64-*"

   $ # Install the downloaded wheel:
   $ pip install cuda-bindings-python312-cuda13*-linux-64-*/cuda_bindings*.whl[all]

Replace ``python312`` with your Python version (e.g. ``python310``, ``python311``,
``python313``, ``python314``, ``python314t``). For aarch64, replace ``linux-64``
with ``linux-aarch64``; for Windows, use ``win-64``. Only the current CUDA
major version is built on ``main``; wheels for the prior CUDA major are
published from the corresponding backport branch.

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

Requirements
^^^^^^^^^^^^

* CUDA Toolkit headers[^1]
* CUDA Runtime static library[^2]

[^1]: User projects that ``cimport`` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the ``cuda.bindings`` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

[^2]: The CUDA Runtime static library (``libcudart_static.a`` on Linux, ``cudart_static.lib`` on Windows) is part of the CUDA Toolkit. If using conda packages, it is contained in the ``cuda-cudart-static`` package.

Source builds require that the provided CUDA headers are of the same major.minor version as the ``cuda.bindings`` you're trying to build. Despite this requirement, note that the minor version compatibility is still maintained. Use the ``CUDA_PATH`` (or ``CUDA_HOME``) environment variable to specify the location of your headers. If both are set, ``CUDA_PATH`` takes precedence. For example, if your headers are located in ``/usr/local/cuda/include``, then you should set ``CUDA_PATH`` with:

.. code-block:: console

   $ export CUDA_PATH=/usr/local/cuda

See :doc:`Environment Variables <environment_variables>` for a description of other build-time environment variables.

.. note::

   Only ``cydriver``, ``cyruntime`` and ``cynvrtc`` are impacted by the header requirement.

Editable Install
^^^^^^^^^^^^^^^^

You can use:

.. code-block:: console

   $ pip install -v -e .

to install the module as editable in your current Python environment (e.g. for testing of porting other libraries to use the binding).
