.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Installation
============

Runtime Requirements
--------------------

``cuda.bindings`` supports the same platforms as CUDA. Runtime dependencies are:

* Linux (x86-64, arm64) and Windows (x86-64)
* Python 3.9 - 3.13
* Driver: Linux (580.65.06 or later) Windows (580.88 or later)
* Optionally, NVRTC, nvJitLink, NVVM, and cuFile from CUDA Toolkit 13.x

.. note::

   The optional CUDA Toolkit components are now installed via the ``cuda-toolkit`` metapackage from PyPI for improved dependency resolution. Components can also be installed via Conda, OS-specific package managers, or local installers (as described in the CUDA Toolkit `Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_ and `Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ Installation Guides).

Starting from v12.8.0, ``cuda-python`` becomes a meta package which currently depends only on ``cuda-bindings``; in the future more sub-packages will be added to ``cuda-python``. In the instructions below, we still use ``cuda-python`` as example to serve existing users, but everything is applicable to ``cuda-bindings`` as well.

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
* ``nvidia-nvvm`` (NVVM library)
* ``nvidia-cufile`` (cuFile library, Linux only)

These are now installed through the ``cuda-toolkit`` metapackage for improved dependency resolution.

Installing from Conda
---------------------

.. code-block:: console

   $ conda install -c conda-forge cuda-python

.. note::

   When using conda, the ``cuda-version`` metapackage can be used to control the versions of CUDA Toolkit components that are installed to the conda environment.

For example:

.. code-block:: console

   $ conda install -c conda-forge cuda-python cuda-version=13

Installing from Source
----------------------

Requirements
^^^^^^^^^^^^

* CUDA Toolkit headers[^1]
* CUDA Runtime static library[^2]

[^1]: User projects that ``cimport`` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the ``cuda.bindings`` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

[^2]: The CUDA Runtime static library (``libcudart_static.a`` on Linux, ``cudart_static.lib`` on Windows) is part of the CUDA Toolkit. If using conda packages, it is contained in the ``cuda-cudart-static`` package.

Source builds require that the provided CUDA headers are of the same major.minor version as the ``cuda.bindings`` you're trying to build. Despite this requirement, note that the minor version compatibility is still maintained. Use the ``CUDA_HOME`` (or ``CUDA_PATH``) environment variable to specify the location of your headers. For example, if your headers are located in ``/usr/local/cuda/include``, then you should set ``CUDA_HOME`` with:

.. code-block:: console

   $ export CUDA_HOME=/usr/local/cuda

See `Environment Variables <environment_variables.rst>`_ for a description of other build-time environment variables.

.. note::

   Only ``cydriver``, ``cyruntime`` and ``cynvrtc`` are impacted by the header requirement.

Editable Install
^^^^^^^^^^^^^^^^

You can use:

.. code-block:: console

   $ pip install -v -e .

to install the module as editable in your current Python environment (e.g. for testing of porting other libraries to use the binding).
