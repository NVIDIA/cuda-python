.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

*******************************************************
cuda-pathfinder: Utilities for locating CUDA components
*******************************************************

.. image:: https://img.shields.io/badge/NVIDIA-black?logo=nvidia
   :target: https://www.nvidia.com/
   :alt: NVIDIA

`cuda.pathfinder <https://nvidia.github.io/cuda-python/cuda-pathfinder/>`_
aims to be a one-stop solution for locating CUDA components. Currently
it supports locating and loading dynamic libraries (``.so``, ``.dll``);
support for headers and other artifacts is in progress.

* `Documentation <https://nvidia.github.io/cuda-python/cuda-pathfinder/>`_
* `Releases <https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/release.html>`_
* `Repository <https://github.com/NVIDIA/cuda-python/tree/main/cuda_pathfinder/>`_
* `Issue tracker <https://github.com/NVIDIA/cuda-python/issues/>`_ (select component ``cuda.pathfinder``)

``cuda.pathfinder`` is under active development. Feedback and suggestions are welcome.


Installation
============

.. code-block:: bash

   pip install cuda-pathfinder

``cuda-pathfinder`` is `CUDA Toolkit (CTK) <https://developer.nvidia.com/cuda-toolkit>`_
version-agnostic. It follows the general CUDA Toolkit support policy: the
two most recent major versions are supported simultaneously.
