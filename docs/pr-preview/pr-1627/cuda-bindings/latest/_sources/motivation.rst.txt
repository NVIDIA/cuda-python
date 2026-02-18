.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Motivation
==========
What is CUDA Python?
--------------------

NVIDIA’s CUDA Python provides `Cython <https://cython.org/>`_ bindings and Python
wrappers for the driver and runtime API for existing toolkits and libraries to
simplify GPU-based accelerated processing. Python is one of the most popular
programming languages for science, engineering, data analytics, and deep
learning applications.  The goal of CUDA Python is to unify
the Python ecosystem with a single set of interfaces that provide full coverage
of and access to the CUDA host APIs from Python.

Why CUDA Python?
----------------

CUDA Python provides uniform APIs and bindings for inclusion into existing
toolkits and libraries to simplify GPU-based parallel processing for HPC, data
science, and AI.

`Numba <https://numba.pydata.org/>`_, a Python compiler from
`Anaconda <https://www.anaconda.com/>`_ that can compile Python code for execution
on CUDA-capable GPUs, provides Python developers with an easy entry into
GPU-accelerated computing and a path for using increasingly sophisticated CUDA
code with a minimum of new syntax and jargon. Numba has its own CUDA driver API
bindings that can now be replaced with CUDA Python. With CUDA Python and Numba,
you get the best of both worlds: rapid iterative development with Python and the
speed of a compiled language targeting both CPUs and NVIDIA GPUs.

`CuPy <https://cupy.dev/>`_ is a
`NumPy <https://numpy.org/>`_/`SciPy <https://www.scipy.org/>`_ compatible Array
library, from `Preferred Networks <https://www.preferred.jp/en/>`_, for
GPU-accelerated computing with Python. CUDA Python simplifies the CuPy build
and allows for a faster and smaller memory footprint when importing the CuPy
Python module. In the future, when more CUDA Toolkit libraries are supported,
CuPy will have a lighter maintenance overhead and have fewer wheels to
release. Users benefit from a faster CUDA runtime!

Our goal is to help unify the Python CUDA ecosystem with a single standard set
of interfaces, providing full coverage of, and access to, the CUDA host APIs
from Python. We want to provide a foundation for the ecosystem to build on top
of in unison to allow composing different accelerated libraries together to
solve the problems at hand. We also want to lower the barrier to entry for
Python developers to utilize NVIDIA GPUs.
