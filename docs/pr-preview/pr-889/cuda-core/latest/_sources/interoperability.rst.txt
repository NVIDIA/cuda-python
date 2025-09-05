.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.core.experimental

Interoperability
================

``cuda.core`` is designed to be interoperable with other Python GPU libraries. Below
we cover a list of possible such scenarios.


Current device/context
----------------------

The :meth:`Device.set_current` method ensures that the calling host thread has
an active CUDA context set to current. This CUDA context can be seen and accessed
by other GPU libraries without any code change. For libraries built on top of
the `CUDA runtime <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>`_,
this is as if ``cudaSetDevice`` is called.

Since CUDA contexts are per-thread constructs, in a multi-threaded program each
host thread should call this method.

Conversely, if any GPU library already sets a device (or context) to current, this
method ensures that the same device/context is picked up by and shared with
``cuda.core``.


``__cuda_stream__`` protocol
----------------------------

The :class:`~_stream.Stream` class is a vocabulary type representing CUDA streams
in Python. While we encourage new Python projects to start using streams (and other
CUDA types) from ``cuda.core``, we understand that there are already several projects
exposing their own stream types.

To address this issue, we propose the :attr:`~_stream.IsStreamT.__cuda_stream__` protocol
(currently version 0) as follows: For any Python objects that are meant to be interpreted
as a stream, they should add a ``__cuda_stream__`` *method* that returns a 2-tuple: The
version number (``0``) and the address of ``cudaStream_t`` (both as Python ``int``):

.. code-block:: python

   class MyStream:

       def __cuda_stream__(self):
           return (0, self.ptr)

       ...

Then such objects can be understood and wrapped by :meth:`Device.create_stream`.

We suggest all existing Python projects that already expose a stream class to also support
this protocol. For new Python projects that need to access CUDA streams, we encourage you
to use :class:`~_stream.Stream` from ``cuda.core`` directly.


Memory view utilities for CPU/GPU buffers
-----------------------------------------

The Python community has defined protocols such as CUDA Array Interface (CAI) [1]_ and DLPack
[2]_ (part of the Python array API standard [3]_) for facilitating zero-copy data exchange
between two GPU projects. In particular, performance considerations prompted the protocol
designs gearing toward *stream-ordered* operations so as to avoid unnecessary synchronizations.
While the designs are robust, *implementing* such protocols can be tricky and often requires
a few iterations to ensure correctness.

``cuda.core`` offers a :func:`~utils.args_viewable_as_strided_memory` decorator for
extracting the metadata (such as pointer address, shape, strides, and dtype) from any
Python objects supporting either CAI or DLPack and returning a :class:`~utils.StridedMemoryView` object, see the
`strided_memory_view.py <https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/strided_memory_view.py>`_
example. Alternatively, a :class:`~utils.StridedMemoryView` object can be explicitly
constructed without using the decorator. This provides a *concrete implementation* to both
protocols that is **array-library-agnostic**, so that all Python projects can just rely on this
without either re-implementing (the consumer-side of) the protocols or tying to any particular
array libraries.

The :attr:`~utils.StridedMemoryView.is_device_accessible` attribute can be used to check
whether or not the underlying buffer can be accessed on GPU.

.. rubric:: Footnotes

.. [1] https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
.. [2] https://dmlc.github.io/dlpack/latest/python_spec.html
.. [3] https://data-apis.org/array-api/latest/design_topics/data_interchange.html
