.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Tips and Tricks
---------------

Getting the address of underlying C objects from the low-level bindings
=======================================================================

.. warning::

   Using ``int(cuda_obj)`` to retrieve the underlying address of a CUDA object is deprecated and
   subject to future removal. Please switch to use :func:`~cuda.bindings.utils.get_cuda_native_handle`
   instead.

All CUDA C types are exposed to Python as Python classes. For example, the :class:`~cuda.bindings.driver.CUstream` type is exposed as a class with methods :meth:`~cuda.bindings.driver.CUstream.getPtr()` and :meth:`~cuda.bindings.driver.CUstream.__int__()` implemented.

There is an important distinction between the ``getPtr()`` method and the behaviour of ``__int__()``. Since a ``CUstream`` is itself just a pointer, calling ``instance_of_CUstream.getPtr()`` returns the pointer *to* the pointer, instead of the value of the ``CUstream`` C object that is the pointer to the underlying stream handle. ``int(instance_of_CUstream)`` returns the value of the ``CUstream`` converted to a Python int and is the actual address of the underlying handle.


Lifetime management of the CUDA objects
=======================================

All of the Python classes do not manage the lifetime of the underlying CUDA C objects. It is the user's responsibility to use the appropriate APIs to explicitly destruct the objects following the CUDA Programming Guide.


Getting and setting attributes of extension types
=================================================

While the bindings outwardly present the attributes of extension types in a pythonic way, they can't always be interacted with in a Pythonic style. Often the getters/setters (__getitem__(), __setitem__()) are actually a translation step to convert values between Python and C. For example, in some cases, attempting to modify an attribute in place, will lead to unexpected behavior due to the design of the underlying implementation. For this reason, users should use the getters and setters directly when interacting with extension types.

An example of this is the :class:`~cuda.bindings.driver.CULaunchConfig` type.

.. code-block:: python

    cfg = cuda.CUlaunchConfig()

    cfg.numAttrs += 1
    attr = cuda.CUlaunchAttribute()

    ...

    # This works. We are passing the new attribute to the setter
    drv_cfg.attrs = [attr]

    # This does not work. We are only modifying the returned attribute in place
    drv_cfg.attrs.append(attr)
