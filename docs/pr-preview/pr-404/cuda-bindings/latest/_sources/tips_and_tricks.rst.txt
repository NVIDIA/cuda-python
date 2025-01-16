Tips and Tricks
---------------

Getting the address of underlying C objects from the low-level bindings
=======================================================================

All CUDA C types are exposed to Python as Python classes. For example, the :class:`~cuda.bindings.driver.CUstream` type is exposed as a class with methods :meth:`~cuda.bindings.driver.CUstream.getPtr()` and :meth:`~cuda.bindings.driver.CUstream.__int__()` implemented.

There is an important distinction between the ``getPtr()`` method and the behaviour of ``__int__()``. If you need to get the pointer address *of* the underlying ``CUstream`` C object wrapped in the Python class, you can do so by calling ``int(instance_of_CUstream)``, which returns the address as a Python `int`, while calling ``instance_of_CUstream.getPtr()`` returns the pointer *to* the ``CUstream`` C object (that is, ``&CUstream``) as a Python `int`.
