Tips and Tricks
---------------

Getting the address of underlying C objects from the low-level bindings
=======================================================================

All CUDA C types are exposed to Python as Python classes. For example, the :class:`~cuda.bindings.driver.CUstream` type is exposed as a class with methods :meth:`~cuda.bindings.driver.CUstream.getPtr()` and :meth:`~cuda.bindings.driver.CUstream.__int__()` implemented.

There is an important distinction between the ``getPtr()`` method and the behaviour of ``__int__()``. If you need to get the pointer address *of* the underlying ``CUstream`` C object wrapped in the Python class, you can do so by calling ``int(instance_of_CUstream)``, which returns the address as a Python `int`, while calling ``instance_of_CUstream.getPtr()`` returns the pointer *to* the ``CUstream`` C object (that is, ``&CUstream``) as a Python `int`.


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

