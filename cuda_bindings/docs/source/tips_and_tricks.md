# Tips and Tricks

## Getting the address of underlying C objects from the low-level bindings

Within the low-level object wrappers CUDA types are exposed to Python as Python classes. For example, the CUdevice type is exposed as a PyObject with both an implementation of `GetPtr()`, and `__int__()`.

There is an important distinction between the `getPtr()` method and the behaviour of `__int__()`. If the user wants to get the address of the underlying C object, wrapped in the cdef python class, they should call `int(CUdeviceInstance)`, which returns a pointer to the object, while calling `CUdeviceInstance.getPtr()` returns the `void**` address of the pointer to the object.