cimport cpython
from libc.stdint cimport intptr_t
from libcpp.utility cimport move
from cython.operator cimport dereference as deref


cdef bint is_nested_sequence(data):
    if not cpython.PySequence_Check(data):
        return False
    else:
        for i in data:
            if not cpython.PySequence_Check(i):
                return False
        else:
            return True


cdef void* get_buffer_pointer(buf, Py_ssize_t size, readonly=True) except*:
    """The caller must ensure ``buf`` is alive when the returned pointer is in use."""
    cdef void* bufPtr
    cdef int flags = cpython.PyBUF_ANY_CONTIGUOUS
    if not readonly:
        flags |= cpython.PyBUF_WRITABLE
    cdef int status = -1
    cdef cpython.Py_buffer view

    if isinstance(buf, int):
        bufPtr = <void*><intptr_t>buf
    else:  # try buffer protocol
        try:
            status = cpython.PyObject_GetBuffer(buf, &view, flags)
            assert view.len == size
            assert view.ndim == 1
        except Exception as e:
            adj = "writable " if not readonly else ""
            raise ValueError(
                 "buf must be either a Python int representing the pointer "
                f"address to a valid buffer, or a 1D contiguous {adj}"
                 "buffer, of size bytes") from e
        else:
            bufPtr = view.buf
        finally:
            if status == 0:
                cpython.PyBuffer_Release(&view)

    return bufPtr


# Cython can't infer the ResT overload when it is wrapped in nullable_unique_ptr,
# so we need a dummy (__unused) input argument to help it
cdef int get_resource_ptr(nullable_unique_ptr[vector[ResT]] &in_out_ptr, object obj, ResT* __unused) except 1:
    if cpython.PySequence_Check(obj):
        vec = new vector[ResT](len(obj))
        # set the ownership immediately to avoid leaking the `vec` memory in
        # case of exception in the following loop
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            deref(vec)[i] = obj[i]
    else:
        in_out_ptr.reset(<vector[ResT]*><intptr_t>obj, False)
    return 0


cdef int get_resource_ptrs(nullable_unique_ptr[ vector[PtrT*] ] &in_out_ptr, object obj, PtrT* __unused) except 1:
    if cpython.PySequence_Check(obj):
        vec = new vector[PtrT*](len(obj))
        # set the ownership immediately to avoid leaking the `vec` memory in
        # case of exception in the following loop
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            deref(vec)[i] = <PtrT*><intptr_t>(obj[i])
    else:
        in_out_ptr.reset(<vector[PtrT*]*><intptr_t>obj, False)
    return 0


cdef int get_char_ptrs(nullable_unique_ptr[ vector[char*] ] &in_out_ptr, object obj) except 1:
    if cpython.PySequence_Check(obj):
        vec = new vector[char*](len(obj))
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            #__TODO__ is there a lifetime difference between this char* and some other ptrT*
            deref(vec)[i] = obj[i]
    else:
        in_out_ptr.reset(<vector[char*]*><intptr_t>obj, False)
    return 0


cdef int get_nested_resource_ptr(nested_resource[ResT] &in_out_ptr, object obj, ResT* __unused) except 1:
    cdef nullable_unique_ptr[ vector[intptr_t] ] nested_ptr
    cdef nullable_unique_ptr[ vector[vector[ResT]] ] nested_res_ptr
    cdef vector[intptr_t]* nested_vec = NULL
    cdef vector[vector[ResT]]* nested_res_vec = NULL
    cdef size_t i = 0, length = 0
    cdef intptr_t addr

    if is_nested_sequence(obj):
        length = len(obj)
        nested_res_vec = new vector[vector[ResT]](length)
        nested_vec = new vector[intptr_t](length)
        # set the ownership immediately to avoid leaking memory in case of
        # exception in the following loop
        nested_res_ptr.reset(nested_res_vec, True)
        nested_ptr.reset(nested_vec, True)
        for i, obj_i in enumerate(obj):
            deref(nested_res_vec)[i] = obj_i
            deref(nested_vec)[i] = <intptr_t>(deref(nested_res_vec)[i].data())
    elif cpython.PySequence_Check(obj):
        length = len(obj)
        nested_vec = new vector[intptr_t](length)
        nested_ptr.reset(nested_vec, True)
        for i, addr in enumerate(obj):
            deref(nested_vec)[i] = addr
        nested_res_ptr.reset(NULL, False)
    else:
        # obj is an int (ResT**)
        nested_res_ptr.reset(NULL, False)
        nested_ptr.reset(<vector[intptr_t]*><intptr_t>obj, False)

    in_out_ptr.ptrs = move(nested_ptr)
    in_out_ptr.nested_resource_ptr = move(nested_res_ptr)
    return 0


class FunctionNotFoundError(RuntimeError): pass

class NotSupportedError(RuntimeError): pass


cdef tuple get_nvjitlink_dso_version_suffix(int driver_ver):
    if 12000 <= driver_ver < 13000:
        return ('12', '')
    else:
        raise NotSupportedError('only CUDA 12 driver is supported')