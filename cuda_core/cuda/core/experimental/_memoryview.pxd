from libc.stdint cimport uintptr_t


cdef class _MDSPAN:
    cdef:
        # this must be a pointer to a host mdspan object
        readonly uintptr_t _ptr
        # if the host mdspan is exported from any Python object,
        # we need to keep a reference to that object alive
        readonly object _exporting_obj
