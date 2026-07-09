# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only weak handles for resource-handle lifetime checks.

This module is **not** part of the public ``cuda.core`` API. It is built into
the package (like other private ``_utils`` modules) purely so the test suite can
observe, deterministically, when the strong references that keep a CUDA resource
alive have all been released -- without relying on driver- or hardware-specific
side effects (for example, whether freed device memory happens to remain
readable).

Every resource handle is owned by a C++ ``std::shared_ptr``. A **weak handle**
is a non-owning ``std::weak_ptr`` observer of that control block: truthy while
some strong owner remains, falsy once the last one is gone. Use :func:`weak_handle`
to obtain a weak handle from a supported front-end object.

To support another type, add a ``cdef _weak_from_<type>`` that reads its ``cdef``
handle field (see ``*.pxd``), assigns to :ctype:`OpaqueHandle`, and extend the
``isinstance`` chain in :func:`weak_handle`. Types whose slots hold arbitrary
Python owners via ``make_opaque_py`` are not covered here -- use
:class:`weakref.ref` on a weak-referenceable owner object in tests instead.
"""

from cuda.core._memory._buffer cimport Buffer
from cuda.core._resource_handles cimport OpaqueHandle


# Cython cannot spell ``weak_ptr[const void]`` inline (the ``const void``
# template argument fails to parse), so the weak type and its one constructor
# are provided by a small inline C++ shim local to this test-only module. This
# keeps the production resource_handles translation units untouched.
cdef extern from *:
    """
    #include <memory>
    namespace cuda_core_test {
    using OpaqueWeakHandle = std::weak_ptr<const void>;
    static inline OpaqueWeakHandle make_weak(const std::shared_ptr<const void>& h) {
        return OpaqueWeakHandle(h);
    }
    }  // namespace cuda_core_test
    """
    cppclass OpaqueWeakHandle "cuda_core_test::OpaqueWeakHandle":
        OpaqueWeakHandle()
        bint expired()
        long use_count()
    OpaqueWeakHandle make_weak "cuda_core_test::make_weak" (const OpaqueHandle& h)


cdef class WeakHandle:
    """Non-owning weak handle for a resource's shared control block.

    Truthy while some strong owner of the underlying resource handle remains,
    falsy once the last strong reference is released. Obtain instances via
    :func:`weak_handle` rather than constructing directly.
    """

    cdef OpaqueWeakHandle _w

    def __bool__(self):
        return not self._w.expired()

    def expired(self):
        """Return ``True`` once every strong owner of the handle is gone."""
        return self._w.expired()

    def use_count(self):
        """Number of strong owners currently sharing the handle."""
        return self._w.use_count()


cdef WeakHandle _weak_from_opaque(OpaqueHandle h):
    # Build the weak handle from a (temporary) strong handle. The strong copy
    # lives only for the duration of this call, so it does not perturb the
    # reference count the weak handle later reports.
    cdef WeakHandle wh = WeakHandle.__new__(WeakHandle)
    wh._w = make_weak(h)
    return wh


cdef WeakHandle _weak_from_buffer(Buffer buf):
    cdef OpaqueHandle h = buf._h_ptr
    if not h:
        raise ValueError("Buffer has no active allocation")
    return _weak_from_opaque(h)


def weak_handle(obj):
    """Return a :class:`WeakHandle` observing the resource behind ``obj``.

    Currently supports :class:`~cuda.core.Buffer` (device allocation handle).
    See the module docstring for how to add more types.

    Raises
    ------
    ValueError
        If ``obj`` is a :class:`~cuda.core.Buffer` with no active allocation.
    TypeError
        If ``obj`` is not a supported type.
    """
    if isinstance(obj, Buffer):
        return _weak_from_buffer(obj)
    raise TypeError(
        f"weak_handle() does not support {type(obj).__name__!r}; "
        "supported types: Buffer"
    )
