# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# WSL-specific locale guard, used by cuda.core.system.get_process_name() to
# work around a bug in NVML's WSL implementation where nvmlSystemGetProcessName
# returns mojibake when the calling thread is in a non-"C" locale. See
# get_process_name() for the full backstory.
#
# This module is only compiled on Linux (build_hooks.py excludes it on Windows)
# because it uses the POSIX per-thread locale APIs (newlocale/uselocale/
# freelocale), which are not available on MSVC. Callers must guard imports of
# this module with try/except ImportError.


cdef extern from "locale.h" nogil:
    ctypedef void *locale_t
    int LC_ALL_MASK
    locale_t newlocale(int category_mask, const char *locale, locale_t base)
    locale_t uselocale(locale_t newloc)
    void freelocale(locale_t locobj)


cdef class c_locale_guard:
    """Context manager that pins the calling thread to the "C" locale.

    Uses POSIX newlocale/uselocale/freelocale so other threads' view of the
    locale is unaffected. Restores the previous thread locale on exit.
    """
    cdef locale_t _c_locale
    cdef locale_t _prev_locale
    cdef bint _active

    def __cinit__(self) -> None:
        self._c_locale = <locale_t>0
        self._prev_locale = <locale_t>0
        self._active = False

    def __enter__(self):
        self._c_locale = newlocale(LC_ALL_MASK, b"C", <locale_t>0)
        if self._c_locale == <locale_t>0:
            raise RuntimeError("Failed to create C locale")
        self._prev_locale = uselocale(self._c_locale)
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            uselocale(self._prev_locale)
            freelocale(self._c_locale)
            self._active = False
        return False
