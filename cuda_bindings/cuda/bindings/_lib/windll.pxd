# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString

cdef extern from "windows.h":
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    ) nogil

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, const char* lpProcName) nogil

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags) nogil:
    cdef wchar_t* wpath
    with gil:
        wpath = PyUnicode_AsWideCharString(path, NULL)
    return <uintptr_t>_LoadLibraryExW(
        wpath,
        hFile,
        dwFlags
    )

cdef inline FARPROC GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)
