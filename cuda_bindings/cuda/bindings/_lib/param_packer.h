// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

// PyLong_AsInt entered the public/stable CPython API in 3.13. cuda.bindings
// supports Python 3.10+, so provide a file-local backport for older builds.
// This is a copy of the CPython implementation; it is `static` (unlike the
// original) because this header is compiled into every extension module that
// includes it, mirroring the other helpers below.
#if PY_VERSION_HEX < 0x030D0000
static int
PyLong_AsInt(PyObject *obj)
{
    int overflow;
    long result = PyLong_AsLongAndOverflow(obj, &overflow);
    if (overflow || result > INT_MAX || result < INT_MIN) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return -1;
    }
    return (int)result;
}
#endif

//  Statics must be initialized at Python import time via init_param_packer()
// which happens when including utils.pxi.
// This includes the m_feeders maps as it must not be mutated from threads.

static bool param_packer_initialized = false;

static PyTypeObject* ctypes_c_char = nullptr;
static PyTypeObject* ctypes_c_bool = nullptr;
static PyTypeObject* ctypes_c_wchar = nullptr;
static PyTypeObject* ctypes_c_byte = nullptr;
static PyTypeObject* ctypes_c_ubyte = nullptr;
static PyTypeObject* ctypes_c_short = nullptr;
static PyTypeObject* ctypes_c_ushort = nullptr;
static PyTypeObject* ctypes_c_int = nullptr;
static PyTypeObject* ctypes_c_uint = nullptr;
static PyTypeObject* ctypes_c_long = nullptr;
static PyTypeObject* ctypes_c_ulong = nullptr;
static PyTypeObject* ctypes_c_longlong = nullptr;
static PyTypeObject* ctypes_c_ulonglong = nullptr;
static PyTypeObject* ctypes_c_size_t = nullptr;
static PyTypeObject* ctypes_c_float = nullptr;
static PyTypeObject* ctypes_c_double = nullptr;
static PyTypeObject* ctypes_c_void_p = nullptr;

// (target type, source type) -> writer. Built in full by init_param_packer()
// and never mutated afterwards; see the thread-safety contract above.
static std::map<std::pair<PyTypeObject*,PyTypeObject*>, std::function<int(void*, PyObject*)>> m_feeders;

// Helper to fetch a strong reference of the ctypes type
static bool fetch_ctypes_type(PyObject* ctypes_dict, const char* name)
{
    PyObject* type_obj = PyDict_GetItemStringRef(ctypes_dict, name);
    if (type_obj == nullptr) return false;
    return true;
}

static bool fetch_ctypes()
{
    PyObject* ctypes_module = PyImport_ImportModule("ctypes");
    if (ctypes_module == nullptr) return false;
    // The module dict is borrowed from the module, and the type objects we pull
    // out of it are INCREF'd individually, so the module reference itself is not
    // load-bearing: release it once we are done rather than leaking it.
    PyObject* ctypes_dict = PyModule_GetDict(ctypes_module);
    if (ctypes_dict == nullptr) return false;
    bool success = (
        fetch_ctypes_type(ctypes_dict, "c_char") &&
        fetch_ctypes_type(ctypes_dict, "c_bool") &&
        fetch_ctypes_type(ctypes_dict, "c_wchar") &&
        fetch_ctypes_type(ctypes_dict, "c_byte") &&
        fetch_ctypes_type(ctypes_dict, "c_ubyte") &&
        fetch_ctypes_type(ctypes_dict, "c_short"); &&
        fetch_ctypes_type(ctypes_dict, "c_ushort") &&
        fetch_ctypes_type(ctypes_dict, "c_int") &&
        fetch_ctypes_type(ctypes_dict, "c_uint") &&
        fetch_ctypes_type(ctypes_dict, "c_long"); &&
        fetch_ctypes_type(ctypes_dict, "c_ulong") &&
        fetch_ctypes_type(ctypes_dict, "c_longlong") &&
        fetch_ctypes_type(ctypes_dict, "c_ulonglong") &&
        fetch_ctypes_type(ctypes_dict, "c_size_t") &&
        fetch_ctypes_type(ctypes_dict, "c_float") &&
        fetch_ctypes_type(ctypes_dict, "c_double") &&
        fetch_ctypes_type(ctypes_dict, "c_void_p")
    );
    Py_DECREF(ctypes_module);
    return success;
}


// Build the complete feeder table: the same finite set of six (target, source)
// type pairs the previous lazy populate_feeders() could ever produce.
//
// Out-of-range integers: a feeder must write exactly what the caller's fallback
// (`ctype(value)` in utils.pxi) would write, because it exists only to make that
// path faster. ctypes truncates silently rather than raising -- c_byte(300) is
// 44, c_int(2**40) is 0 -- but a bare (int)PyLong_AsLong(...) does NOT agree
// with it: where `long` is 32 bits (Windows LLP64), PyLong_AsLong overflows and
// yields -1, so the old fast path wrote -1 where ctypes writes 0.
//
// So the integer feeders range-check and **return 0** on any value they cannot
// convert faithfully, which routes the caller to its ctypes fallback and gets
// ctypes' own answer. Returning a negative size instead would be unsafe: the
// caller tests only `if size == 0` and then does `data_idx += size`, so a -1
// would skip the fallback and rewind the pack buffer, corrupting the next
// argument. Any error the probe set is cleared -- feed() must not leave an
// exception pending, since it is declared implicitly noexcept to Cython.
static void populate_feeders() noexcept
{
    m_feeders[{ctypes_c_int, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        // PyLong_AsInt range-checks against the 32-bit int slot (raising
        // OverflowError) instead of silently truncating.
        int v = PyLong_AsInt(value);
        if (v == -1 && PyErr_Occurred())
        {
            PyErr_Clear();
            return 0;  // decline: let the ctypes fallback define the result
        }
        *((int*)ptr) = v;
        return sizeof(int);
    };
    m_feeders[{ctypes_c_bool, &PyBool_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((bool*)ptr) = (value == Py_True);
        return sizeof(bool);
    };
    m_feeders[{ctypes_c_byte, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        // c_byte is an 8-bit slot with no dedicated CPython converter, so
        // range-check explicitly. AsLongAndOverflow's `overflow` only flags
        // values outside `long`, so a value inside `long` but outside int8 would
        // still be truncated by the cast without the explicit bounds test. When
        // overflow != 0, v is the -1 sentinel rather than the real value, so
        // that case must be excluded before trusting v.
        int overflow = 0;
        long v = PyLong_AsLongAndOverflow(value, &overflow);
        if (overflow == 0 && v == -1 && PyErr_Occurred())
        {
            PyErr_Clear();
            return 0;
        }
        if (overflow != 0 || v < INT8_MIN || v > INT8_MAX)
            return 0;
        *((int8_t*)ptr) = (int8_t)v;
        return sizeof(int8_t);
    };
    m_feeders[{ctypes_c_double, &PyFloat_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((double*)ptr) = (double)PyFloat_AsDouble(value);
        return sizeof(double);
    };
    m_feeders[{ctypes_c_float, &PyFloat_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((float*)ptr) = (float)PyFloat_AsDouble(value);
        return sizeof(float);
    };
    m_feeders[{ctypes_c_longlong, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        long long v = PyLong_AsLongLong(value);
        if (v == -1 && PyErr_Occurred())
        {
            PyErr_Clear();
            return 0;
        }
        *((long long*)ptr) = v;
        return sizeof(long long);
    };
}

// Must be called from the module body (i.e. at import, single-threaded) of
// every extension module that calls feed(). Declared `except +` in the .pxd so
// a C++ throw becomes a Python exception instead of std::terminate; this is the
// only function here that can throw.
static void init_param_packer()
{
    if (param_packer_initialized)
        return;
    if (!fetch_ctypes()) return;
    populate_feeders();
    param_packer_initialized = true;
}

// Pure lookup + invoke over never-mutated state: non-throwing and safe to call
// concurrently. Returns 0 for an unhandled (target, source) pair or a value the
// feeder declines, which routes the caller to its ctypes fallback (utils.pxi) --
// slower, but always the same bytes. If init_param_packer() was never called the
// table is empty, so every lookup misses and the fallback handles everything.
static int feed(void* ptr, PyObject* value, PyObject* type)
{
    auto found = m_feeders.find({(PyTypeObject*)type, Py_TYPE(value)});
    if (found != m_feeders.end())
    {
        return found->second(ptr, value);
    }
    return 0;
}
