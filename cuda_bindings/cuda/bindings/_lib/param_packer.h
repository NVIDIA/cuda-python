// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <map>
#include <functional>
#include <climits>
#include <cstdint>

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

// Statics must be initialized at Python import time via init_param_packer()
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

// (target type, source type)
static std::map<std::pair<PyTypeObject*,PyTypeObject*>, std::function<int(void*, PyObject*)>> m_feeders;

// Helper to fetch a strong reference of the ctypes type.
static PyTypeObject* fetch_ctypes_type(PyObject* ctypes_module, const char* name)
{
    return (PyTypeObject*)PyObject_GetAttrString(ctypes_module, name);
}

static bool fetch_ctypes()
{
    PyObject* ctypes_module = PyImport_ImportModule("ctypes");
    if (ctypes_module == nullptr) return false;
    // Parenthesize each assignment: `=` binds looser than `&&`.
    bool success = (
        (ctypes_c_char = fetch_ctypes_type(ctypes_module, "c_char")) &&
        (ctypes_c_bool = fetch_ctypes_type(ctypes_module, "c_bool")) &&
        (ctypes_c_wchar = fetch_ctypes_type(ctypes_module, "c_wchar")) &&
        (ctypes_c_byte = fetch_ctypes_type(ctypes_module, "c_byte")) &&
        (ctypes_c_ubyte = fetch_ctypes_type(ctypes_module, "c_ubyte")) &&
        (ctypes_c_short = fetch_ctypes_type(ctypes_module, "c_short")) &&
        (ctypes_c_ushort = fetch_ctypes_type(ctypes_module, "c_ushort")) &&
        (ctypes_c_int = fetch_ctypes_type(ctypes_module, "c_int")) &&
        (ctypes_c_uint = fetch_ctypes_type(ctypes_module, "c_uint")) &&
        (ctypes_c_long = fetch_ctypes_type(ctypes_module, "c_long")) &&
        (ctypes_c_ulong = fetch_ctypes_type(ctypes_module, "c_ulong")) &&
        (ctypes_c_longlong = fetch_ctypes_type(ctypes_module, "c_longlong")) &&
        (ctypes_c_ulonglong = fetch_ctypes_type(ctypes_module, "c_ulonglong")) &&
        (ctypes_c_size_t = fetch_ctypes_type(ctypes_module, "c_size_t")) &&
        (ctypes_c_float = fetch_ctypes_type(ctypes_module, "c_float")) &&
        (ctypes_c_double = fetch_ctypes_type(ctypes_module, "c_double")) &&
        (ctypes_c_void_p = fetch_ctypes_type(ctypes_module, "c_void_p"))  // == c_voidp
    );
    Py_DECREF(ctypes_module);
    return success;
}


// Initialize common (target_type, Python type) pairs for fast argument feeding.
static void populate_feeders()
{
    m_feeders[{ctypes_c_int, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        // PyLong_AsInt range-checks against the 32-bit int slot and raises
        // OverflowError itself, so an out-of-range value is rejected rather
        // than silently truncated.
        int v = PyLong_AsInt(value);
        if (v == -1 && PyErr_Occurred())
            return -1;
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
        // range-check explicitly against INT8_MIN/INT8_MAX. AsLongAndOverflow's
        // `overflow` only flags values outside `long` (64-bit on LP64), so a
        // value in that range would be silently truncated by (int8_t)v without
        // the explicit bounds check. When overflow!=0, v is the -1 sentinel
        // (not the real value), so that case must be caught before trusting v.
        int overflow = 0;
        long v = PyLong_AsLongAndOverflow(value, &overflow);
        if (overflow == 0 && v == -1 && PyErr_Occurred())
            return -1;  // non-overflow conversion error; exception already set
        if (overflow != 0 || v < INT8_MIN || v > INT8_MAX)
        {
            PyErr_SetString(PyExc_OverflowError,
                "Python int is out of range for a c_byte (8-bit) kernel argument");
            return -1;
        }
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
            return -1;
        *((long long*)ptr) = v;
        return sizeof(long long);
    };
}

// Call once from each consuming module body (import, single-threaded).
static void init_param_packer()
{
    if (param_packer_initialized)
        return;
    if (!fetch_ctypes()) return;
    populate_feeders();
    param_packer_initialized = true;
}

// Never-mutated lookup. 0 -> ctypes fallback; -1 -> exception already set.
static int feed(void* ptr, PyObject* value, PyObject* type)
{
    auto found = m_feeders.find({(PyTypeObject*)type, Py_TYPE(value)});
    if (found != m_feeders.end())
    {
        return found->second(ptr, value);
    }
    return 0;
}
