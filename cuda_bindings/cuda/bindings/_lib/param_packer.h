// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
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

// Thread-safety contract (free-threaded builds declare Py_MOD_GIL_NOT_USED, so
// there is no GIL serializing the calls below):
//
// Every static in this header is written exactly once, by init_param_packer(),
// while the importing thread is still the only thread that can reach it --
// Python's import machinery guarantees a module body runs to completion before
// any other thread can call into the module. Afterwards the state is read-only,
// which is what makes the hot feed() path safe to call concurrently without a
// lock: concurrent std::map::find() on a map that is never mutated is safe by
// the C++ standard.
//
// Do not reintroduce lazy initialization here. This header is textually
// compiled into every extension module that cimports param_packer.pxd, so each
// such module owns an independent copy of this state and would need its own
// racy lazy path.

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

// Look a type up in the ctypes module dict and return a *strong* reference.
// PyDict_GetItemString returns a borrowed reference; we upgrade it to a strong
// one so the cached PyTypeObject* stays valid for the process lifetime even if
// the ctypes module itself is later dropped from sys.modules.
static PyTypeObject* fetch_ctypes_type(PyObject* ctypes_dict, const char* name)
{
    if (target_t == ctypes_c_int)
    {
        if (source_t == &PyLong_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
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
            return;
        }
    } else if (target_t == ctypes_c_bool) {
        if (source_t == &PyBool_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
            {
                *((bool*)ptr) = (value == Py_True);
                return sizeof(bool);
            };
            return;
        }
    } else if (target_t == ctypes_c_byte) {
        if (source_t == &PyLong_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
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
            return;
        }
    } else if (target_t == ctypes_c_double) {
        if (source_t == &PyFloat_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
            {
                *((double*)ptr) = (double)PyFloat_AsDouble(value);
                return sizeof(double);
            };
            return;
        }
    } else if (target_t == ctypes_c_float) {
        if (source_t == &PyFloat_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
            {
                *((float*)ptr) = (float)PyFloat_AsDouble(value);
                return sizeof(float);
            };
            return;
        }
    } else if (target_t == ctypes_c_longlong) {
        if (source_t == &PyLong_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
            {
                *((long long*)ptr) = (long long)PyLong_AsLongLong(value);
                return sizeof(long long);
            };
            return;
        }
    }
}

static void fetch_ctypes()
{
    PyObject* ctypes_module = PyImport_ImportModule("ctypes");
    if (ctypes_module == nullptr)
        throw std::runtime_error("Cannot import ctypes module");
    // The module dict is borrowed from the module, and the type objects we pull
    // out of it are INCREF'd individually, so the module reference itself is not
    // load-bearing: release it once we are done rather than leaking it.
    try
    {
        PyObject* ctypes_dict = PyModule_GetDict(ctypes_module);  // borrowed
        if (ctypes_dict == nullptr)
            throw std::runtime_error(std::string("FAILURE @ ") + std::string(__FILE__) + " : " + std::to_string(__LINE__));
        // supported types
        ctypes_c_char = fetch_ctypes_type(ctypes_dict, "c_char");
        ctypes_c_bool = fetch_ctypes_type(ctypes_dict, "c_bool");
        ctypes_c_wchar = fetch_ctypes_type(ctypes_dict, "c_wchar");
        ctypes_c_byte = fetch_ctypes_type(ctypes_dict, "c_byte");
        ctypes_c_ubyte = fetch_ctypes_type(ctypes_dict, "c_ubyte");
        ctypes_c_short = fetch_ctypes_type(ctypes_dict, "c_short");
        ctypes_c_ushort = fetch_ctypes_type(ctypes_dict, "c_ushort");
        ctypes_c_int = fetch_ctypes_type(ctypes_dict, "c_int");
        ctypes_c_uint = fetch_ctypes_type(ctypes_dict, "c_uint");
        ctypes_c_long = fetch_ctypes_type(ctypes_dict, "c_long");
        ctypes_c_ulong = fetch_ctypes_type(ctypes_dict, "c_ulong");
        ctypes_c_longlong = fetch_ctypes_type(ctypes_dict, "c_longlong");
        ctypes_c_ulonglong = fetch_ctypes_type(ctypes_dict, "c_ulonglong");
        ctypes_c_size_t = fetch_ctypes_type(ctypes_dict, "c_size_t");
        ctypes_c_float = fetch_ctypes_type(ctypes_dict, "c_float");
        ctypes_c_double = fetch_ctypes_type(ctypes_dict, "c_double");
        ctypes_c_void_p = fetch_ctypes_type(ctypes_dict, "c_void_p"); // == c_voidp
    }
    catch (...)
    {
        Py_DECREF(ctypes_module);
        throw;
    }
    Py_DECREF(ctypes_module);
}


// Build the complete feeder table. This is the same finite set of six
// (target type, source type) pairs the previous lazy populate_feeders() could
// ever produce, so building it eagerly is behavior-neutral.
static void populate_feeders()
{
    m_feeders[{ctypes_c_int, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((int*)ptr) = (int)PyLong_AsLong(value);
        return sizeof(int);
    };
    m_feeders[{ctypes_c_bool, &PyBool_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((bool*)ptr) = (value == Py_True);
        return sizeof(bool);
    };
    m_feeders[{ctypes_c_byte, &PyLong_Type}] = [](void* ptr, PyObject* value) -> int
    {
        *((int8_t*)ptr) = (int8_t)PyLong_AsLong(value);
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
        *((long long*)ptr) = (long long)PyLong_AsLongLong(value);
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
    fetch_ctypes();
    populate_feeders();
    param_packer_initialized = true;
}

// Pure lookup + invoke over never-mutated state: non-throwing and safe to call
// concurrently. If init_param_packer() was never called the table is empty, so
// every lookup misses and returns 0, which routes the caller to its ctypes
// fallback (utils.pxi) -- slower, but still correct.
static int feed(void* ptr, PyObject* value, PyObject* type)
{
    auto found = m_feeders.find({(PyTypeObject*)type, Py_TYPE(value)});
    if (found != m_feeders.end())
    {
        return found->second(ptr, value);
    }
    return 0;
}
