// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <map>
#include <functional>
#include <stdexcept>
#include <string>

// Strong reference to the ctypes module, acquired once at import in
// init_param_packer() and intentionally never released. This ref is
// load-bearing: the ctypes_c_* pointers below are *borrowed* references into
// the ctypes module dict, and stay valid only while this strong ref keeps the
// module (and therefore its type objects) alive. Do not Py_DECREF it.
static PyObject* ctypes_module = nullptr;

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

// Import ctypes and cache pointers to its scalar type objects. May throw
// std::runtime_error (translated to a Python exception via the `except +`
// declaration on init_param_packer in param_packer.pxd). Called exactly once,
// from init_param_packer() at module import while single-threaded.
static void fetch_ctypes()
{
    ctypes_module = PyImport_ImportModule("ctypes");
    if (ctypes_module == nullptr)
        throw std::runtime_error("Cannot import ctypes module");
    // get method addressof
    PyObject* ctypes_dict = PyModule_GetDict(ctypes_module);
    if (ctypes_dict == nullptr)
        throw std::runtime_error(std::string("FAILURE @ ") + std::string(__FILE__) + " : " + std::to_string(__LINE__));
    // supportedtypes
    ctypes_c_char = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_char");
    ctypes_c_bool = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_bool");
    ctypes_c_wchar = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_wchar");
    ctypes_c_byte = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_byte");
    ctypes_c_ubyte = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_ubyte");
    ctypes_c_short = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_short");
    ctypes_c_ushort = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_ushort");
    ctypes_c_int = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_int");
    ctypes_c_uint = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_uint");
    ctypes_c_long = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_long");
    ctypes_c_ulong = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_ulong");
    ctypes_c_longlong = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_longlong");
    ctypes_c_ulonglong = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_ulonglong");
    ctypes_c_size_t = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_size_t");
    ctypes_c_float = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_float");
    ctypes_c_double = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_double");
    ctypes_c_void_p = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_void_p"); // == c_voidp
}


// (target type, source type)
static std::map<std::pair<PyTypeObject*,PyTypeObject*>, std::function<int(void*, PyObject*)>> m_feeders;

static void populate_feeders(PyTypeObject* target_t, PyTypeObject* source_t)
{
    if (target_t == ctypes_c_int)
    {
        if (source_t == &PyLong_Type)
        {
            m_feeders[{target_t,source_t}] = [](void* ptr, PyObject* value) -> int
            {
                *((int*)ptr) = (int)PyLong_AsLong(value);
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
                *((int8_t*)ptr) = (int8_t)PyLong_AsLong(value);
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

// Initialize all shared state once, at module import, while single-threaded
// (called as a bare module-level statement from utils.pxi -- mirroring the
// _resource_handles.pyx pattern of calling initialize_deferred_cleanup() at
// import). This fetches the ctypes type pointers and pre-builds the *entire*
// feeder table, so the hot feed() path below is afterwards a pure read of
// never-mutated global state. That is what makes feed() safe to call
// concurrently from multiple threads under free-threading (Py_MOD_GIL_NOT_USED)
// without any lock: doing the work lazily inside feed() would race the map and
// the ctypes lazy-init across threads launching distinct kernels.
//
// May throw (e.g. if `import ctypes` fails); declared `except +` in the pxd so
// the failure surfaces as a Python exception during module import rather than
// std::terminate.
static void init_param_packer()
{
    if (ctypes_module != nullptr)
        return;  // defensive: module import already runs exactly once
    fetch_ctypes();
    // Pre-build every feeder the old lazy path could ever have created. This is
    // a fixed, finite set of (target, source) type pairs, so a table built here
    // is identical to the one feed() used to build on demand; any unmatched
    // pair still falls through to feed() returning 0 -> the ctype() fallback in
    // utils.pxi. After this, m_feeders is never mutated again.
    populate_feeders(ctypes_c_int,      &PyLong_Type);
    populate_feeders(ctypes_c_bool,     &PyBool_Type);
    populate_feeders(ctypes_c_byte,     &PyLong_Type);
    populate_feeders(ctypes_c_double,   &PyFloat_Type);
    populate_feeders(ctypes_c_float,    &PyFloat_Type);
    populate_feeders(ctypes_c_longlong, &PyLong_Type);
}

// Hot path. Read-only lookup on the import-time-populated feeder table; no
// mutation, no allocation, no throwing C-API translation -> effectively
// noexcept and safe under concurrent, GIL-free calls. Returns 0 for any
// unhandled (target, source) pair so the caller applies its ctype() fallback.
static int feed(void* ptr, PyObject* value, PyObject* type)
{
    PyTypeObject* pto = (PyTypeObject*)type;
    auto found = m_feeders.find({pto,value->ob_type});
    if (found != m_feeders.end())
    {
        return found->second(ptr, value);
    }
    return 0;
}
