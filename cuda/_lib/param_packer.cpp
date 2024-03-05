// Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
#include <Python.h>
#include "param_packer.h"

#include <map>
#include <functional>
#include <stdexcept>
#include <string>

PyObject* enum_module = nullptr;
PyTypeObject* enum_Enum = nullptr;

PyObject* ctypes_module = nullptr;
PyObject* ctypes_addressof = nullptr;
PyObject* addressof_param_tuple = nullptr;

PyTypeObject* ctypes_c_char = nullptr;
PyTypeObject* ctypes_c_bool = nullptr;
PyTypeObject* ctypes_c_wchar = nullptr;
PyTypeObject* ctypes_c_byte = nullptr;
PyTypeObject* ctypes_c_ubyte = nullptr;
PyTypeObject* ctypes_c_short = nullptr;
PyTypeObject* ctypes_c_ushort = nullptr;
PyTypeObject* ctypes_c_int = nullptr;
PyTypeObject* ctypes_c_uint = nullptr;
PyTypeObject* ctypes_c_long = nullptr;
PyTypeObject* ctypes_c_ulong = nullptr;
PyTypeObject* ctypes_c_longlong = nullptr;
PyTypeObject* ctypes_c_ulonglong = nullptr;
PyTypeObject* ctypes_c_size_t = nullptr;
PyTypeObject* ctypes_c_float = nullptr;
PyTypeObject* ctypes_c_double = nullptr;
PyTypeObject* ctypes_c_void_p = nullptr;

PyTypeObject* ctypes_c_ssize_t = nullptr;
PyTypeObject* ctypes_c_longdouble = nullptr;
PyTypeObject* ctypes_c_char_p = nullptr;
PyTypeObject* ctypes_c_wchar_p = nullptr;
PyTypeObject* ctypes_c_structure = nullptr;

void fetch_ctypes()
{
    ctypes_module = PyImport_ImportModule("ctypes");
    if (ctypes_module == nullptr)
        throw std::runtime_error("Cannot import ctypes module");
    // get method addressof
    PyObject* ctypes_dict = PyModule_GetDict(ctypes_module);
    if (ctypes_dict == nullptr)
        throw std::runtime_error(std::string("FAILURE @ ") + std::string(__FILE__) + " : " + std::to_string(__LINE__));
    // supportedtypes
    ctypes_c_int = (PyTypeObject*) PyDict_GetItemString(ctypes_dict, "c_int");
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
std::map<std::pair<PyTypeObject*,PyTypeObject*>, std::function<int(void*, PyObject*)>> m_feeders;

void populate_feeders(PyTypeObject* target_t, PyTypeObject* source_t)
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

int feed(void* ptr, PyObject* value, PyObject* type)
{
    PyTypeObject* pto = (PyTypeObject*)type;
    if (ctypes_c_int == nullptr)
        fetch_ctypes();
    auto found = m_feeders.find({pto,value->ob_type});
    if (found == m_feeders.end())
    {
        populate_feeders(pto, value->ob_type);
        found = m_feeders.find({pto,value->ob_type});
    }
    if (found != m_feeders.end())
    {
        return found->second(ptr, value);
    }
    return 0;
}
