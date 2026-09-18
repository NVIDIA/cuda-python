// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Filling the driver and compiler-library function tables from cuda-bindings.
//
// cuda-bindings loads each library and resolves its symbols once (for the
// driver, with cuGetProcAddress). cuda.bindings._internal.<lib>
// ._inspect_function_pointers() returns that table as {name: address}, where a
// zero address means the library does not provide the symbol. This file copies
// the entries cuda.core uses into the p_ pointers declared in driver_api.hpp.
//
// The fill runs Python, so it acquires the GIL and must never run under a C++
// lock (the GIL is the outermost lock; see DESIGN.md). The slot stores happen
// under fill_mutex with no Python call inside, and the ready flag is published
// with release semantics after them, so readers that see the flag see the
// pointers. Two threads may both compute the table; they store identical
// values, one after the other.
//
// Failures never propagate as exceptions and never leave a Python error set:
// they are recorded, reported through report_message(), and every affected
// DRIVER_CALL then returns an error status from a trampoline (driver_api.hpp).

#include "py.hpp"
#include "driver_api.hpp"
#include "error.hpp"
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace cuda_core::rt {

namespace {

constexpr std::size_t kTables = 4;
constexpr std::size_t kMaxEntries = 128;

std::atomic<bool> table_ready[kTables];
std::atomic<bool> unavailable_reported[kTables];
std::mutex fill_mutex;
char fill_error[kTables][512] = {};

std::size_t index_of(FnTable table) noexcept { return static_cast<std::size_t>(table); }

const char* module_name(FnTable table) noexcept {
    switch (table) {
        case FnTable::driver: return "cuda.bindings._internal.driver";
        case FnTable::nvrtc: return "cuda.bindings._internal.nvrtc";
        case FnTable::nvvm: return "cuda.bindings._internal.nvvm";
        case FnTable::nvjitlink: return "cuda.bindings._internal.nvjitlink";
    }
    return "cuda.bindings._internal";
}

const char* library_name(FnTable table) noexcept {
    switch (table) {
        case FnTable::driver: return "CUDA driver";
        case FnTable::nvrtc: return "NVRTC";
        case FnTable::nvvm: return "NVVM";
        case FnTable::nvjitlink: return "nvJitLink";
    }
    return "library";
}

// Copy the pending Python exception's text into buf and clear it.
void take_python_error(char* buf, std::size_t size) noexcept {
#if PY_VERSION_HEX >= 0x030C0000
    PyObject* exc = PyErr_GetRaisedException();
#else
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject* exc = value;
    Py_XDECREF(type);
    Py_XDECREF(traceback);
#endif
    PyObject* text = exc ? PyObject_Str(exc) : nullptr;
    const char* utf8 = text ? PyUnicode_AsUTF8(text) : nullptr;
    std::snprintf(buf, size, "%s", utf8 ? utf8 : "unknown error");
    Py_XDECREF(text);
    Py_XDECREF(exc);
    PyErr_Clear();
}

void record_failure(FnTable table, const char* message) noexcept {
    {
        std::lock_guard<std::mutex> lock(fill_mutex);
        std::snprintf(fill_error[index_of(table)], sizeof(fill_error[0]), "%s", message);
    }
    report_message(message);
}

}  // namespace

bool fn_table_ready(FnTable table) noexcept {
    return table_ready[index_of(table)].load(std::memory_order_acquire);
}

const char* fn_table_error(FnTable table) noexcept {
    const char* text = fill_error[index_of(table)];
    return text[0] ? text : nullptr;
}

bool ensure_fn_table(FnTable table) noexcept {
    const std::size_t idx = index_of(table);
    if (table_ready[idx].load(std::memory_order_acquire)) {
        return true;
    }
    std::size_t count = 0;
    const FnEntry* entries = fn_table_entries(table, &count);
    if (entries == nullptr || count > kMaxEntries) {
        record_failure(table, "internal cuda.core error, please report: function table has an unexpected size");
        return false;
    }
    if (!Py_IsInitialized() || py_is_finalizing()) {
        record_failure(table, "cuda.core cannot resolve driver functions while the interpreter is shutting down");
        return false;
    }

    char message[640];
    char cause[384];
    void* values[kMaxEntries] = {};

    GILAcquireGuard gil;
    if (!gil.acquired()) {
        record_failure(table, "cuda.core cannot resolve driver functions while the interpreter is shutting down");
        return false;
    }

    PyObject* module = PyImport_ImportModule(module_name(table));
    if (module == nullptr) {
        take_python_error(cause, sizeof(cause));
        std::snprintf(message, sizeof(message),
                      "cuda.core cannot import %s from the installed cuda-bindings: %s", module_name(table), cause);
        record_failure(table, message);
        return false;
    }
    PyObject* pointers = PyObject_CallMethod(module, "_inspect_function_pointers", nullptr);
    Py_DECREF(module);
    if (pointers == nullptr) {
        take_python_error(cause, sizeof(cause));
        std::snprintf(message, sizeof(message), "cuda-bindings could not load the %s: %s", library_name(table), cause);
        record_failure(table, message);
        return false;
    }
    if (!PyDict_Check(pointers)) {
        Py_DECREF(pointers);
        std::snprintf(message, sizeof(message),
                      "internal cuda.core error, please report: %s._inspect_function_pointers() did not return a dict",
                      module_name(table));
        record_failure(table, message);
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        PyObject* item = PyDict_GetItemString(pointers, entries[i].key);  // borrowed; no error on a miss
        if (item == nullptr) {
            Py_DECREF(pointers);
            std::snprintf(message, sizeof(message),
                          "the installed cuda-bindings has no entry for %s (%s). cuda.core was compiled against a "
                          "cuda.h that names this symbol differently than the cuda-bindings in use; install the "
                          "cuda-bindings this cuda.core requires",
                          entries[i].name, entries[i].key);
            record_failure(table, message);
            return false;
        }
        void* address = PyLong_AsVoidPtr(item);
        if (address == nullptr && PyErr_Occurred()) {
            Py_DECREF(pointers);
            take_python_error(cause, sizeof(cause));
            std::snprintf(message, sizeof(message),
                          "internal cuda.core error, please report: the entry for %s in %s is not an address: %s",
                          entries[i].key, module_name(table), cause);
            record_failure(table, message);
            return false;
        }
        values[i] = address;
    }
    Py_DECREF(pointers);

    // Every function introduced at or before the first release of the CUDA
    // major series is present in every driver cuda.core supports; a null one
    // means the driver is older than that. Newer functions may be null and are
    // gated on the driver version in Cython.
    if (table == FnTable::driver) {
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] == nullptr && entries[i].introduced <= CUDA_CORE_BUILD_MAJOR * 1000) {
                std::snprintf(message, sizeof(message),
                              "the installed CUDA driver does not provide %s, which every driver of the CUDA %d "
                              "series provides; this cuda.core build requires a CUDA %d driver",
                              entries[i].name, CUDA_CORE_BUILD_MAJOR, CUDA_CORE_BUILD_MAJOR);
                record_failure(table, message);
                return false;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(fill_mutex);
        if (!table_ready[idx].load(std::memory_order_relaxed)) {
            for (std::size_t i = 0; i < count; ++i) {
                *entries[i].slot = values[i];
            }
            fill_error[idx][0] = 0;
            table_ready[idx].store(true, std::memory_order_release);
        }
    }
    return true;
}

void report_unavailable_fn(FnTable table, const char* name) noexcept {
    // Once per table: the first unavailable call is the informative one.
    if (unavailable_reported[index_of(table)].exchange(true)) {
        return;
    }
    char message[768];
    if (const char* reason = fn_table_error(table)) {
        std::snprintf(message, sizeof(message), "cuda.core could not call %s: %s", name, reason);
    } else {
        std::snprintf(message, sizeof(message),
                      "internal cuda.core error, please report: %s was called but the installed %s does not "
                      "provide it; a feature gate is missing or wrong. The call returned an error instead.",
                      name, library_name(table));
    }
    report_message(message);
}

}  // namespace cuda_core::rt
