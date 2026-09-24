// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// This file fills the driver and compiler-library function tables from cuda-bindings.
//
// cuda-bindings loads each library and resolves its symbols once. For the
// driver it uses cuGetProcAddress. cuda.bindings._internal.<lib>
// ._inspect_function_pointers() returns that table as {name: address}, where a
// zero address means the library does not provide the symbol. This file copies
// the entries cuda.core uses into the p_ pointers declared in driver_api.hpp.
//
// The fill runs Python, so it acquires the GIL. It must never run under a C++
// lock, because the GIL is the outermost lock (see DESIGN.md). The slot stores
// happen under fill_mutex with no Python call inside. The fill then publishes
// the ready flag with release semantics, so a reader that sees the flag sees
// the pointers. Two threads may both compute the table. They store identical
// values, one after the other.
//
// A failure never propagates as an exception and never leaves a Python error
// set. The fill records it, reports it through report_message(), and latches
// it: no later call retries the table, re-imports, or re-warns. Every
// affected DRIVER_CALL then returns an error status from a trampoline in
// driver_api.hpp, and the raised error carries the reason as a note.
// A fill can run while a Python exception propagates, for example when a
// deleter makes its first driver call during unwinding. The fill saves the
// pending exception before its Python calls and restores it afterwards.

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
std::atomic<bool> table_failed[kTables];
std::atomic<bool> unavailable_reported[kTables];
std::mutex fill_mutex;
char fill_error[kTables][512] = {};  // guarded by fill_mutex

// Saves the pending Python exception on construction and restores it on
// destruction. The Python calls in between start from a clean error state, and
// the caller's exception survives. Requires the GIL.
class PendingExceptionGuard {
public:
    PendingExceptionGuard() noexcept {
#if PY_VERSION_HEX >= 0x030C0000
        exc_ = PyErr_GetRaisedException();
#else
        PyErr_Fetch(&type_, &value_, &traceback_);
#endif
    }
    ~PendingExceptionGuard() {
        PyErr_Clear();  // drop anything the guarded calls left set
#if PY_VERSION_HEX >= 0x030C0000
        PyErr_SetRaisedException(exc_);
#else
        PyErr_Restore(type_, value_, traceback_);
#endif
    }
    PendingExceptionGuard(const PendingExceptionGuard&) = delete;
    PendingExceptionGuard& operator=(const PendingExceptionGuard&) = delete;

private:
#if PY_VERSION_HEX >= 0x030C0000
    PyObject* exc_ = nullptr;
#else
    PyObject* type_ = nullptr;
    PyObject* value_ = nullptr;
    PyObject* traceback_ = nullptr;
#endif
};

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

// Copies the pending Python exception's text into buf and clears it. Returns
// true when the exception says nothing about cuda-bindings or the driver: an
// interruption such as KeyboardInterrupt or SystemExit, or exhaustion such as
// MemoryError or RecursionError. The fill reports such a failure but does not
// latch it, so the next call tries again.
bool take_python_error(char* buf, std::size_t size) noexcept {
    const bool transient = PyErr_Occurred()
                           && (!PyErr_ExceptionMatches(PyExc_Exception) || PyErr_ExceptionMatches(PyExc_MemoryError)
                               || PyErr_ExceptionMatches(PyExc_RecursionError));
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
    return transient;
}

void record_failure(FnTable table, const char* message, bool latch = true) noexcept {
    {
        std::lock_guard<std::mutex> lock(fill_mutex);
        std::snprintf(fill_error[index_of(table)], sizeof(fill_error[0]), "%s", message);
    }
    if (latch) {
        table_failed[index_of(table)].store(true, std::memory_order_release);
    }
    report_message(message);
}

}  // namespace

bool fn_table_ready(FnTable table) noexcept {
    return table_ready[index_of(table)].load(std::memory_order_acquire);
}

bool fn_table_error(FnTable table, char* buffer, std::size_t size) noexcept {
    std::lock_guard<std::mutex> lock(fill_mutex);
    const char* text = fill_error[index_of(table)];
    if (!text[0]) {
        return false;
    }
    std::snprintf(buffer, size, "%s", text);
    return true;
}

bool ensure_fn_table(FnTable table) noexcept {
    const std::size_t idx = index_of(table);
    if (table_ready[idx].load(std::memory_order_acquire)) {
        return true;
    }
    if (table_failed[idx].load(std::memory_order_acquire)) {
        return false;  // latched: the fill reported its reason when it failed
    }
    std::size_t count = 0;
    const FnEntry* entries = fn_table_entries(table, &count);
    if (entries == nullptr || count > kMaxEntries) {
        record_failure(table, "internal cuda.core error, please report: function table has an unexpected size");
        return false;
    }
    if (!Py_IsInitialized() || py_is_finalizing()) {
        record_failure(table, "cuda.core cannot resolve driver functions during interpreter shutdown");
        return false;
    }

    char message[640];
    char cause[384];
    void* values[kMaxEntries] = {};

    GILAcquireGuard gil;
    if (!gil.acquired()) {
        record_failure(table, "cuda.core cannot resolve driver functions during interpreter shutdown");
        return false;
    }
    PendingExceptionGuard pending;

    PyObject* module = PyImport_ImportModule(module_name(table));
    if (module == nullptr) {
        const bool transient = take_python_error(cause, sizeof(cause));
        std::snprintf(message, sizeof(message),
                      "cuda.core cannot import %s from the installed cuda-bindings: %s", module_name(table), cause);
        record_failure(table, message, !transient);
        return false;
    }
    PyObject* pointers = PyObject_CallMethod(module, "_inspect_function_pointers", nullptr);
    Py_DECREF(module);
    if (pointers == nullptr) {
        const bool transient = take_python_error(cause, sizeof(cause));
        std::snprintf(message, sizeof(message), "cuda-bindings could not load the %s: %s", library_name(table), cause);
        record_failure(table, message, !transient);
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
                          "the installed cuda-bindings has no entry for %s (%s). This cuda.core build used a "
                          "cuda.h that names this symbol differently from the installed cuda-bindings. Install the "
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
    // major series is present in every driver cuda.core supports. A null one
    // means the driver is older than that. A newer function may be null, and
    // the Cython layer gates its use on the driver version.
    if (table == FnTable::driver) {
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] == nullptr && entries[i].introduced <= CUDA_CORE_BUILD_MAJOR * 1000) {
                std::snprintf(message, sizeof(message),
                              "the installed CUDA driver lacks %s, which every CUDA %d driver provides. "
                              "CUDA %d.%d introduced this function. This cuda.core build needs a newer driver",
                              entries[i].name, CUDA_CORE_BUILD_MAJOR, entries[i].introduced / 1000,
                              entries[i].introduced / 10 % 100);
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
    char reason[sizeof(fill_error[0])];
    const bool failed_fill = fn_table_error(table, reason, sizeof(reason));
    char message[768];
    if (failed_fill) {
        std::snprintf(message, sizeof(message), "cuda.core could not call %s: %s", name, reason);
    } else {
        std::snprintf(message, sizeof(message),
                      "internal cuda.core error, please report: cuda.core called %s but the installed %s does not "
                      "provide it. A feature gate is missing or wrong. The call returned an error instead.",
                      name, library_name(table));
    }
    if (table == FnTable::driver) {
        // The trampoline returns CUDA_ERROR_NOT_INITIALIZED. The Cython error
        // path attaches this message as a note to the CUDAError it raises.
        note_driver_table_failure(message);
    }
    if (failed_fill) {
        return;  // the fill reported its reason when it failed, and the note carries it to each raised error
    }
    // A gate bug: warn once per table. The first unavailable call is the informative one.
    if (unavailable_reported[index_of(table)].exchange(true)) {
        return;
    }
    report_message(message);
}

}  // namespace cuda_core::rt
