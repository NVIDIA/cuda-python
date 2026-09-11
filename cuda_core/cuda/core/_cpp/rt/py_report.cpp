// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "error.hpp"
#include "internal.hpp"
#include <atomic>
#include <cstdio>

namespace cuda_core::rt {

using namespace detail;

namespace {
// Warning category registered by _utils/cuda_utils.pyx (cuda.core.CUDAWarning).
std::atomic<PyObject*> warning_category{nullptr};
}  // namespace

// Report a message that could not be raised. Emits cuda.core.CUDAWarning via
// the Python warnings machinery; if that itself fails (for example because the
// warning was promoted to an error), the failure is written as an unraisable
// exception, the CPython convention for exceptions in destructors. Falls back
// to stderr when the interpreter cannot be used.
void report_message(const char* message) noexcept {
    PyObject* category = warning_category.load(std::memory_order_acquire);
    if (category && Py_IsInitialized() && !py_is_finalizing()) {
        GILAcquireGuard gil;
        if (gil.acquired()) {
            // Deleters can run while a Python exception is propagating; keep it.
#if PY_VERSION_HEX >= 0x030C0000
            PyObject* pending = PyErr_GetRaisedException();
#else
            PyObject *pending_type, *pending_value, *pending_tb;
            PyErr_Fetch(&pending_type, &pending_value, &pending_tb);
#endif
            if (PyErr_WarnEx(category, message, 1) != 0) {
                PyObject* subject = PyUnicode_FromString(message);
                PyErr_WriteUnraisable(subject);
                Py_XDECREF(subject);
            }
#if PY_VERSION_HEX >= 0x030C0000
            PyErr_SetRaisedException(pending);
#else
            PyErr_Restore(pending_type, pending_value, pending_tb);
#endif
            return;
        }
    }
    std::fprintf(stderr, "%s\n", message);
    std::fflush(stderr);
}

void register_warning_category(PyObject* category) noexcept {
    warning_category.store(category, std::memory_order_release);
}

namespace {
// Attach `message` as a PEP 678 note to the exception currently being handled.
// Returns false when there is none or the interpreter cannot be used.
bool add_note_to_handled_exception(const char* message) noexcept {
#if PY_VERSION_HEX >= 0x030B0000
    if (!Py_IsInitialized() || py_is_finalizing()) {
        return false;
    }
    GILAcquireGuard gil;
    if (!gil.acquired()) {
        return false;
    }
    PyObject* exc = PyErr_GetHandledException();
    if (!exc) {
        return false;
    }
    PyObject* result = PyObject_CallMethod(exc, "add_note", "s", message);
    Py_DECREF(exc);
    if (!result) {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(result);
    return true;
#else
    (void)message;
    return false;
#endif
}
}  // namespace

void note_or_report_cuda_error(const char* operation, CUresult status, const char* detail) noexcept {
    if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED) {
        return;
    }
    char message[512];
    format_cuda_error(message, sizeof(message), operation, status, detail);
    if (!add_note_to_handled_exception(message)) {
        report_message(message);
    }
}

}  // namespace cuda_core::rt
