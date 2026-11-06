# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bridge CUDA driver error logs into Python's :mod:`logging` module via
the ``cuLogs*`` API.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Callable

from cuda.bindings import driver as _driver
from cuda.core._utils.cuda_utils import handle_return

__all__ = ["register_cuda_error_log", "unregister_cuda_error_log"]

# C callback signature for cuLogsRegisterCallback:
#     void callback(void *userData, CUlogLevel logLevel, char *message, size_t length)
_CUlogsCallback_functype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)
PyLogCallback = Callable[[int, bytes], None]

# Keep a reference while registered, or GC'ing it leaves the driver with
# a dangling function pointer which will crash on the next log message.
_lock = threading.Lock()
_c_callback: ctypes._FuncPointer | None = None
_callback_handle = None  # CUlogsCallbackHandle, needed to unregister


def _make_c_callback(logger: logging.Logger, callback: PyLogCallback | None) -> ctypes._FuncPointer:
    """Build the ctypes C callback passed to cuLogsRegisterCallback."""

    def _default_callback(cu_log_level: int, message: bytes) -> None:
        cuda_level_to_py_level = {
            int(_driver.CUlogLevel.CU_LOG_LEVEL_ERROR): logging.ERROR,
            int(_driver.CUlogLevel.CU_LOG_LEVEL_WARNING): logging.WARNING,
        }
        msg = message.decode("utf-8", errors="replace").rstrip("\n")
        py_level = cuda_level_to_py_level.get(cu_log_level, logging.ERROR)
        logger.log(py_level, "[CUDA Driver] %s", msg)

    emit = callback if callback is not None else _default_callback

    def _on_log(_user_data, log_level, message_ptr, length):
        try:
            message = ctypes.string_at(message_ptr, length) if message_ptr else b""
            emit(log_level, message)
        except Exception:
            logger.exception("cuda_error_log: error while forwarding driver log")

    return _CUlogsCallback_functype(_on_log)


def register_cuda_error_log(
    logger: logging.Logger | None = None,
    callback: PyLogCallback | None = None,
) -> logging.Logger:
    """Register a bridge that forwards CUDA driver ``cuLogs*`` messages to
    a Python :class:`logging.Logger`.

    Parameters
    ----------
    logger : logging.Logger, optional
        Target logger. If not provided, a default is created via
        ``logging.getLogger("cuda.driver")``.
    callback : Callable[[int, bytes], None], optional
        Overrides the default forwarding. Called as
        ``callback(cu_log_level, message)`` with undecoded ``message`` and
        unmapped ``cu_log_level``; decoding/mapping is the callback's job.

    Returns
    -------
    logging.Logger
        The ``logger`` argument above (or the default logger it resolved
        to). If a bridge was already registered, it is unregistered first,
        so this logger/callback replaces it.

    """
    global _c_callback, _callback_handle

    if logger is None:
        logger = logging.getLogger("cuda.driver")

    if callback is not None and not callable(callback):
        raise TypeError(f"callback must be callable or None, got {type(callback)!r}")

    with _lock:
        if _callback_handle is not None:
            # Already registered: unregister the old callback/handle first,
            # then fall through to register the new logger/callback below.
            handle_return(_driver.cuLogsUnregisterCallback(_callback_handle))
            _c_callback = None
            _callback_handle = None

        c_callback = _make_c_callback(logger, callback)
        addr = ctypes.cast(c_callback, ctypes.c_void_p).value

        handle = handle_return(_driver.cuLogsRegisterCallback(addr, None))

        # Keep the ctypes callback alive (see _c_callback comment above).
        _c_callback = c_callback
        _callback_handle = handle

    return logger


def unregister_cuda_error_log() -> None:
    """Unregister the bridge installed by :func:`register_cuda_error_log`."""
    global _c_callback, _callback_handle

    with _lock:
        if _callback_handle is None:
            return

        handle_return(_driver.cuLogsUnregisterCallback(_callback_handle))

        _c_callback = None
        _callback_handle = None
