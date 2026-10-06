# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for bridging CUDA driver ``cuLogs*`` messages into Python
``logging`` (see design doc / GitHub issue #671).
"""

import logging

import pytest

from cuda.bindings import driver
from cuda.core.utils import register_cuda_error_log, unregister_cuda_error_log


@pytest.fixture(autouse=True)
def teardown_bridge():
    yield
    # Unregister after every test so a leaked bridge can't leak into others.
    unregister_cuda_error_log()


# Usage 1: show driver error log interactively
def test_cuda_driver_log_captured_via_caplog(caplog):
    driver.cuInit(0)

    logger_name = "cuda.driver.caplog_test"
    logger_obj = logging.getLogger(logger_name)
    register_cuda_error_log(logger_obj)

    # cuDeviceGet(9999) is out of range; driver writes two ERROR logs
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)

    messages = [r.message for r in caplog.records if r.name == logger_name]

    assert len(messages) == 2, f"Expected exactly 2 driver log messages, got:\n{messages}"
    assert "Parameter ordinal must be between 0 and 1" in messages[0]
    assert "CUDA_ERROR_INVALID_DEVICE" in messages[1] and "cuDeviceGet" in messages[1]

    # After unregistering, driver logs are no longer forwarded
    unregister_cuda_error_log()
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records)
    assert "Parameter ordinal must be between 0 and 1" not in caplog.text


# Usage 2: redirect driver log to a file
def test_cuda_driver_log_written_to_file(tmp_path):
    driver.cuInit(0)

    log_file = tmp_path / "cuda_driver.log"
    logger_name = "cuda.driver.file_test"
    logger_obj = logging.getLogger(logger_name)
    logger_obj.setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger_obj.addHandler(file_handler)

    try:
        register_cuda_error_log(logger_obj)
        driver.cuDeviceGet(9999)
    finally:
        logger_obj.removeHandler(file_handler)
        file_handler.close()

    content = log_file.read_text(encoding="utf-8")

    assert "Parameter ordinal must be between 0 and 1" in content
    assert "CUDA_ERROR_INVALID_DEVICE" in content and "cuDeviceGet" in content


# Usage 3: register -> unregister -> rebind to another logger
def test_cuda_driver_log_rebind_logger(caplog):
    driver.cuInit(0)

    name_a, name_b = "cuda.driver.rebind_a", "cuda.driver.rebind_b"

    register_cuda_error_log(logging.getLogger(name_a))
    with caplog.at_level(logging.ERROR, logger=name_a):
        driver.cuDeviceGet(9999)
    assert any(r.name == name_a for r in caplog.records)

    # Must unregister first to free the driver-side slot, then bind the new logger
    unregister_cuda_error_log()
    caplog.clear()
    register_cuda_error_log(logging.getLogger(name_b))
    with caplog.at_level(logging.ERROR, logger=name_b):
        driver.cuDeviceGet(9999)

    assert not any(r.name == name_a for r in caplog.records)
    assert any(r.name == name_b for r in caplog.records)


# Usage 4: the logger's level threshold applies to driver logs too
def test_cuda_driver_log_level_filtering(caplog):
    driver.cuInit(0)

    logger_name = "cuda.driver.level_test"
    register_cuda_error_log(logging.getLogger(logger_name))

    # CRITICAL threshold filters out ERROR-level driver logs
    with caplog.at_level(logging.CRITICAL, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records)

    # Lowering to ERROR threshold lets them through
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert any(r.name == logger_name for r in caplog.records)


# Usage 5: custom Handler counts errors, for monitoring/alerting integration
def test_cuda_driver_log_alert_counter():
    driver.cuInit(0)

    class ErrorCounterHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.ERROR)
            self.count = 0

        def emit(self, record):
            self.count += 1

    logger_obj = logging.getLogger("cuda.driver.alert_test")
    counter_handler = ErrorCounterHandler()
    logger_obj.addHandler(counter_handler)

    try:
        register_cuda_error_log(logger_obj)
        driver.cuDeviceGet(9999)
    finally:
        logger_obj.removeHandler(counter_handler)

    assert counter_handler.count == 2, "Expected exactly 2 ERROR-level driver logs to be counted"


# Usage 6: use a custom python callback function to change log level and log format
def test_cuda_driver_log_custom_callback_overrides_default(caplog):
    driver.cuInit(0)

    logger_name = "my_library.cuda-driver"
    logger_obj = logging.getLogger(logger_name)

    # Log everything as DEBUG, prefixed with the logger name instead of "[CUDA Driver] "
    def my_callback(cu_log_level, message):
        msg = message.decode("utf-8", errors="replace").rstrip("\n")
        logger_obj.debug("%s: %s", logger_obj.name, msg)

    register_cuda_error_log(logger_obj, callback=my_callback)

    # Default WARNING threshold filters out the DEBUG-level driver logs
    with caplog.at_level(logging.WARNING, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not caplog.records

    caplog.clear()

    # Lowering the threshold to DEBUG lets them through, with the custom level/format
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        driver.cuDeviceGet(9999)

    assert len(caplog.records) == 2
    assert all(r.levelno == logging.DEBUG for r in caplog.records)
    assert all(r.message.startswith(f"{logger_name}: ") for r in caplog.records)


# callback must be callable or None, else fail fast
def test_cuda_driver_log_callback_must_be_callable_or_none():
    with pytest.raises(TypeError):
        register_cuda_error_log(logging.getLogger("cuda.driver.bad_callback_test"), callback="not-callable")
