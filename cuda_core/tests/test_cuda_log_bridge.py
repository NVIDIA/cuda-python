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


# 使用方式1: register(logger) 后用 caplog 捕获 driver log
def test_cuda_driver_log_captured_via_caplog(caplog):
    driver.cuInit(0)

    logger_name = "cuda.driver.caplog_test"
    logger_obj = logging.getLogger(logger_name)
    register_cuda_error_log(logger_obj)

    # cuDeviceGet(9999) 越界，驱动写入两条 ERROR 日志
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)

    messages = [r.message for r in caplog.records if r.name == logger_name]

    assert any("Parameter ordinal must be between 0 and 1" in m for m in messages), (
        f"Expected ordinal error in caplog, got:\n{messages}"
    )

    assert any("CUDA_ERROR_INVALID_DEVICE" in m and "cuDeviceGet" in m for m in messages), (
        f"Expected cuDeviceGet return error in caplog, got:\n{messages}"
    )


# 使用方式2: 通过 FileHandler 把 driver log 写到文件
def test_cuda_driver_log_written_to_file(tmp_path):
    driver.cuInit(0)

    # 独立 logger name，避免污染 root logger / 影响其他用例
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
        # close() flush 落盘；removeHandler 防止跨用例累加 handler
        logger_obj.removeHandler(file_handler)
        file_handler.close()

    content = log_file.read_text(encoding="utf-8")

    assert "Parameter ordinal must be between 0 and 1" in content, (
        f"Expected ordinal error in log file, got:\n{content}"
    )

    assert "CUDA_ERROR_INVALID_DEVICE" in content and "cuDeviceGet" in content, (
        f"Expected cuDeviceGet return error in log file, got:\n{content}"
    )


# 使用方式3: register -> unregister -> 换绑另一个 logger
def test_cuda_driver_log_rebind_logger(caplog):
    driver.cuInit(0)

    name_a, name_b = "cuda.driver.rebind_a", "cuda.driver.rebind_b"

    register_cuda_error_log(logging.getLogger(name_a))
    with caplog.at_level(logging.ERROR, logger=name_a):
        driver.cuDeviceGet(9999)
    assert any(r.name == name_a for r in caplog.records)

    # 必须先 unregister 释放驱动侧 slot，再绑新 logger
    unregister_cuda_error_log()
    caplog.clear()
    register_cuda_error_log(logging.getLogger(name_b))
    with caplog.at_level(logging.ERROR, logger=name_b):
        driver.cuDeviceGet(9999)

    assert not any(r.name == name_a for r in caplog.records)
    assert any(r.name == name_b for r in caplog.records)


# 使用方式4: logger 的 level 阈值对驱动日志同样生效
def test_cuda_driver_log_level_filtering(caplog):
    driver.cuInit(0)

    logger_name = "cuda.driver.level_test"
    register_cuda_error_log(logging.getLogger(logger_name))

    # CRITICAL 阈值过滤掉 ERROR 级别的驱动日志
    with caplog.at_level(logging.CRITICAL, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records)

    # 降到 ERROR 阈值就能捕获到
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert any(r.name == logger_name for r in caplog.records)


# 使用方式5: 自定义 Handler 统计错误次数，接入监控/告警
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

    assert counter_handler.count >= 1, "Expected at least one ERROR-level driver log to be counted"


# 使用方式6: 用 callback 把 driver log 接进已有的 logger
def test_cuda_driver_log_custom_callback_overrides_default(caplog):
    driver.cuInit(0)

    logger_name = "my_library.cuda-driver"
    logger_obj = logging.getLogger(logger_name)

    # 按 raw CUlogLevel 分发；ERROR 记成 DEBUG
    def my_callback(user_data, cu_log_level, message, length):
        msg = message.decode("utf-8", errors="replace").rstrip("\n")
        if cu_log_level == driver.CUlogLevel.CU_LOG_LEVEL_WARNING:
            logger_obj.warning("%s: %s", logger_obj.name, msg)
        elif cu_log_level == driver.CUlogLevel.CU_LOG_LEVEL_ERROR:
            logger_obj.debug("%s: %s", logger_obj.name, msg)

    register_cuda_error_log(logger_obj, callback=my_callback)

    # 默认 WARNING 阈值下，DEBUG 级别的驱动日志被过滤掉
    with caplog.at_level(logging.WARNING, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records), (
        "DEBUG-level driver log should be filtered out at the default WARNING level"
    )

    caplog.clear()

    # 调到 DEBUG 阈值才能收到
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        driver.cuDeviceGet(9999)

    records = [r for r in caplog.records if r.name == logger_name]
    assert records, "Expected driver log records forwarded via my_callback once the threshold is DEBUG"

    # my_callback 生效：级别统一改成了 DEBUG
    assert all(r.levelno == logging.DEBUG for r in records), (
        f"Expected every record logged as DEBUG via my_callback, got levels: {[r.levelname for r in records]}"
    )

    # 消息前缀是 logger name，不是默认的 "[CUDA Driver] "
    messages = [r.message for r in records]
    assert any(m.startswith(f"{logger_name}: ") for m in messages), (
        f"Expected messages prefixed with '{logger_name}: ', got:\n{messages}"
    )
    assert not any(m.startswith("[CUDA Driver] ") for m in messages), (
        f"Default '[CUDA Driver] ' forwarding should be overridden by my_callback, got:\n{messages}"
    )


# callback 必须可调用或 None，否则应尽早报错
def test_cuda_driver_log_callback_must_be_callable_or_none():
    with pytest.raises(TypeError):
        register_cuda_error_log(logging.getLogger("cuda.driver.bad_callback_test"), callback="not-callable")
