# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for bridging CUDA driver ``cuLogs*`` messages into Python
``logging`` (see design doc / GitHub issue #671).
"""

import logging

import pytest

from cuda.bindings import driver
from cuda.core.utils import register_cuda_log_bridge, unregister_cuda_log_bridge


@pytest.fixture(autouse=True)
def teardown_bridge():
    yield
    # Guarantee the callback is unregistered after every test, so a leaked
    # bridge/slot from one test can't affect later ones.
    unregister_cuda_log_bridge()


# 使用方式1: 调用 register(logger), 通过 caplog 捕获 driver log
def test_cuda_driver_log_captured_via_caplog(caplog):
    # 1. 初始化 CUDA Driver
    driver.cuInit(0)

    # 2. 注册桥接，将驱动日志路由到用户提供的 logger
    logger_name = "cuda.driver.caplog_test"
    logger_obj = logging.getLogger(logger_name)
    register_cuda_log_bridge(logger_obj)

    # 3. 触发驱动错误：ordinal 9999 必然越界，驱动会写入两条 [E] 日志
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)

    # 4. 验证 caplog 中包含两条驱动错误日志
    messages = [r.message for r in caplog.records if r.name == logger_name]

    assert any("Parameter ordinal must be between 0 and 1" in m for m in messages), (
        f"Expected ordinal error in caplog, got:\n{messages}"
    )

    assert any("CUDA_ERROR_INVALID_DEVICE" in m and "cuDeviceGet" in m for m in messages), (
        f"Expected cuDeviceGet return error in caplog, got:\n{messages}"
    )


# 使用方式2: 通过 logging handler 把 driver log 倒入到文件
def test_cuda_driver_log_written_to_file(tmp_path):
    # 1. 初始化 CUDA Driver
    driver.cuInit(0)

    # 2. 准备一个独立的 logger，挂上写文件的 FileHandler
    #    （用独立 logger name，避免污染 root logger / 影响其他用例）
    log_file = tmp_path / "cuda_driver.log"
    logger_name = "cuda.driver.file_test"
    logger_obj = logging.getLogger(logger_name)
    logger_obj.setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger_obj.addHandler(file_handler)

    try:
        # 3. 注册桥接，将驱动日志路由到该 logger
        register_cuda_log_bridge(logger_obj)

        # 4. 触发驱动错误：ordinal 9999 必然越界，驱动会写入两条 [E] 日志
        driver.cuDeviceGet(9999)
    finally:
        # 5. 显式移除并关闭 handler：
        #    - close() 会 flush 缓冲区，确保日志落盘再读取
        #    - removeHandler 避免 handler 残留导致文件句柄泄漏 / 影响后续用例
        logger_obj.removeHandler(file_handler)
        file_handler.close()

    # 6. 校验文件内容
    content = log_file.read_text(encoding="utf-8")

    assert "Parameter ordinal must be between 0 and 1" in content, (
        f"Expected ordinal error in log file, got:\n{content}"
    )

    assert "CUDA_ERROR_INVALID_DEVICE" in content and "cuDeviceGet" in content, (
        f"Expected cuDeviceGet return error in log file, got:\n{content}"
    )


# 使用方式3: 动态绑定解绑 register -> unregister -> 再register
def test_cuda_driver_log_rebind_logger(caplog):
    driver.cuInit(0)

    name_a, name_b = "cuda.driver.rebind_a", "cuda.driver.rebind_b"

    # 先绑定 logger_a，确认能收到日志
    register_cuda_log_bridge(logging.getLogger(name_a))
    with caplog.at_level(logging.ERROR, logger=name_a):
        driver.cuDeviceGet(9999)
    assert any(r.name == name_a for r in caplog.records)

    # 换绑到 logger_b：必须先 unregister 释放驱动侧 slot，再用新 logger register
    unregister_cuda_log_bridge()
    caplog.clear()
    register_cuda_log_bridge(logging.getLogger(name_b))
    with caplog.at_level(logging.ERROR, logger=name_b):
        driver.cuDeviceGet(9999)

    # 旧 logger 不应再收到任何消息，新 logger 应正常收到
    assert not any(r.name == name_a for r in caplog.records)
    assert any(r.name == name_b for r in caplog.records)


# 使用方式4: 日志过滤 —— logger 的 level 阈值对驱动日志同样生效
def test_cuda_driver_log_level_filtering(caplog):
    driver.cuInit(0)

    logger_name = "cuda.driver.level_test"
    register_cuda_log_bridge(logging.getLogger(logger_name))

    # 阈值设为 CRITICAL：ERROR 级别的驱动日志应被过滤，caplog 收不到任何记录
    with caplog.at_level(logging.CRITICAL, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records)

    # 放低阈值到 ERROR：同样的错误应该能被捕获到
    with caplog.at_level(logging.ERROR, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert any(r.name == logger_name for r in caplog.records)


# 使用方式5: 监控/告警集成 —— 用自定义 Handler 统计驱动错误次数，
# 可据此接入 Prometheus/Datadog 等监控系统做告警
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
        register_cuda_log_bridge(logger_obj)
        driver.cuDeviceGet(9999)
    finally:
        logger_obj.removeHandler(counter_handler)

    assert counter_handler.count >= 1, "Expected at least one ERROR-level driver log to be counted"


# 使用方式6: integrate driver logs into existing logger through callback
def test_cuda_driver_log_custom_callback_overrides_default(caplog):
    driver.cuInit(0)

    # (a) 创建一个叫做 "my_library.cuda-driver" 的 logger
    logger_name = "my_library.cuda-driver"
    logger_obj = logging.getLogger(logger_name)

    # (b) 自定义 my_callback：不管驱动原始级别是 ERROR 还是 WARNING，
    #     统一记成 DEBUG；并在打印的消息前面加上 logger name 前缀
    def my_callback(logger, level, message):
        logger.debug("%s: %s", logger.name, message)

    # (c) register 时传入 callback，覆盖掉内部默认的
    #     logger.log(level, "[CUDA Driver] %s", message) 转发逻辑
    register_cuda_log_bridge(logger_obj, callback=my_callback)

    # 1. logging 默认的 WARNING 级别阈值下，DEBUG 级别的驱动日志会被过滤掉，
    #    caplog 收不到任何记录
    with caplog.at_level(logging.WARNING, logger=logger_name):
        driver.cuDeviceGet(9999)
    assert not any(r.name == logger_name for r in caplog.records), (
        "DEBUG-level driver log should be filtered out at the default WARNING level"
    )

    caplog.clear()

    # 2. 把 logger 的阈值调到 DEBUG，才能收到 my_callback 记录的驱动日志
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        driver.cuDeviceGet(9999)

    records = [r for r in caplog.records if r.name == logger_name]
    assert records, "Expected driver log records forwarded via my_callback once the threshold is DEBUG"

    # 验证 my_callback 生效：级别被统一改写成了 DEBUG（而不是默认的 ERROR/WARNING）
    assert all(r.levelno == logging.DEBUG for r in records), (
        f"Expected every record logged as DEBUG via my_callback, got levels: {[r.levelname for r in records]}"
    )

    # 验证消息前缀是 logger name（而不是 register_cuda_log_bridge 默认的 "[CUDA Driver] " 前缀）
    messages = [r.message for r in records]
    assert any(m.startswith(f"{logger_name}: ") for m in messages), (
        f"Expected messages prefixed with '{logger_name}: ', got:\n{messages}"
    )
    assert not any(m.startswith("[CUDA Driver] ") for m in messages), (
        f"Default '[CUDA Driver] ' forwarding should be overridden by my_callback, got:\n{messages}"
    )


# callback 必须是可调用对象或 None，否则应尽早报错
def test_cuda_driver_log_callback_must_be_callable_or_none():
    with pytest.raises(TypeError):
        register_cuda_log_bridge(logging.getLogger("cuda.driver.bad_callback_test"), callback="not-callable")
