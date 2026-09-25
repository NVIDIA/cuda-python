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
