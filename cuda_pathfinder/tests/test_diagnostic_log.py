# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in logging in load_nvidia_dynamic_lib. No GPU or NVIDIA library needed."""

import importlib
import logging
import os
from unittest.mock import patch

import pytest

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as mod
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._utils import diagnostic_log

ENV_VAR = diagnostic_log.ENV_VAR_NAME
LOGGER_NAME = diagnostic_log.LOGGER_NAME
LOADED = LoadedDL("/opt/cuda/lib64/libcudart.so.12", False, 1234, "CUDA_PATH")


def reload_with_env(value):
    """Reimport diagnostic_log with ENV_VAR set to *value* (None to unset)."""
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    if value is not None:
        env[ENV_VAR] = value
    with patch.dict(os.environ, env, clear=True):
        return importlib.reload(diagnostic_log)


@pytest.fixture(autouse=True)
def restore_diagnostic_log():
    logger = logging.getLogger(LOGGER_NAME)
    saved = (logger.level, list(logger.handlers))
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    reload_with_env(None)


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.agent_authored(model="claude-opus-5")
def test_disabled_by_default(value):
    assert reload_with_env(value).LOGGER is None


@pytest.mark.parametrize(("value", "level"), [("INFO", logging.INFO), (" debug ", logging.DEBUG), ("10", 10)])
@pytest.mark.agent_authored(model="claude-opus-5")
def test_env_var_enables_logger(value, level):
    logger = reload_with_env(value).LOGGER
    assert logger is not None
    assert logger.name == LOGGER_NAME
    assert logger.level == level
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_invalid_value_warns_and_stays_disabled():
    with pytest.warns(UserWarning, match=ENV_VAR):
        assert reload_with_env("VERBOSE").LOGGER is None


@pytest.mark.agent_authored(model="claude-opus-5")
def test_does_not_configure_root_logger():
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    reload_with_env("DEBUG")
    assert (root.level, root.handlers) == before


@pytest.mark.agent_authored(model="claude-opus-5")
def test_logging_not_imported_at_module_scope():
    with open(diagnostic_log.__file__) as f:
        lines = f.read().splitlines()
    assert not [line for line in lines if line.startswith(("import logging", "from logging"))]


@pytest.fixture
def load_cudart(monkeypatch):
    """Call load_nvidia_dynamic_lib('cudart') with the search stubbed out."""

    def call(logger, result=LOADED):
        monkeypatch.setattr(mod, "LOGGER", logger)

        def fake_load(_libname):
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(mod, "_load_lib_no_cache", fake_load)
        mod.load_nvidia_dynamic_lib.cache_clear()
        try:
            return mod.load_nvidia_dynamic_lib("cudart")
        finally:
            mod.load_nvidia_dynamic_lib.cache_clear()

    return call


@pytest.mark.agent_authored(model="claude-opus-5")
def test_successful_load_logs_one_info_record(load_cudart, caplog):
    logger = reload_with_env("INFO").LOGGER
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        assert load_cudart(logger) is LOADED

    records = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert records[0].getMessage() == "loaded cudart from /opt/cuda/lib64/libcudart.so.12 (found via CUDA_PATH)"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_failed_load_logs_nothing(load_cudart, caplog):
    logger = reload_with_env("DEBUG").LOGGER
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME), pytest.raises(DynamicLibNotFoundError):
        load_cudart(logger, DynamicLibNotFoundError("not found"))
    assert [r for r in caplog.records if r.name == LOGGER_NAME] == []


@pytest.mark.agent_authored(model="claude-opus-5")
def test_silent_when_disabled(load_cudart, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert load_cudart(None) is LOADED
    assert [r for r in caplog.records if r.name == LOGGER_NAME] == []
