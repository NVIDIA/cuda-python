# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings.utils import envvar_bool

_VAR = "CUDA_PYTHON_TEST_ENVVAR_BOOL"


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("0", False, id="zero"),
        pytest.param(" 0 ", False, id="zero-padded"),
        pytest.param("", False, id="empty"),
        pytest.param("   ", False, id="blank"),
        pytest.param("1", True, id="one"),
        pytest.param("2", True, id="two"),
        pytest.param("-0", False, id="negative-zero"),
        pytest.param("false", False, id="false"),
        pytest.param("FALSE", False, id="false-upper"),
        pytest.param("no", False, id="no"),
        pytest.param("off", False, id="off"),
        pytest.param("true", True, id="true"),
        pytest.param("True", True, id="true-capitalised"),
        pytest.param("yes", True, id="yes"),
        pytest.param("on", True, id="on"),
        # Anything unrecognised keeps the historical set-means-true behaviour.
        pytest.param("banana", True, id="unrecognised"),
    ],
)
def test_envvar_bool_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv(_VAR, raw)
    assert envvar_bool(_VAR) is expected


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("default", [False, True])
def test_envvar_bool_unset_returns_default(monkeypatch, default):
    monkeypatch.delenv(_VAR, raising=False)
    assert envvar_bool(_VAR, default) is default


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("raw", ["", "   "])
def test_envvar_bool_blank_returns_default(monkeypatch, raw):
    """Blank is "not set", so it must not override a True default."""
    monkeypatch.setenv(_VAR, raw)
    assert envvar_bool(_VAR, True) is True
