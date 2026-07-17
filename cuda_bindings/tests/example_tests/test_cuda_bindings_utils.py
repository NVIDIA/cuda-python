# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared cuda.bindings sample helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_UTILITIES_DIR = Path(__file__).resolve().parents[3] / "samples" / "cuda_bindings" / "Utilities"
sys.path.insert(0, str(_UTILITIES_DIR))

from cuda_bindings_utils import check_cmd_line_flag, get_cmd_line_argument_int


@pytest.mark.parametrize(
    "argument, flag, expected",
    (
        ("--device=3", "device=", 3),
        ("device=3", "device=", 3),
        ("--kernel=7", "kernel=", 7),
        ("kernel=7", "kernel=", 7),
    ),
)
@pytest.mark.agent_authored(model="gpt-5")
def test_value_flag_accepts_documented_and_legacy_forms(
    monkeypatch: pytest.MonkeyPatch, argument: str, flag: str, expected: int
) -> None:
    monkeypatch.setattr(sys, "argv", ["sample.py", argument])

    assert check_cmd_line_flag(flag)
    assert get_cmd_line_argument_int(flag) == expected


@pytest.mark.agent_authored(model="gpt-5")
def test_flag_parameter_accepts_leading_dashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sample.py", "--device=4"])

    assert check_cmd_line_flag("--device=")
    assert get_cmd_line_argument_int("--device=") == 4


@pytest.mark.agent_authored(model="gpt-5")
def test_bare_flag_accepts_documented_and_legacy_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sample.py", "--help", "?"])

    assert check_cmd_line_flag("help")
    assert check_cmd_line_flag("?")


@pytest.mark.agent_authored(model="gpt-5")
def test_flags_do_not_match_option_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sample.py", "--device-name=3", "--helpful"])

    assert not check_cmd_line_flag("device=")
    assert not check_cmd_line_flag("help")
    assert get_cmd_line_argument_int("device=") == 0


@pytest.mark.parametrize("argument", ("--device=", "--device=invalid"))
@pytest.mark.agent_authored(model="gpt-5")
def test_invalid_integer_value_preserves_zero_default(monkeypatch: pytest.MonkeyPatch, argument: str) -> None:
    monkeypatch.setattr(sys, "argv", ["sample.py", argument])

    assert check_cmd_line_flag("device=")
    assert get_cmd_line_argument_int("device=") == 0
