# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from cuda.bindings._example_helpers import check_cmd_line_flag, get_cmd_line_argument_int


@pytest.fixture
def argv(monkeypatch):
    """Replace sys.argv, keeping a realistic program name in argv[0]."""

    def _argv(*args, prog="example.py"):
        monkeypatch.setattr(sys, "argv", [prog, *args])

    return _argv


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("args", "flag", "expected"),
    [
        ((), "device=", False),
        (("device=", "0"), "device=", True),
        (("help",), "help", True),  # a boolean flag has nothing after it
        (("wA=", "128", "hA=", "256"), "hA=", True),
        (("wA=", "128"), "hA=", False),
    ],
)
def test_check_cmd_line_flag(argv, args, flag, expected):
    argv(*args)
    assert check_cmd_line_flag(flag) is expected


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_cmd_line_flag_ignores_the_program_name(argv):
    argv(prog="help")
    assert check_cmd_line_flag("help") is False


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), 0),
        (("device=", "3"), 3),
        (("wA=", "128", "device=", "2"), 2),
        (("device=",), 0),  # nothing follows the flag
        (("nomatch", "7"), 0),
    ],
)
def test_get_cmd_line_argument_int(argv, args, expected):
    argv(*args)
    value = get_cmd_line_argument_int("device=")
    assert value == expected
    assert isinstance(value, int)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_get_cmd_line_argument_int_ignores_the_program_name(argv):
    argv("3", prog="device=")
    assert get_cmd_line_argument_int("device=") == 0
