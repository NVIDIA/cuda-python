# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The INFO summary is the only output of the info_summary_append fixture.

Several suites (test_load_nvidia_dynamic_lib, test_driver_lib_loading,
test_find_*) report what they discovered exclusively through it, so the
conditions that suppress it are worth pinning down.
"""

from __future__ import annotations

import types

import pytest
from conftest import pytest_terminal_summary


class FakeTerminalReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_sep(self, sep: str, title: str) -> None:
        self.lines.append(f"{sep} {title}")

    def line(self, message: str) -> None:
        self.lines.append(message)


def make_config(**option_attrs: object) -> types.SimpleNamespace:
    option = types.SimpleNamespace(verbose=1, **option_attrs)
    return types.SimpleNamespace(
        option=option,
        custom_info=["some_test: hdr_dir='/somewhere'"],
        getoption=lambda name: getattr(option, name),
    )


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "option_attrs",
    [
        {},  # neither plugin installed
        {"iterations": 1},  # pytest-freethreaded installed, single iteration
        {"count": 1},  # pytest-repeat installed, single run
        {"iterations": 1, "count": 1},
    ],
    ids=["no-plugins", "freethreaded-single", "repeat-single", "both-single"],
)
def test_info_summary_is_emitted_for_a_single_pass(option_attrs):
    reporter = FakeTerminalReporter()
    pytest_terminal_summary(reporter, 0, make_config(**option_attrs))
    assert reporter.lines == ["= INFO summary", "INFO some_test: hdr_dir='/somewhere'"]


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "option_attrs",
    [{"iterations": 2}, {"count": 2}],
    ids=["freethreaded-repeats", "repeat-repeats"],
)
def test_info_summary_is_suppressed_when_tests_repeat(option_attrs):
    reporter = FakeTerminalReporter()
    pytest_terminal_summary(reporter, 0, make_config(**option_attrs))
    assert reporter.lines == []


@pytest.mark.agent_authored(model="claude-opus-5")
def test_info_summary_is_suppressed_without_verbose():
    reporter = FakeTerminalReporter()
    config = make_config()
    config.option.verbose = 0
    pytest_terminal_summary(reporter, 0, config)
    assert reporter.lines == []
