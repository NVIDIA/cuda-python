# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


def pytest_configure(config):
    config.custom_info = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    if not config.getoption("verbose"):
        return
    if hasattr(config.option, "iterations"):  # pytest-freethreaded runs all tests at least twice
        return
    if getattr(config.option, "count", 1) > 1:  # pytest-repeat
        return

    if config.custom_info:
        terminalreporter.write_sep("=", "INFO summary")
        for msg in config.custom_info:
            terminalreporter.line(f"INFO {msg}")


@pytest.fixture
def info_summary_append(request):
    def _append(message):
        request.config.custom_info.append(f"{request.node.name}: {message}")

    return _append


@pytest.fixture(autouse=True)
def reset_search_context():
    """Reset the default search context between tests."""
    from cuda.pathfinder._utils.toolchain_tracker import reset_default_context

    reset_default_context()
    yield
    reset_default_context()
