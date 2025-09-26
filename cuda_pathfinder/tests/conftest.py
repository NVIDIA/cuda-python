# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


def pytest_configure(config):
    config.custom_info = []


def _cli_has_flag(args, flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in args)


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    if not config.getoption("verbose"):
        return
    if _cli_has_flag(config.invocation_params.args, "--iterations"):
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
