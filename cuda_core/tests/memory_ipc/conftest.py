# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-directory conftest for memory IPC tests.

Applies an outer-guard ``pytest.mark.timeout`` to every test in this directory.
Individual tests still drive their own per-process waits using
``child_timeout_sec()`` from ``helpers.child_processes``; this marker is the
final fallback so that no IPC test can wedge the CI runner for hours if
deadlock occurs.
"""

import pathlib

import pytest

_HERE = pathlib.Path(__file__).parent.resolve()
_TIMEOUT_SEC = 300  # 5 minutes per test; generous compared to child_timeout_sec().


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.timeout(_TIMEOUT_SEC)
    for item in items:
        try:
            item_path = pathlib.Path(str(item.fspath)).resolve()
        except OSError:
            continue
        if _HERE in item_path.parents:
            item.add_marker(marker)
