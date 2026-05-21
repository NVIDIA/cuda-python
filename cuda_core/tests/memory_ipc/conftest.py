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
from helpers.child_processes import child_timeout_sec

_HERE = pathlib.Path(__file__).parent.resolve()


def _outer_timeout_sec() -> int:
    # The worst-case IPC test has three sequential CHILD_TIMEOUT_SEC waits in
    # the failure path (e.g. TestIpcReexport: event_c.wait, proc_b.join,
    # proc_c.join). Scaling by 3 lets such a test reach its own asserts before
    # the outer guard fires, while still cutting the budget by half on
    # non-sanitizer runs (90 s vs the previous 300 s) and scaling up under
    # compute-sanitizer (360 s).
    return 3 * child_timeout_sec()


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.timeout(_outer_timeout_sec())
    for item in items:
        try:
            item_path = pathlib.Path(str(item.fspath)).resolve()
        except OSError:
            continue
        if _HERE in item_path.parents:
            item.add_marker(marker)
