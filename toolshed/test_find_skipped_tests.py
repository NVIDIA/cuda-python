# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from toolshed.find_skipped_tests import extract_test_status_sets


@pytest.mark.agent_authored(model="gpt-5")
def test_extract_test_status_sets_understands_subtest_parent_passes():
    log = """
##[group]run-tests core
tests/system/test_device.py::test_all_skipped SUBSKIPPED(i=0)
tests/system/test_device.py::test_all_skipped SUBSKIPPED(i=1)
tests/system/test_device.py::test_all_skipped PASSED
tests/system/test_device.py::test_mixed SUBSKIPPED(i=0)
tests/system/test_device.py::test_mixed SUBPASSED(i=1)
tests/system/test_device.py::test_mixed PASSED
tests/system/test_device.py::test_ordinary PASSED
"""

    skipped, non_skipped, test_suites = extract_test_status_sets(log)

    assert skipped == {
        "tests/system/test_device.py::test_all_skipped",
        "tests/system/test_device.py::test_mixed",
    }
    assert non_skipped == {
        "tests/system/test_device.py::test_mixed",
        "tests/system/test_device.py::test_ordinary",
    }
    assert test_suites == {
        "tests/system/test_device.py::test_all_skipped": "cuda_core",
        "tests/system/test_device.py::test_mixed": "cuda_core",
    }
