# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DescriptorSpec

# load_dl_windows imports ctypes.wintypes and requires ctypes.windll at module
# scope, so it cannot even be imported elsewhere; keep the import inside the test.
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows-only DLL loader")


@pytest.mark.agent_authored(model="claude-opus-5")
def test_already_loaded_check_queries_newest_dll_first(monkeypatch):
    """The already-loaded probe must use the same new -> old order as the system search.

    ``windows_dlls`` is tabulated oldest-first and every other entry point
    reverses it, so querying it forward here would report the oldest loaded
    version of a library that is present in more than one version.
    """
    from cuda.pathfinder._dynamic_libs import load_dl_windows

    queried: list[str] = []

    class FakeKernel32:
        @staticmethod
        def GetModuleHandleW(dll_name: str) -> int:
            queried.append(dll_name)
            return 0  # nothing is loaded, so every name is probed

    monkeypatch.setattr(load_dl_windows, "kernel32", FakeKernel32)

    desc = DescriptorSpec(
        name="test_lib",
        packaged_with="other",
        windows_dlls=("testlib64_11.dll", "testlib64_12.dll", "testlib64_13.dll"),
    )

    assert load_dl_windows.check_if_already_loaded_from_elsewhere(desc, False) is None
    assert queried == ["testlib64_13.dll", "testlib64_12.dll", "testlib64_11.dll"]
