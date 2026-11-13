# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

from cuda.pathfinder._utils.resolve_path import resolve_path


@pytest.mark.agent_authored(model="gpt-6")
def test_resolve_path_anchors_relative_parts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "toolkit" / "library.a"
    target.parent.mkdir()
    target.touch()

    found = resolve_path("toolkit", Path("library.a"))
    assert found == target.resolve()
    monkeypatch.chdir(tmp_path.parent)
    assert found.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink parent traversal")
@pytest.mark.agent_authored(model="gpt-6")
def test_resolve_path_follows_symlink_before_parent(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "leaf").mkdir(parents=True)
    (sdk / "include").mkdir()
    target = sdk / "include" / "header.h"
    target.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(sdk / "leaf", target_is_directory=True)

    assert resolve_path(alias, "..", "include", "header.h") == target.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative paths")
@pytest.mark.thread_unsafe
@pytest.mark.agent_authored(model="gpt-6")
def test_resolve_path_anchors_another_drive_current_directory(monkeypatch, tmp_path):
    target = tmp_path / "toolkit" / "library.a"
    target.parent.mkdir()
    target.touch()
    used_drives = ctypes.windll.kernel32.GetLogicalDrives()
    drive = next(f"{chr(ord('A') + index)}:" for index in range(25, 2, -1) if not used_drives & (1 << index))

    subst = Path(os.environ["SYSTEMROOT"], "System32", "subst.exe").resolve()
    assert subst.is_absolute() and subst.is_file()
    subprocess.run(  # noqa: S603 - trusted System32 executable and test-owned arguments.
        [str(subst), drive, str(tmp_path)], check=True, capture_output=True
    )
    try:
        # Remember a cwd on the temporary drive, then switch back to the host drive.
        monkeypatch.chdir(drive + "\\")
        monkeypatch.chdir(tmp_path)
        assert Path.cwd().drive.casefold() != drive.casefold()
        found = resolve_path(drive + "toolkit", "library.a")
        assert found == target.resolve()
    finally:
        subprocess.run(  # noqa: S603 - remove only the temporary drive created by this test.
            [str(subst), drive, "/d"], check=True, capture_output=True
        )
