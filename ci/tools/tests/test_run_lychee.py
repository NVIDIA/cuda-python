# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_lychee import main


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_main_uses_hook_environment_despite_other_binary_on_path(tmp_path, monkeypatch, platform):
    scripts = tmp_path / "hook environment" / ("Scripts" if platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    name = "lychee.exe" if platform == "win32" else "lychee"
    executable = scripts / name
    executable.touch()
    other_directory = tmp_path / "other installation"
    other_directory.mkdir()
    (other_directory / name).touch(mode=0o755)
    monkeypatch.setenv("PATH", str(other_directory))
    arguments = ["--cache", "--max-concurrency=4", "guide with spaces.md"]
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(scripts)),
        patch("run_lychee.sys.platform", platform),
        patch("run_lychee.subprocess.run", return_value=subprocess.CompletedProcess([], 2)) as run,
    ):
        assert main(arguments) == 2
    run.assert_called_once_with([str(executable), *arguments], check=False)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_main_reports_missing_hook_binary_without_falling_back_to_path(tmp_path, monkeypatch):
    scripts = tmp_path / "hook environment" / "bin"
    other_directory = tmp_path / "other installation"
    other_directory.mkdir()
    (other_directory / "lychee").touch(mode=0o755)
    monkeypatch.setenv("PATH", str(other_directory))
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(scripts)),
        patch("run_lychee.sys.platform", "linux"),
        patch("run_lychee.subprocess.run") as run,
        pytest.raises(SystemExit, match="lychee was not found") as error,
    ):
        main(["guide.md"])
    assert str(scripts / "lychee") in str(error.value)
    run.assert_not_called()
