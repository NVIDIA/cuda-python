# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_lychee import main, repo_link_remaps


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_remaps_encode_checkout_path_and_preserve_url_suffix(tmp_path):
    root = tmp_path / "checkout space$#"
    arguments = repo_link_remaps(root)
    for index, view in enumerate(("blob", "tree")):
        assert arguments[2 * index] == "--remap"
        pattern, replacement = arguments[2 * index + 1].split()
        url = f"https://github.com/NVIDIA/cuda-python/{view}/main/docs/a%20guide.md?plain=1#some-heading"
        assert re.sub(pattern, replacement, url) == f"{root.as_uri()}/docs/a%20guide.md?plain=1#some-heading"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/cuda-python/blob/12.9.x/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/v13.0.0/cuda_bindings",
        "https://github.com/NVIDIA/cuda-python/blob/0123456789abcdef/README.md",
        "https://github.com/NVIDIA/cuda-python/blob/mainly/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/main",
        "https://github.com/rwgk/cuda-python/blob/main/README.md",
        "https://github.com/NVIDIA/other-repository/tree/main/docs",
        "http://github.com/NVIDIA/cuda-python/blob/main/README.md",
        "https://raw.githubusercontent.com/NVIDIA/cuda-python/main/README.md",
        "https://nvidia.github.io/cuda-python/latest/",
        "https://example.com/redirect?url=https://github.com/NVIDIA/cuda-python/blob/main/README.md",
        "https://example.com/https://github.com/NVIDIA/cuda-python/tree/main/docs",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_remaps_leave_other_targets_and_nested_urls_unchanged(tmp_path, url):
    arguments = repo_link_remaps(tmp_path)
    for index in (1, 3):
        pattern, replacement = arguments[index].split()
        assert re.sub(pattern, replacement, url) == url


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
    run.assert_called_once_with([str(executable), *repo_link_remaps(), *arguments], check=False)


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
