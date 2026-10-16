# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_lychee
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


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_native_reports_preserve_each_precommit_chunk_exit_code(tmp_path, monkeypatch):
    executable = tmp_path / "lychee"
    executable.touch()
    monkeypatch.setenv("LYCHEE_REPORT_DIR", str(tmp_path / "reports"))
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(tmp_path)),
        patch("run_lychee.sys.platform", "linux"),
        patch(
            "run_lychee.subprocess.run",
            side_effect=[subprocess.CompletedProcess([], 2), subprocess.CompletedProcess([], 0)],
        ) as run,
    ):
        assert main(["one.md"]) == 2
        assert main(["two.rst"]) == 0
    reports = []
    for call, code in zip(run.call_args_list, (2, 0), strict=True):
        command = call.args[0]
        assert command[5:8] == ["--format", "json", "--output"]
        report = Path(command[8])
        assert report.parent == tmp_path / "reports"
        assert json.loads(report.with_suffix(".status.json").read_text()) == {"exit_code": code}
        reports.append(report)
    assert reports[0] != reports[1]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_hook_checks_untracked_html_in_its_own_cache(tmp_path, monkeypatch):
    (tmp_path / "lychee").touch()
    docs = tmp_path / "artifacts/docs"
    docs.mkdir(parents=True)
    (docs / "index.html").write_text('<a href="guide.html#section">Guide</a>')
    (docs / "guide.html").write_text('<h1 id="section">Guide</h1>')
    (docs / "_static").mkdir()
    (docs / "_static/theme.html").write_text("ignored theme")
    cache = tmp_path / "rendered cache"
    monkeypatch.setenv("LYCHEE_CACHE_DIR", str(cache))
    monkeypatch.setattr(run_lychee, "ROOT", tmp_path)
    monkeypatch.delenv("LYCHEE_REPORT_DIR", raising=False)
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(tmp_path)),
        patch("run_lychee.sys.platform", "linux"),
        patch("run_lychee.repo_link_remaps", return_value=["--remap", "rendered mapping"]) as remaps,
        patch("run_lychee.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        assert main(["--rendered", "--cache"]) == 0
    assert (cache / "inputs.txt").read_text().splitlines() == [str(docs / "guide.html"), str(docs / "index.html")]
    run.assert_called_once_with(
        [
            str(tmp_path / "lychee"),
            "--remap",
            "rendered mapping",
            "--files-from",
            str(cache / "inputs.txt"),
            "--include-fragments=full",
            "--config",
            str(tmp_path / "lychee.toml"),
            "--cache",
        ],
        check=False,
        cwd=cache,
    )
    remaps.assert_called_once_with(tmp_path, rendered=True)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_hook_keeps_github_views_and_fragments_at_checked_commit(tmp_path, monkeypatch):
    commit = "a" * 40
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA-dev/cuda-python-private")
    with patch(
        "run_lychee.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=commit + "\n")
    ) as run:
        arguments = repo_link_remaps(tmp_path, rendered=True)
    run.assert_called_once_with(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    for index, view in enumerate(("blob", "tree")):
        pattern, replacement = arguments[2 * index + 1].split()
        url = f"https://github.com/NVIDIA/cuda-python/{view}/main/README.md#heading"
        assert (
            re.sub(pattern, replacement, url)
            == f"https://github.com/NVIDIA-dev/cuda-python-private/{view}/{commit}/README.md#heading"
        )


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_hook_rejects_missing_build_before_invoking_native_checker(tmp_path, monkeypatch):
    (tmp_path / "lychee").touch()
    monkeypatch.setattr(run_lychee, "ROOT", tmp_path)
    monkeypatch.setenv("LYCHEE_CACHE_DIR", str(tmp_path / "cache"))
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(tmp_path)),
        patch("run_lychee.sys.platform", "linux"),
        patch("run_lychee.subprocess.run") as run,
        pytest.raises(ValueError, match="No documentation inputs"),
    ):
        main(["--rendered", "--cache"])
    run.assert_not_called()
