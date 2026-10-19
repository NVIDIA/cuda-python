# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_lychee
from run_lychee import main, rendered_inputs, write_inputs


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
        assert command[1:4] == ["--format", "json", "--output"]
        report = Path(command[4])
        assert report.parent == (tmp_path / "reports").resolve()
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
    cache = tmp_path / ".lychee-rendered"
    monkeypatch.setattr(run_lychee, "ROOT", tmp_path)
    monkeypatch.delenv("LYCHEE_REPORT_DIR", raising=False)
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(tmp_path)),
        patch("run_lychee.sys.platform", "linux"),
        patch("run_lychee.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        assert main(["--rendered", "--cache"]) == 0
    assert (cache / "inputs.txt").read_text().splitlines() == [str(docs / "guide.html"), str(docs / "index.html")]
    run.assert_called_once_with(
        [
            str(tmp_path / "lychee"),
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


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_hook_rejects_missing_build_before_invoking_native_checker(tmp_path, monkeypatch):
    (tmp_path / "lychee").touch()
    monkeypatch.setattr(run_lychee, "ROOT", tmp_path)
    with (
        patch("run_lychee.sysconfig.get_path", return_value=str(tmp_path)),
        patch("run_lychee.sys.platform", "linux"),
        patch("run_lychee.subprocess.run") as run,
        pytest.raises(ValueError, match="No documentation inputs"),
    ):
        main(["--rendered", "--cache"])
    run.assert_not_called()


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_rendered_inputs_include_all_components_and_exclude_static_assets(tmp_path):
    expected = ["cuda-bindings/latest/api.html", "cuda-core/latest/guide.html", "latest/index.html"]
    for name in [*expected, "cuda-core/latest/_static/theme.html", "_static/vendor/embed.html", "versions.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
    (tmp_path / "directory.html").mkdir()

    assert rendered_inputs(tmp_path) == [tmp_path / name for name in expected]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_write_inputs_keeps_spaces_and_writes_absolute_paths(tmp_path):
    output = tmp_path / "lists/files.txt"
    paths = [tmp_path / "docs/a guide.rst", tmp_path / "README.md"]

    write_inputs(paths, output)

    assert output.read_text(encoding="utf-8") == "".join(f"{path}\n" for path in paths)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_write_inputs_rejects_paths_that_would_split_into_multiple_inputs(tmp_path):
    output = tmp_path / "files.txt"

    with pytest.raises(ValueError, match="must not contain newlines"):
        write_inputs([tmp_path / "two\ninputs.md"], output)

    assert not output.exists()
