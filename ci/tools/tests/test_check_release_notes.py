# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from check_release_notes import (
    check_release_notes,
    is_post_release,
    load_backport_branch,
    main,
    parse_version_from_tag,
)


@pytest.fixture
def git_source(tmp_path, monkeypatch):
    """Model read-only Git commands without creating commits in test fixtures."""
    files = {}
    calls = []
    commit = "1" * 40

    def run(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is True
        calls.append(command)
        if command[1] == "rev-parse":
            return subprocess.CompletedProcess(command, 0, stdout=f"{commit}\n", stderr="")
        if command[1] == "cat-file":
            object_name = kwargs["input"].strip()
            ref, path = object_name.split(":", 1)
            assert ref == commit
            stdout = "blob\n" if path in files else f"{object_name} missing\n"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if command[1] == "show":
            ref, path = command[2].split(":", 1)
            assert ref == commit
            return subprocess.CompletedProcess(command, 0, stdout=files[path], stderr=b"")
        pytest.fail(f"Unexpected Git command: {command}")

    monkeypatch.setattr("check_release_notes.subprocess.run", run)
    return files, calls


class TestParseVersionFromTag:
    def test_plain_tag_bindings(self):
        assert parse_version_from_tag("v13.1.0", "cuda-bindings") == "13.1.0"

    def test_plain_tag_python(self):
        assert parse_version_from_tag("v13.1.0", "cuda-python") == "13.1.0"

    def test_component_prefix_core(self):
        assert parse_version_from_tag("cuda-core-v0.7.0", "cuda-core") == "0.7.0"

    def test_component_prefix_pathfinder(self):
        assert parse_version_from_tag("cuda-pathfinder-v1.5.2", "cuda-pathfinder") == "1.5.2"

    def test_post_release(self):
        assert parse_version_from_tag("v12.6.2.post1", "cuda-bindings") == "12.6.2.post1"

    def test_invalid_tag(self):
        assert parse_version_from_tag("not-a-tag", "cuda-core") is None

    def test_no_v_prefix(self):
        assert parse_version_from_tag("13.1.0", "cuda-bindings") is None

    def test_component_prefix_mismatch(self):
        # cuda-core-v* must not be accepted for component=cuda-pathfinder
        assert parse_version_from_tag("cuda-core-v0.7.0", "cuda-pathfinder") is None

    def test_bare_v_rejected_for_core(self):
        # bare v* belongs to cuda-bindings/cuda-python, not cuda-core
        assert parse_version_from_tag("v0.7.0", "cuda-core") is None

    def test_unknown_component(self):
        assert parse_version_from_tag("v13.1.0", "bogus") is None

    def test_path_traversal_rejected(self):
        assert parse_version_from_tag("v1.0.0/../evil", "cuda-bindings") is None

    def test_path_separator_rejected(self):
        assert parse_version_from_tag("v1/2/3", "cuda-bindings") is None

    def test_leading_dot_rejected(self):
        assert parse_version_from_tag("v.1.0", "cuda-bindings") is None

    def test_whitespace_rejected(self):
        assert parse_version_from_tag("v1.0.0 ", "cuda-bindings") is None

    def test_trailing_suffix_rejected(self):
        # \w permits alphanumerics + underscore only; hyphens and shell meta-chars are out
        assert parse_version_from_tag("v1.0.0-extra", "cuda-bindings") is None


class TestIsPostRelease:
    def test_normal(self):
        assert not is_post_release("13.1.0")

    def test_post(self):
        assert is_post_release("12.6.2.post1")

    def test_post_no_number(self):
        assert is_post_release("1.0.0.post")


class TestCheckReleaseNotes:
    def _make_notes(self, tmp_path, pkg, version, content="Release notes."):
        d = tmp_path / pkg / "docs" / "source" / "release"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{version}-notes.rst"
        f.write_text(content)
        return f

    def test_present_and_nonempty(self, tmp_path):
        self._make_notes(tmp_path, "cuda_core", "0.7.0")
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert problems == []

    def test_missing(self, tmp_path):
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert problems[0][1] == "missing"

    def test_empty(self, tmp_path):
        self._make_notes(tmp_path, "cuda_core", "0.7.0", content="")
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert problems[0][1] == "empty"

    def test_post_release_skipped(self, tmp_path):
        problems = check_release_notes("v12.6.2.post1", "cuda-bindings", tmp_path)
        assert problems == []

    def test_invalid_tag(self, tmp_path):
        problems = check_release_notes("not-a-tag", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_component_prefix_mismatch(self, tmp_path):
        # Pass a cuda-core tag with component=cuda-pathfinder; must be rejected.
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-pathfinder", tmp_path)
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_unknown_component(self, tmp_path):
        problems = check_release_notes("v13.1.0", "bogus", tmp_path)
        assert len(problems) == 1
        assert "unknown component" in problems[0][1]

    def test_plain_v_tag(self, tmp_path):
        self._make_notes(tmp_path, "cuda_python", "13.1.0")
        problems = check_release_notes("v13.1.0", "cuda-python", tmp_path)
        assert problems == []


class TestLoadBackportBranch:
    def test_from_versions_yml(self, tmp_path):
        d = tmp_path / "ci"
        d.mkdir(parents=True)
        (d / "versions.yml").write_text('backport_branch: "12.9.x"\n')

        assert load_backport_branch(tmp_path) == "12.9.x"

    def test_from_github_ref_name_for_legacy_backport_branch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_REF_NAME", "12.9.x")

        assert load_backport_branch(tmp_path) == "12.9.x"

    def test_ignores_non_backport_github_ref_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_REF_NAME", "main")

        assert load_backport_branch(tmp_path) is None


class TestMain:
    def _make_notes(self, tmp_path, pkg, version, content="Release notes."):
        d = tmp_path / pkg / "docs" / "source" / "release"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{version}-notes.rst").write_text(content)

    def test_success(self, tmp_path):
        d = tmp_path / "cuda_core" / "docs" / "source" / "release"
        d.mkdir(parents=True)
        (d / "0.7.0-notes.rst").write_text("Notes here.")
        rc = main(["--git-tag", "cuda-core-v0.7.0", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 0

    def test_failure(self, tmp_path):
        rc = main(["--git-tag", "cuda-core-v0.7.0", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 1

    def test_post_skip(self, tmp_path):
        rc = main(["--git-tag", "v12.6.2.post1", "--component", "cuda-bindings", "--repo-root", str(tmp_path)])
        assert rc == 0

    def test_unparsable_tag_returns_2(self, tmp_path):
        rc = main(["--git-tag", "not-a-tag", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 2

    def test_path_traversal_returns_2(self, tmp_path):
        rc = main(["--git-tag", "v1.0.0/../evil", "--component", "cuda-bindings", "--repo-root", str(tmp_path)])
        assert rc == 2

    def test_component_prefix_mismatch_returns_2(self, tmp_path):
        rc = main(
            [
                "--git-tag",
                "cuda-core-v0.7.0",
                "--component",
                "cuda-pathfinder",
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert rc == 2

    def test_mainline_bindings_requires_backport_decision(self, tmp_path, capsys):
        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "<backport-git-tag>" in captured.err

    def test_mainline_bindings_accepts_not_planned(self, tmp_path):
        self._make_notes(tmp_path, "cuda_bindings", "13.3.0")

        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
                "--backport-git-tag",
                "not planned",
            ]
        )

        assert rc == 0

    def test_mainline_bindings_checks_planned_backport_notes(self, tmp_path, capsys):
        self._make_notes(tmp_path, "cuda_bindings", "13.3.0")

        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
                "--backport-git-tag",
                "v12.9.7",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "12.9.7-notes.rst" in captured.err

    def test_mainline_bindings_accepts_planned_backport_notes(self, tmp_path):
        self._make_notes(tmp_path, "cuda_bindings", "13.3.0")
        self._make_notes(tmp_path, "cuda_bindings", "12.9.7")

        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
                "--backport-git-tag",
                "v12.9.7",
            ]
        )

        assert rc == 0

    def test_mainline_cuda_python_accepts_planned_backport_notes(self, tmp_path):
        self._make_notes(tmp_path, "cuda_python", "13.3.0")
        self._make_notes(tmp_path, "cuda_python", "12.9.7")

        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-python",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
                "--backport-git-tag",
                "v12.9.7",
            ]
        )

        assert rc == 0

    def test_backport_bindings_missing_notes_warns_without_failing(self, tmp_path, monkeypatch, capsys):
        summary_path = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

        rc = main(
            [
                "--git-tag",
                "v12.9.7",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert "::warning file=cuda_bindings/docs/source/release/12.9.7-notes.rst::" in captured.out
        assert "12.9.7-notes.rst" in summary_path.read_text()

    def test_mainline_bindings_rejects_non_backport_tag(self, tmp_path):
        self._make_notes(tmp_path, "cuda_bindings", "13.3.0")

        rc = main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--backport-branch",
                "12.9.x",
                "--backport-git-tag",
                "v13.2.0",
            ]
        )

        assert rc == 2


@pytest.mark.parametrize("version", ["1.5.2", "1.5.2.post1"])
@pytest.mark.parametrize("contents", ["", "Release notes."])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_pathfinder_notes_allow_empty_files_and_require_post_notes(tmp_path, monkeypatch, version, contents):
    notes = tmp_path / "cuda_pathfinder" / "docs" / "source" / "release" / f"{version}-notes.rst"
    notes.parent.mkdir(parents=True)
    notes.write_text(contents)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            ["--git-tag", f"cuda-pathfinder-v{version}", "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]
        )
        == 0
    )
    assert output.read_text() == f"version={version}\n"


@pytest.mark.parametrize("version", ["1.5.2", "1.5.2.post1"])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_pathfinder_missing_notes_fail_without_version_output(tmp_path, monkeypatch, version):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            ["--git-tag", f"cuda-pathfinder-v{version}", "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]
        )
        == 1
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "component,tag,version,extra_args",
    [
        ("cuda-core", "cuda-core-v0.7.0", "0.7.0", []),
        ("cuda-bindings", "v12.9.7", "12.9.7", ["--backport-branch", "12.9.x"]),
        ("cuda-python", "v13.3.0", "13.3.0", ["--backport-branch", "12.9.x", "--backport-git-tag", "not planned"]),
        ("cuda-bindings", "v12.9.7.post1", "12.9.7.post1", []),
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1")
def test_successful_checks_expose_version_for_every_policy(tmp_path, monkeypatch, component, tag, version, extra_args):
    if component == "cuda-core" or version == "13.3.0":
        notes = tmp_path / component.replace("-", "_") / "docs" / "source" / "release" / f"{version}-notes.rst"
        notes.parent.mkdir(parents=True)
        notes.write_text("Release notes.")
    output = tmp_path / "output"
    output.write_text("existing=value\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main(["--git-tag", tag, "--component", component, "--repo-root", str(tmp_path), *extra_args]) == 0
    assert output.read_text() == f"existing=value\nversion={version}\n"


@pytest.mark.parametrize("tag", ["invalid", "cuda-core-v1.5.2"])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_invalid_pathfinder_tags_do_not_expose_version(tmp_path, monkeypatch, tag):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main(["--git-tag", tag, "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("contents,expected", [(b"", 0), (b"Release notes.", 0), (None, 1)])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_git_ref_reads_pathfinder_post_notes_instead_of_worktree(tmp_path, monkeypatch, git_source, contents, expected):
    files, calls = git_source
    path = "cuda_pathfinder/docs/source/release/1.5.2.post1-notes.rst"
    worktree_notes = tmp_path / path
    worktree_notes.parent.mkdir(parents=True)
    worktree_notes.write_text("Working tree notes must not decide the tagged release policy.")
    if contents is not None:
        files[path] = contents
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "--git-tag",
                "cuda-pathfinder-v1.5.2.post1",
                "--component",
                "cuda-pathfinder",
                "--repo-root",
                str(tmp_path),
                "--git-ref",
                "refs/tags/cuda-pathfinder-v1.5.2.post1",
            ]
        )
        == expected
    )
    assert calls[0] == [
        "git",
        "rev-parse",
        "--verify",
        "--end-of-options",
        "refs/tags/cuda-pathfinder-v1.5.2.post1^{commit}",
    ]
    assert worktree_notes.read_text() == "Working tree notes must not decide the tagged release policy."
    assert output.exists() == (expected == 0)


@pytest.mark.parametrize("component", ["cuda-bindings", "cuda-python"])
@pytest.mark.parametrize("backport_present,expected", [(True, 0), (False, 1)])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_git_ref_reads_release_backport_configuration_and_both_notes(
    tmp_path, monkeypatch, git_source, component, backport_present, expected
):
    files, _ = git_source
    files["ci/versions.yml"] = b'backport_branch: "12.9.x"\n'
    package = component.replace("-", "_")
    files[f"{package}/docs/source/release/13.3.0-notes.rst"] = b"Mainline notes."
    if backport_present:
        files[f"{package}/docs/source/release/12.9.7-notes.rst"] = b"Backport notes."
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "versions.yml").write_text('backport_branch: "12.8.x"\n')
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "--git-tag",
                "v13.3.0",
                "--component",
                component,
                "--repo-root",
                str(tmp_path),
                "--git-ref",
                "refs/tags/v13.3.0",
                "--backport-git-tag",
                "v12.9.7",
            ]
        )
        == expected
    )
    assert output.exists() == (expected == 0)


@pytest.mark.agent_authored(model="gpt-6.1")
def test_missing_git_ref_is_an_error_not_missing_notes(tmp_path, monkeypatch, capsys):
    def run(command, **kwargs):
        assert command[1] == "rev-parse"
        raise subprocess.CalledProcessError(128, command, stderr="fatal: Needed a single revision\n")

    monkeypatch.setattr("check_release_notes.subprocess.run", run)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "--git-tag",
                "cuda-pathfinder-v1.5.2",
                "--component",
                "cuda-pathfinder",
                "--repo-root",
                str(tmp_path),
                "--git-ref",
                "refs/tags/missing",
            ]
        )
        == 2
    )
    assert "cannot read Git release-notes source" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.agent_authored(model="gpt-6.1")
def test_git_read_failure_is_not_treated_as_missing_notes(tmp_path, monkeypatch, git_source, capsys):
    files, _ = git_source
    files["cuda_pathfinder/docs/source/release/1.5.2-notes.rst"] = b"Release notes."
    run = subprocess.run

    def failing_show(command, **kwargs):
        if command[1] == "show":
            raise subprocess.CalledProcessError(128, command, stderr=b"fatal: cannot read blob\n")
        return run(command, **kwargs)

    monkeypatch.setattr("check_release_notes.subprocess.run", failing_show)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "--git-tag",
                "cuda-pathfinder-v1.5.2",
                "--component",
                "cuda-pathfinder",
                "--repo-root",
                str(tmp_path),
                "--git-ref",
                "refs/tags/cuda-pathfinder-v1.5.2",
            ]
        )
        == 2
    )
    assert "cannot read blob" in capsys.readouterr().err
    assert not output.exists()
