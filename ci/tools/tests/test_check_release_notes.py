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


NOTES = "cuda_bindings/releasenotes"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603,S607


def _release(repo: Path, tag: str, note: str | None) -> None:
    """Commit an optional note file, then tag."""
    if note is not None:
        path = repo / NOTES / f"{note}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\nfixes:\n  - x\n", encoding="utf-8")
    else:
        (repo / "unrelated.txt").write_text(tag, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", tag)
    _git(repo, "tag", tag)


@pytest.fixture
def notes_repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.test")
    _git(tmp_path, "config", "user.name", "Test")
    _release(tmp_path, "v13.4.3", None)  # before note files: still needs a hand-written page
    return tmp_path


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
class TestReleaseNotes:
    def test_first_notes_release_needs_a_note(self, notes_repo):
        _release(notes_repo, "v13.5.0", None)
        problems = check_release_notes("v13.5.0", "cuda-bindings", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added in the tree"]

    def test_first_notes_release_with_note(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        assert check_release_notes("v13.5.0", "cuda-bindings", notes_repo) == []

    def test_minor_release_without_new_note_fails(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        _release(notes_repo, "v13.6.0", None)
        problems = check_release_notes("v13.6.0", "cuda-bindings", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added since v13.5.0"]

    def test_minor_release_counts_notes_since_previous_minor(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        _release(notes_repo, "v13.5.1", "b-0000000000000002")  # added after 13.5.0, so part of 13.6.0's range
        _release(notes_repo, "v13.6.0", None)
        assert check_release_notes("v13.6.0", "cuda-bindings", notes_repo) == []

    def test_minor_release_may_start_at_a_patch_number(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        _release(notes_repo, "v13.6.0rc1", None)  # 13.6.0 was only ever a pre-release
        _release(notes_repo, "v13.6.1", "b-0000000000000002")
        assert check_release_notes("v13.6.1", "cuda-bindings", notes_repo) == []
        _release(notes_repo, "v13.7.0", None)
        problems = check_release_notes("v13.7.0", "cuda-bindings", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added since v13.6.1"]

    def test_patch_release_without_new_note_fails(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        _release(notes_repo, "v13.5.1", None)
        problems = check_release_notes("v13.5.1", "cuda-bindings", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added since v13.5.0"]

    def test_patch_release_with_new_note_passes(self, notes_repo):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        _release(notes_repo, "v13.5.1", "b-0000000000000002")
        assert check_release_notes("v13.5.1", "cuda-bindings", notes_repo) == []

    def test_prerelease_tag_needs_no_note(self, notes_repo):
        _release(notes_repo, "v13.5.0rc1", None)
        assert check_release_notes("v13.5.0rc1", "cuda-bindings", notes_repo) == []

    def test_missing_tag_is_reported(self, notes_repo):
        problems = check_release_notes("v13.5.0", "cuda-bindings", notes_repo)
        assert [path for path, _reason in problems] == ["<tag>"]

    def test_shallow_clone_is_rejected(self, notes_repo, tmp_path_factory):
        _release(notes_repo, "v13.5.0", "a-0000000000000001")
        shallow = tmp_path_factory.mktemp("shallow") / "clone"
        subprocess.run(  # noqa: S603
            ["git", "clone", "-q", "--depth", "1", f"file://{notes_repo}", str(shallow)],  # noqa: S607
            check=True,
        )
        problems = check_release_notes("v13.5.0", "cuda-bindings", shallow)
        assert [path for path, _reason in problems] == ["<clone>"]

    def test_earlier_releases_still_need_the_handwritten_page(self, notes_repo):
        problems = check_release_notes("v13.4.3", "cuda-bindings", notes_repo)
        assert [reason for _path, reason in problems] == ["missing"]

    def test_other_components_are_unaffected(self, notes_repo):
        problems = check_release_notes("cuda-core-v1.2.0", "cuda-core", notes_repo)
        assert [reason for _path, reason in problems] == ["missing"]

    def test_cuda_core_uses_its_own_tag_namespace_and_notes_dir(self, notes_repo):
        _release(notes_repo, "cuda-core-v1.2.1", None)  # before note files: still needs a hand-written page
        assert [r for _p, r in check_release_notes("cuda-core-v1.2.1", "cuda-core", notes_repo)] == ["missing"]

        _release(notes_repo, "cuda-core-v1.3.0", "a-0000000000000001")  # a bindings note does not count
        problems = check_release_notes("cuda-core-v1.3.0", "cuda-core", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added in the tree"]

        path = notes_repo / "cuda_core" / "releasenotes" / "b-0000000000000002.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("---\nfixes:\n  - x\n", encoding="utf-8")
        _git(notes_repo, "add", "-A")
        _git(notes_repo, "commit", "-q", "-m", "core note")
        _git(notes_repo, "tag", "cuda-core-v1.3.1")
        problems = check_release_notes("cuda-core-v1.3.1", "cuda-core", notes_repo)
        assert problems == []  # the note was added after 1.3.0

        _release(notes_repo, "cuda-core-v1.3.2", None)
        problems = check_release_notes("cuda-core-v1.3.2", "cuda-core", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added since cuda-core-v1.3.1"]

    def test_cuda_pathfinder_cutover(self, notes_repo):
        # 1.8.x (including patch releases) keeps hand-written pages; 1.9.0 and later need a note file.
        for tag in ("cuda-pathfinder-v1.8.3", "cuda-pathfinder-v1.8.4"):
            _release(notes_repo, tag, None)
            assert [r for _p, r in check_release_notes(tag, "cuda-pathfinder", notes_repo)] == ["missing"]
        _release(notes_repo, "cuda-pathfinder-v1.9.0", None)
        problems = check_release_notes("cuda-pathfinder-v1.9.0", "cuda-pathfinder", notes_repo)
        assert [reason for _path, reason in problems] == ["no release notes added in the tree"]
