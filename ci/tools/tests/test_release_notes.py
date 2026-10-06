# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "cuda_python" / "docs" / "exts"))
import release_notes
import release_ranges

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SECTION_KEYS = {"features", "issues", "upgrade", "deprecations", "critical", "security", "fixes", "other"}
NOTES_DIR = "pkg/releasenotes"

SECTIONS = [("features", "New Features"), ("fixes", "Bug Fixes"), ("issues", "Known Issues")]
PR_URL = "https://example.test/pull/{number}"


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_parse_release_tags_ignores_prereleases_and_other_namespaces():
    tags = ["v1.0.0", "v1.0.0rc1", "v1.0.0a0", "cuda-core-v1.2.0", "v1.0", "v1.0.1", "xv1.0.2"]
    assert set(release_ranges.parse_release_tags(tags, "v")) == {(1, 0, 0), (1, 0, 1)}
    assert set(release_ranges.parse_release_tags(tags, "cuda-core-v")) == {(1, 2, 0)}


def _parse(text):
    return tuple(int(part) for part in text.split("."))


CANDIDATES = [_parse(v) for v in ["13.5.0", "13.5.1", "13.5.2", "13.5.3", "13.6.0", "13.6.1", "14.0.0"]]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("13.5.0", None),
        ("13.5.1", "13.5.0"),
        ("13.5.3", "13.5.2"),
        ("13.5.2", "13.5.1"),
        ("13.6.0", "13.5.0"),  # a minor release re-lists the previous line's patches
        ("13.6.1", "13.6.0"),
        ("14.0.0", "13.6.0"),
    ],
)
def test_previous_release(version, expected):
    result = release_ranges.previous_release(_parse(version), CANDIDATES)
    if expected is None:
        assert result is None
    else:
        assert result == _parse(expected)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_minor_release_may_start_at_a_patch_number():
    # 13.4.0 never existed (it was only a pre-release): 13.4.1 is the minor release of its line.
    candidates = [_parse(v) for v in ["13.3.0", "13.3.1", "13.4.1", "13.4.2", "13.5.0", "13.5.1", "13.6.2"]]
    assert release_ranges.previous_release((13, 4, 1), candidates) == (13, 3, 0)
    assert release_ranges.previous_release((13, 4, 2), candidates) == (13, 4, 1)
    assert release_ranges.previous_release((13, 5, 0), candidates) == (13, 4, 1)
    assert release_ranges.previous_release((13, 6, 2), candidates) == (13, 5, 0)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("component", sorted(release_ranges.PACKAGES))
def test_first_notes_version_is_a_minor_release(component):
    # previous_release only sees releases from first_version on.  If that were a patch release
    # (X.Y.Z, Z > 0), it would be mistaken for the minor release of its line, and the next minor
    # release would be made relative to it instead of to the earlier patch-free baseline.
    first_version = release_ranges.PACKAGES[component].first_version
    assert first_version[2] == 0, (
        f"{component}: first version with note files {release_ranges.format_version(first_version)} must be an X.Y.0 release"
    )


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_previous_release_skips_missing_patch_numbers():
    assert release_ranges.previous_release((1, 0, 3), [(1, 0, 0), (1, 0, 3)]) == (1, 0, 0)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_render_page_orders_sections_links_prs_and_uses_issue_source():
    notes = [
        {"data": {"fixes": ["A fix."], "issues": ["Not shown: issues come from the issue source."]}, "pr": "7"},
        {"data": {"features": ["A feature.\n\nMore detail."]}, "pr": None},
    ]
    issues = [{"data": {"issues": ["An open issue."]}, "pr": "3"}]
    page = release_notes.render_page("Title", SECTIONS, notes, issues, PR_URL)
    assert page.index("New Features") < page.index("Bug Fixes") < page.index("Known Issues")
    assert "- A fix. (`PR #7 <https://example.test/pull/7>`__)" in page
    assert "- A feature.\n\n  More detail." in page
    assert "An open issue. (`PR #3 <https://example.test/pull/3>`__)" in page
    assert "Not shown" not in page


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_render_page_empty():
    assert "There are no release notes" in release_notes.render_page("Title", SECTIONS, [], [], PR_URL)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _note(repo: Path, name: str, section: str, text: str, uid: str) -> Path:
    path = repo / NOTES_DIR / f"{name}-{uid:0>16}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{section}:\n  - |\n    {text}\n", encoding="utf-8")
    return path


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.test")
    _git(root, "config", "user.name", "Test")
    (root / NOTES_DIR).mkdir(parents=True)
    (root / "docs" / "source").mkdir(parents=True)

    _note(root, "first-fix", "fixes", "First fix.", "1")
    _note(root, "open-issue", "issues", "Open issue.", "2")
    _note(root, "wsl-issue", "issues", "Resolved issue.", "3")
    _commit(root, "Initial notes (#1)")
    _git(root, "tag", "v1.0.0")
    _git(root, "tag", "v1.0.0rc1")

    _note(root, "new-feature", "features", "New feature.", "4")
    _commit(root, "Add feature (#2)")
    _note(root, "patch-fix", "fixes", "Fix needed in the patch.", "5")
    _commit(root, "Fix (#3)")
    _git(root, "tag", "v1.1.0")

    # The patch release is tagged on a maintenance branch that diverges from main.
    _git(root, "checkout", "-q", "-b", "maint", "v1.0.0")
    _note(root, "patch-fix", "fixes", "Fix needed in the patch.", "5")
    _commit(root, "Backport fix (#3)")
    _git(root, "tag", "v1.0.1")

    _git(root, "checkout", "-q", "main")
    _git(root, "rm", "-q", str(root / NOTES_DIR / "wsl-issue-0000000000000003.yaml"))
    _note(root, "after", "upgrade", "Change after the release.", "6")
    _commit(root, "Resolve issue and add change")
    return root


def _app(root: Path):
    config = types.SimpleNamespace(
        release_notes_dir=NOTES_DIR,
        release_notes_tag_prefix="v",
        release_notes_first_version="1.0.0",
        release_notes_title="Package {version}",
        release_notes_unreleased_title="In development",
        release_notes_prolog=".. currentmodule:: pkg",
        release_notes_pr_url=PR_URL,
        release_notes_output_dir="release",
    )
    return types.SimpleNamespace(srcdir=str(root / "docs" / "source"), config=config)


def _pages(root: Path) -> dict[str, str]:
    out = root / "docs" / "source" / "release"
    return {p.name: p.read_text(encoding="utf-8") for p in out.glob("*-notes.rst")}


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_generate_pages_ranges_and_known_issues(repo):
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)

    # No page for the rc tag; one per release tag plus the unreleased page.
    assert set(pages) == {"1.0.0-notes.rst", "1.0.1-notes.rst", "1.1.0-notes.rst", "unreleased-notes.rst"}

    assert "First fix." in pages["1.0.0-notes.rst"]

    # The prolog comes before the title, so release_date still finds the title underline.
    assert pages["1.0.0-notes.rst"].startswith(
        release_notes.GENERATED_MARKER + ".. currentmodule:: pkg\n\nPackage 1.0.0\n"
    )

    # The patch page lists only the backported note, even though the tag is
    # not an ancestor of main.
    assert "Fix needed in the patch." in pages["1.0.1-notes.rst"]
    assert "First fix." not in pages["1.0.1-notes.rst"]

    # The minor page lists the same fix again, plus the feature, but not the
    # notes of the previous minor release.
    assert "Fix needed in the patch." in pages["1.1.0-notes.rst"]
    assert "New feature." in pages["1.1.0-notes.rst"]
    assert "First fix." not in pages["1.1.0-notes.rst"]
    assert "(`PR #2 <https://example.test/pull/2>`__)" in pages["1.1.0-notes.rst"]

    # Known issues are listed on every release that still contains them ...
    for name in ("1.0.0-notes.rst", "1.0.1-notes.rst", "1.1.0-notes.rst"):
        assert "Open issue." in pages[name]
        assert "Resolved issue." in pages[name]

    # ... and the unreleased page shows only what is new, with the resolved issue gone.
    unreleased = pages["unreleased-notes.rst"]
    assert unreleased.removeprefix(release_notes.GENERATED_MARKER + ".. currentmodule:: pkg\n\n").startswith(
        "In development\n"
    )
    assert "Change after the release." in unreleased
    assert "Open issue." in unreleased
    assert "Resolved issue." not in unreleased
    assert "Fix needed in the patch." not in unreleased


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_edited_known_issues_do_not_change_earlier_pages(repo):
    # The issues entry of a note that still exists is removed (the note keeps another section).
    path = _note(repo, "open-issue", "fixes", "Now a fix.", "2")
    _commit(repo, "Resolve the open issue")
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)
    assert path.is_file()
    # Pages for tags keep the issue as it was at the tag ...
    assert "Open issue." in pages["1.0.0-notes.rst"]
    assert "Open issue." in pages["1.1.0-notes.rst"]
    # ... but the unreleased page reflects the checkout.
    assert "Open issue." not in pages["unreleased-notes.rst"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_empty_note_at_a_tag_is_tolerated(repo):
    empty = repo / NOTES_DIR / "empty-note-00000000000000ff.yaml"
    empty.write_text("# nothing yet\n", encoding="utf-8")
    _commit(repo, "Add empty note (#9)")
    _git(repo, "tag", "v1.2.0")
    empty.unlink()
    _commit(repo, "Remove empty note")
    release_notes._generate_pages(_app(repo))
    assert "1.2.0-notes.rst" in _pages(repo)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_generate_pages_removes_stale_pages(repo):
    release_notes._generate_pages(_app(repo))
    release_dir = repo / "docs" / "source" / "release"
    stale = release_dir / "9.9.9-notes.rst"
    stale.write_text(release_notes.GENERATED_MARKER + "stale", encoding="utf-8")
    handwritten = release_dir / "0.9.0-notes.rst"
    handwritten.write_text("hand-written", encoding="utf-8")
    release_notes._generate_pages(_app(repo))
    assert not stale.exists()
    assert handwritten.read_text(encoding="utf-8") == "hand-written"


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_generate_pages_refuses_to_overwrite_handwritten_pages(repo):
    release_dir = repo / "docs" / "source" / "release"
    release_dir.mkdir(parents=True)
    (release_dir / "1.0.0-notes.rst").write_text("hand-written", encoding="utf-8")
    with pytest.raises(release_notes.ExtensionError, match="hand-written"):
        release_notes._generate_pages(_app(repo))


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_generate_pages_rejects_shallow_clone(repo, tmp_path):
    shallow = tmp_path / "shallow"
    subprocess.run(  # noqa: S603
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)],  # noqa: S607
        check=True,
    )
    (shallow / "docs" / "source").mkdir(parents=True, exist_ok=True)
    with pytest.raises(release_notes.ExtensionError, match="shallow"):
        release_notes._generate_pages(_app(shallow))


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_generate_pages_requires_tags(repo):
    for tag in _git(repo, "tag", "--list").split():
        _git(repo, "tag", "-d", tag)
    with pytest.raises(release_notes.ExtensionError, match="no release tags"):
        release_notes._generate_pages(_app(repo))


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("Text. (#12)", "Text. (`PR #12 <https://example.test/pull/12>`__)"),
        (
            "Text. (#12, #34)\n",
            "Text. (`PR #12 <https://example.test/pull/12>`__, `PR #34 <https://example.test/pull/34>`__)",
        ),
        (
            "Text. ( #12 ,#34 )",
            "Text. (`PR #12 <https://example.test/pull/12>`__, `PR #34 <https://example.test/pull/34>`__)",
        ),
        # As a final paragraph, so it also works after a list or a table.
        ("Para.\n\n- item\n\n(#5)", "Para.\n\n- item\n\n(`PR #5 <https://example.test/pull/5>`__)"),
    ],
)
def test_explicit_pr_numbers_are_linked(entry, expected):
    # The automatic number is ignored when the entry names its own.
    assert release_notes._render_entry(entry, "999", PR_URL) == expected


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_explicit_pr_numbers_work_anywhere_in_the_entry():
    link = "`PR #12 <https://example.test/pull/12>`__"
    # In the middle of a sentence, and no automatic link is added.
    assert (
        release_notes._render_entry("Fixed in (#12) as well as other things.", "5", PR_URL)
        == f"Fixed in ({link}) as well as other things."
    )
    # More than one marker, in different paragraphs.
    assert release_notes._render_entry("One (#12).\n\nTwo (#12).", None, PR_URL) == f"One ({link}).\n\nTwo ({link})."
    # Things that merely look similar are not markers.
    for text in ("See issue #12.", "Ends with (#12x)", "Ends with (see #12)", "Ends with (12)"):
        assert release_notes._render_entry(text, None, PR_URL) == text


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "text",
    [
        "Call ``f(#5)`` to do it.",
        "Call `(#99)` to do it.",
        "A literal that\nspans ``f(\n(#5)\n)`` lines.",
        "Both ``(#5)`` and ``(#6, #7)`` are literal text.",
    ],
)
def test_markers_inside_backquoted_text_are_left_alone(text):
    assert not release_notes._has_explicit_prs(text)
    # With no real marker, the automatic link is added to the first paragraph, and the text is intact.
    rendered = release_notes._render_entry(text, "5", PR_URL)
    assert rendered == f"{text} (`PR #5 <https://example.test/pull/5>`__)"


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_marker_outside_backquoted_text_still_works_next_to_one_inside():
    link = "`PR #8 <https://example.test/pull/8>`__"
    # The real marker replaces the automatic link; the literal one is untouched.
    assert release_notes._render_entry("Use ``f(#5)`` (#8).", "1", PR_URL) == f"Use ``f(#5)`` ({link})."
    # Notes whose only markers are literal still use the git lookup.
    assert release_notes._needs_pr_lookup({"fixes": ["Use ``f(#5)``."]})
    assert not release_notes._needs_pr_lookup({"fixes": ["Use ``f(#5)`` (#8)."]})


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_explicit_pr_numbers_skip_the_git_lookup_and_apply_per_entry(repo):
    # The note file is added by a "move" commit (#9) but its entries name the real PRs.
    _git(repo, "checkout", "-q", "main")
    path = repo / NOTES_DIR / "moved-0000000000000007.yaml"
    path.write_text(
        "---\nfixes:\n  - |\n    Moved fix. (#77, #78)\n  - |\n    Entry without a number.\n"
        "issues:\n  - |\n    Moved issue. (#99)\n",
        encoding="utf-8",
    )
    only_explicit = repo / NOTES_DIR / "explicit-0000000000000008.yaml"
    only_explicit.write_text("---\nfeatures:\n  - |\n    All explicit. (#55)\n", encoding="utf-8")
    _commit(repo, "Move notes into place (#9)")
    _git(repo, "tag", "v1.2.0")
    release_notes._generate_pages(_app(repo))
    page = _pages(repo)["1.2.0-notes.rst"]

    assert "Moved fix. (`PR #77 <https://example.test/pull/77>`__, `PR #78 <https://example.test/pull/78>`__)" in page
    assert "Moved issue. (`PR #99 <https://example.test/pull/99>`__)" in page
    assert "All explicit. (`PR #55 <https://example.test/pull/55>`__)" in page
    # An entry without a number in the same note still uses the commit subject.
    assert "Entry without a number. (`PR #9 <https://example.test/pull/9>`__)" in page
    assert page.count("pull/9>") == 1


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_minor_release_starting_at_a_patch_number_and_relisted_patches(repo):
    # 2.0.0 and 2.0.1, then a line that starts at 2.1.1 (2.1.0 was only a pre-release).
    _git(repo, "checkout", "-q", "main")
    _note(repo, "n200", "fixes", "Fix in 2.0.0.", "c")
    _commit(repo, "Fix (#20)")
    _git(repo, "tag", "v2.0.0")
    _git(repo, "tag", "v2.1.0rc1")
    _note(repo, "n201", "fixes", "Fix in 2.0.1.", "d")
    _commit(repo, "Fix (#21)")
    _git(repo, "tag", "v2.0.1")
    _note(repo, "n211", "fixes", "Fix in 2.1.1.", "e")
    _commit(repo, "Fix (#22)")
    _git(repo, "tag", "v2.1.1")
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)

    assert "2.1.0-notes.rst" not in pages
    assert "Fix in 2.0.1." in pages["2.0.1-notes.rst"]
    assert "Fix in 2.0.0." not in pages["2.0.1-notes.rst"]
    # The minor release is relative to 2.0.0, so it re-lists the 2.0.1 patch.
    assert "Fix in 2.1.1." in pages["2.1.1-notes.rst"]
    assert "Fix in 2.0.1." in pages["2.1.1-notes.rst"]
    assert "Fix in 2.0.0." not in pages["2.1.1-notes.rst"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_pages_for_releases_on_other_branches_and_the_unreleased_page(repo):
    # The newest tag, v1.1.1, is only on a maintenance branch; main continues after v1.1.0.
    _git(repo, "checkout", "-q", "-b", "maint-1.1", "v1.1.0")
    _note(repo, "maint-fix", "fixes", "Fix only on the maintenance branch.", "b")
    _commit(repo, "Backport (#30)")
    _git(repo, "tag", "v1.1.1")
    _git(repo, "checkout", "-q", "main")

    # Building on main does not fail, and shows the patch release's page ...
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)
    assert "Fix only on the maintenance branch." in pages["1.1.1-notes.rst"]
    # ... while the unreleased page is relative to the newest release in main's history.
    unreleased = pages["unreleased-notes.rst"]
    assert "Change after the release." in unreleased
    assert "Fix only on the maintenance branch." not in unreleased

    # Building on the maintenance branch, with a newer minor release only on main.
    # (Remove the generated pages first so that committing does not sweep them into git.)
    shutil.rmtree(repo / "docs" / "source" / "release")
    _git(repo, "checkout", "-q", "main")
    _note(repo, "next", "features", "Next minor feature.", "c")
    _commit(repo, "Next feature (#31)")
    _git(repo, "tag", "v1.2.0")
    _git(repo, "checkout", "-q", "maint-1.1")
    _note(repo, "maint-only", "fixes", "Not yet released on the maintenance branch.", "d")
    _commit(repo, "Another fix (#32)")
    (repo / "docs" / "source").mkdir(parents=True, exist_ok=True)
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)
    assert "Next minor feature." in pages["1.2.0-notes.rst"]
    unreleased = pages["unreleased-notes.rst"]
    assert "Not yet released on the maintenance branch." in unreleased
    assert "Next minor feature." not in unreleased
    assert "Fix only on the maintenance branch." not in unreleased  # it is in v1.1.1, the baseline


def _rewrite(repo, name, uid, text):
    """Edit a note in the checkout, without committing (as when the docs are built from main)."""
    path = next((repo / NOTES_DIR).glob(f"{name}-*{uid}.yaml"))
    path.write_text(path.read_text(encoding="utf-8").replace(OLD_TEXTS[name], text), encoding="utf-8")


OLD_TEXTS = {
    "first-fix": "First fix.",
    "patch-fix": "Fix needed in the patch.",
    "open-issue": "Open issue.",
}


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_edits_to_released_notes_change_the_pages_that_list_them(repo):
    # Which notes a page lists comes from the tags; what they say comes from the checkout (known issues excepted).
    _rewrite(repo, "first-fix", "1", "First fix, reworded.")
    _rewrite(repo, "patch-fix", "5", "Patch fix, reworded.")
    _rewrite(repo, "open-issue", "2", "Open issue, reworded.")
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)

    assert "First fix, reworded." in pages["1.0.0-notes.rst"]
    assert "First fix." not in pages["1.0.0-notes.rst"]
    # A note that was backported is on the patch page and on the next minor page, with the new text.
    for name in ("1.0.1-notes.rst", "1.1.0-notes.rst"):
        assert "Patch fix, reworded." in pages[name]
        assert "Fix needed in the patch." not in pages[name]
    # A known issue keeps the text it had at each tag; only the unreleased page shows the edit.
    for name in ("1.0.0-notes.rst", "1.0.1-notes.rst", "1.1.0-notes.rst"):
        assert "Open issue." in pages[name]
        assert "reworded" not in pages[name].split("Known Issues")[1]
    assert "Open issue, reworded." in pages["unreleased-notes.rst"]
    # Editing a note does not move it to another release.
    assert "First fix" not in pages["1.1.0-notes.rst"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_note_missing_from_the_checkout_keeps_its_text_at_the_tag(repo):
    # "Resolved issue." was deleted on main (see the fixture) but was in the tree at 1.0.0 and 1.1.0.
    # The note is not in the checkout, so the pages use its text at their own tags.
    # A note only on the maintenance branch is likewise missing from main's checkout.
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)
    assert "Resolved issue." in pages["1.0.0-notes.rst"]
    assert "Resolved issue." in pages["1.1.0-notes.rst"]
    assert "Resolved issue." not in pages["unreleased-notes.rst"]

    # (Remove the generated pages first so that committing does not sweep them into git.)
    shutil.rmtree(repo / "docs" / "source" / "release")
    _git(repo, "checkout", "-q", "-b", "maint-only", "v1.0.1")
    _note(repo, "branch-only", "fixes", "Only on a maintenance branch.", "f")
    _commit(repo, "Branch-only fix (#40)")
    _git(repo, "tag", "v1.0.2")
    _git(repo, "checkout", "-q", "main")
    (repo / "docs" / "source").mkdir(parents=True, exist_ok=True)
    release_notes._generate_pages(_app(repo))
    assert "Only on a maintenance branch." in _pages(repo)["1.0.2-notes.rst"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_sections_use_only_the_default_keys_with_known_issues_last():
    keys = [key for key, _title in release_ranges.SECTIONS]
    assert set(keys) == DEFAULT_SECTION_KEYS
    assert len(keys) == len(DEFAULT_SECTION_KEYS)
    assert keys[-1] == "issues"
    assert release_ranges.PRELUDE not in keys


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_upgrade_section_is_titled_breaking_changes():
    # cuda_core/docs/source/support.rst points readers to "Breaking Changes" in the release notes.
    assert dict(release_ranges.SECTIONS)["upgrade"] == "Breaking Changes"


def _ignored(package_dir, name):
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", "--no-index", f"{package_dir}/docs/source/release/{name}-notes.rst"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("component", sorted(release_ranges.PACKAGES))
def test_gitignore_covers_generated_pages_and_not_hand_written_ones(component):
    package = release_ranges.PACKAGES[component]
    package_dir = package.notes_dir.removesuffix("/releasenotes")
    major, minor, _patch = package.first_version
    fmt = release_ranges.format_version

    generated = [
        package.first_version,
        (major, minor, 11),
        (major, minor + 1, 0),
        (major, minor + 10, 0),
        (major, minor + 100, 0),
        (major + 1, 0, 0),
        (major + 90, 0, 0),
        (major + 900, 0, 0),
    ]
    for version in generated:
        assert _ignored(package_dir, fmt(version)), f"{package_dir}: {fmt(version)} should be ignored"
    assert _ignored(package_dir, "unreleased")

    # Releases before the first version with note files keep their tracked, hand-written pages.
    hand_written = [(major, minor - 1, 0), (major, minor - 1, 10), (major - 1, 9, 9), (major - 1, 100, 0)]
    for version in hand_written:
        if min(version) < 0:
            continue
        assert not _ignored(package_dir, fmt(version)), f"{package_dir}: {fmt(version)} must not be ignored"
    assert not _ignored(package_dir, f"{fmt((major, minor - 1, 0))}rc1")


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_renamed_note_counts_as_a_new_note_in_the_release_that_contains_the_rename(repo):
    # "New feature." was added for 1.1.0.  Renaming its file is a delete and an add, so the
    # note is not lost (git would otherwise report a rename, not an add), and it is not listed
    # twice.
    _git(repo, "checkout", "-q", "main")
    old = next((repo / NOTES_DIR).glob("new-feature-*.yaml"))
    new = old.with_name("renamed-feature-0000000000000004.yaml")
    _git(repo, "mv", str(old), str(new))
    _git(repo, "commit", "-q", "-m", "Rename a note (#50)")
    _git(repo, "tag", "v1.2.0")
    release_notes._generate_pages(_app(repo))
    pages = _pages(repo)

    assert pages["1.1.0-notes.rst"].count("New feature.") == 1  # built from its own tag: the old name
    assert pages["1.2.0-notes.rst"].count("New feature.") == 1  # a minor release: the new name
