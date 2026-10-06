# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension that generates release-notes pages from per-change note files.

The design is inspired by reno (https://docs.openstack.org/reno/): each change adds a small YAML
file, and the pages are assembled from the git history.  It behaves quite differently, though:
notes are attributed to releases by tag ranges rather than by reno's version labels, known issues
are listed on every release until resolved, and a note's text is always the current text.

A note file is a YAML mapping from a section key to a list of reStructuredText entries.  The
sections, and their order and titles, are ``release_ranges.SECTIONS``; known issues (``issues``) come
last.  A ``prelude`` section is the introduction of a page and is not a titled section.

Releases before ``release_notes_first_version`` keep their hand-written
``release/<version>-notes.rst`` pages.  For every release tag at or after it,
this extension generates ``<release_notes_output_dir>/release/<version>-notes.rst`` from
the notes in ``release_notes_dir``, plus an ``unreleased-notes`` page for notes
that are not yet in any release.  The pages are written next to the
hand-written ones, so all release notes share one URL scheme.  Generated pages
start with a marker comment; the extension refuses to overwrite a file without
it and only deletes stale files that have it.  They are git-ignored.

Which notes appear on which page
================================

Only plain ``<prefix>X.Y.Z`` tags are releases.  Pre-release tags (``rc``,
``a``, ``b``) and other packages' tags are ignored, so notes written during a
pre-release roll into the final release.  Branch names are never consulted:
every page is built from the history reachable from its own tag, so a patch
tag made on a maintenance branch works even though it is not an ancestor of
``main``.

* A **minor** release lists the notes added since the previous minor release.
  A line's minor release is its first release: normally ``X.Y.0``, but if that
  was only a pre-release, the lowest ``X.Y.Z`` that was released.
* A **patch** release (any other release of a line) lists the notes added
  since the previous release of the same ``X.Y`` line.
* The **"In development"** page lists the notes added since the newest release
  that is in the history of the commit being built.  A release tagged on another
  branch is not, but it still gets its own page, built from its own tag.

Because each range is computed independently, a fix that is backported to a
maintenance branch appears on that branch's patch page *and* on the page of
the next minor release that contains it.  This duplication is intentional.

Which notes are on which page is decided by the tags.  The text of a note is
the text in the checkout being built (the docs are published from the main
branch), so editing a note that was already released changes the page of every
release that lists it the next time the docs are published.  A note whose file
is no longer in the checkout (deleted, or only on a maintenance branch) keeps
the text it had at the tag.

Known issues
============

A note is listed on the releases whose range contains it, but known issues
must be listed on every release until they are resolved.  So the ``issues``
section is not taken from the range.  Instead, each page lists every ``issues``
entry in the notes that exist in the tree at that page's tag (the working
``HEAD`` for the unreleased page).  The text of an issue is the text it has at that
tag (the checkout for the unreleased page).  Resolving an issue means deleting the entry (or its
note file); pages for earlier tags keep listing it, unchanged.

Pull request links
==================

Each entry gets a link to the pull request that introduced it.  By default the
pull request is taken from the ``(#N)`` suffix of the subject of the commit that
added the note file (the squash-merge convention); notes whose commit has no
such suffix get no link.

Note that a fix backported by its own pull request on a maintenance branch gets
that backport's number on the patch page, because the note file is added to the
branch by the backport, while the next minor page (built from the main branch,
where the note came from the original pull request) shows the original number.
Use an explicit marker (below) to show the original on both.

An entry can name its pull requests itself with ``(#1234)`` or ``(#1234, #1235)`` anywhere in
its text.  Every such marker is replaced by links and the automatic lookup is skipped for that
entry.  A marker inside an inline literal (``code``) or other backquoted text is just text and is
left alone.  This is for notes whose file was added by a different pull request than the one that
made the change, for example notes moved in from hand-written pages.

The build fails on a shallow clone or when no release tags are present, since
that would silently produce incomplete pages.
"""

from __future__ import annotations

import functools
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import yaml
from release_ranges import (
    PRELUDE,
    SECTIONS,
    format_version,
    parse_release_tags,
    previous_release,
    section_key_re,
)
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)

UNRELEASED = "unreleased"
GENERATED_MARKER = ".. Generated by release_notes; do not edit or commit.\n\n"

_PR_SUFFIX_RE = re.compile(r"\(#(\d+)\)\s*$")
# An explicit "(#1234)" or "(#1234, #1235)" anywhere in an entry, except inside inline literals
# (``...``) or other backquoted text (`...`), where it is just text.  Spans are matched too, so
# that a marker inside one is skipped over rather than found.
_SPAN_OR_PRS_RE = re.compile(r"``.+?``|`[^`]+`|\(\s*(?P<numbers>#\d+(?:\s*,\s*#\d+)*)\s*\)", re.DOTALL)


# ---------------------------------------------------------------------------
# Pure logic (no git, no Sphinx)
# ---------------------------------------------------------------------------


def _indent_entry(text: str) -> str:
    lines = text.strip("\n").split("\n")
    out = [f"- {lines[0]}"]
    for line in lines[1:]:
        if line:
            out.append(f"  {line}")
        else:
            out.append("")
    return "\n".join(out)


def _pr_link(number: str, pr_url: str) -> str:
    return f"`PR #{number} <{pr_url.format(number=number)}>`__"


def _has_explicit_prs(text: str) -> bool:
    return any(m.group("numbers") is not None for m in _SPAN_OR_PRS_RE.finditer(text))


def _render_entry(entry: str, note_pr: str | None, pr_url: str) -> str:
    """Link *entry* to its pull request(s).

    Explicit ``(#N, ...)`` markers anywhere in the entry are replaced by links.
    Otherwise a link to *note_pr* is appended to the first paragraph.
    """
    text = entry.strip("\n")
    if _has_explicit_prs(text):

        def replace(m):
            if m.group("numbers") is None:
                return m.group(0)
            return "(" + ", ".join(_pr_link(number, pr_url) for number in re.findall(r"\d+", m.group("numbers"))) + ")"

        return _SPAN_OR_PRS_RE.sub(replace, text)
    if note_pr is None:
        return text
    first, sep, rest = text.partition("\n\n")
    return f"{first} ({_pr_link(note_pr, pr_url)}){sep}{rest}"


def _entries(data: dict) -> list[str]:
    """All the entries of a parsed note, in every section."""
    entries = []
    for value in data.values():
        if isinstance(value, str):
            value = [value]
        entries.extend(value or [])
    return entries


def _needs_pr_lookup(data: dict) -> bool:
    """Whether some entry of the note has no explicit pull request."""
    return any(not _has_explicit_prs(entry) for entry in _entries(data))


def render_page(
    title: str,
    sections: Iterable[tuple[str, str]],
    notes: list[dict],
    issues: list[dict],
    pr_url: str,
    prolog: str = "",
) -> str:
    """Render a release-notes page.

    *sections* is the ordered ``(key, heading)`` list.  *notes* are the parsed
    notes for the page's range.  *issues* are the notes that supply the known
    issues.  Each is a dict with the parsed note under ``"data"`` and its
    pull-request number (or None) under ``"pr"``.
    """
    lines = []
    if prolog:
        lines.extend([prolog, ""])
    lines.extend([title, "=" * len(title), ""])

    def entries_for(key, source):
        entries = []
        for note in source:
            value = note["data"].get(key) or []
            if isinstance(value, str):
                value = [value]
            entries.extend(_render_entry(entry, note["pr"], pr_url) for entry in value)
        return entries

    prelude = entries_for(PRELUDE, notes)
    if prelude:
        lines.extend(["\n\n".join(e.strip("\n") for e in prelude), ""])

    wrote_section = bool(prelude)
    for key, heading in sections:
        if key == "issues":
            entries = entries_for(key, issues)
        else:
            entries = entries_for(key, notes)
        if not entries:
            continue
        wrote_section = True
        lines.extend([heading, "-" * len(heading), ""])
        for entry in entries:
            lines.extend([_indent_entry(entry), ""])

    if not wrote_section:
        lines.extend(["There are no release notes for this release.", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# git glue
# ---------------------------------------------------------------------------


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check and result.returncode != 0:
        raise ExtensionError(f"release_notes: `git {' '.join(args)}` failed: {result.stderr.strip()}")
    return result.stdout


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],  # noqa: S607
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 0


def _check_clone(repo_root: Path, release_tags: dict) -> None:
    if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise ExtensionError(
            "release_notes: this is a shallow git clone, so release notes cannot be built "
            "(they would silently be incomplete). Fetch the full history, e.g. "
            "`git fetch --unshallow --tags` or `actions/checkout` with `fetch-depth: 0`."
        )
    if not release_tags:
        raise ExtensionError(
            "release_notes: no release tags were found, so release notes cannot be built. "
            "Fetch the tags, e.g. `git fetch --tags`."
        )


def _notes_in_range(repo_root: Path, notes_dir: str, tag: str, earliest_tag: str | None) -> list[str]:
    """The note files added by the commits in ``earliest_tag..tag`` that still exist at *tag*.

    Without *earliest_tag*, the whole history of *tag*.  Only commits reachable from *tag* but not
    from *earliest_tag* count, so a patch release tagged on a maintenance branch lists only what
    that branch added since the release it was made from.  A renamed file counts as a new note
    (a delete and an add), in the release that contains the rename.
    """
    notes_path = notes_dir
    rev = tag
    if earliest_tag is not None:
        rev = f"{earliest_tag}..{tag}"
    added = _git(repo_root, "log", "--no-renames", "--diff-filter=A", "--name-only", "--format=", rev, "--", notes_path)
    present = set(_git(repo_root, "ls-tree", "-r", "--name-only", tag, "--", notes_path).splitlines())
    paths = {path for path in added.splitlines() if path in present and path.endswith(".yaml")}
    return sorted(paths)


_warned_paths: set[str] = set()


# The git lookups below are cached for one build (_generate_pages clears the caches), because a note
# that is both in a page's range and in its known issues is looked up for the same ref twice.
@functools.cache
def _pr_number(repo_root: Path, ref: str, path: str) -> str | None:
    subjects = _git(repo_root, "log", "--diff-filter=A", "--format=%s", ref, "--", path).strip().split("\n")
    # The oldest commit that added the file is the last one listed.
    m = _PR_SUFFIX_RE.search(subjects[-1])
    if m is not None:
        return m.group(1)
    if path not in _warned_paths:
        _warned_paths.add(path)
        # Notes in an unmerged PR have no squash-merge subject yet, which is expected.
        log = logger.info
        if ref != "HEAD":
            log = logger.warning
        log("release_notes: no pull request number found for %s", path)
    return None


def _lookup_pr(repo_root: Path, ref: str, path: str, data: dict) -> str | None:
    """The note's pull request from git, unless every entry names its own."""
    if not _needs_pr_lookup(data):
        return None
    return _pr_number(repo_root, ref, path)


@functools.cache
def _note_at_tag(repo_root: Path, ref: str, path: str) -> dict:
    """Load a note as it was at *ref*."""
    return yaml.safe_load(_git(repo_root, "show", f"{ref}:{path}")) or {}


def _note_data(repo_root: Path, ref: str, path: str) -> dict:
    """Load a note: from the checkout if the file is there, else as it was at *ref*."""
    file = repo_root / path
    if file.is_file():
        return yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return _note_at_tag(repo_root, ref, path)


def _issue_notes(repo_root: Path, ref: str, notes_dir: str) -> list[dict]:
    """Return the notes in the tree at *ref* that contain known issues.

    The issues have the text they had at *ref*, so that later edits do not change earlier pages.
    The unreleased page (``HEAD``) uses the checkout, like the rest of its notes.
    """
    notes_path = notes_dir
    out = _git(repo_root, "grep", "-l", "-E", section_key_re("issues"), ref, "--", notes_path, check=False)
    notes = []
    for line in sorted(out.splitlines()):
        path = line.split(":", 1)[1]
        if ref == "HEAD":
            data = _note_data(repo_root, ref, path)
        else:
            data = _note_at_tag(repo_root, ref, path)
        notes.append({"data": data, "pr": _lookup_pr(repo_root, ref, path, data)})
    return notes


def build_page(
    repo_root: Path,
    notes_dir: str,
    tag: str,
    earliest_tag: str | None,
    title: str,
    pr_url: str,
    prolog: str = "",
) -> str | None:
    """Build the page for *tag*.

    Passing ``tag="HEAD"`` builds the unreleased page, which returns None when
    there are no unreleased notes.
    """
    notes = []
    for filename in _notes_in_range(repo_root, notes_dir, tag, earliest_tag):
        data = _note_data(repo_root, tag, filename)
        notes.append({"data": data, "pr": _lookup_pr(repo_root, tag, filename, data)})
    if tag == "HEAD" and not notes:
        return None
    issues = _issue_notes(repo_root, tag, notes_dir)
    return render_page(title, SECTIONS, notes, issues, pr_url, prolog)


# ---------------------------------------------------------------------------
# Sphinx integration
# ---------------------------------------------------------------------------


def _write_if_changed(path: Path, content: str) -> None:
    content = GENERATED_MARKER + content
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if not existing.startswith(GENERATED_MARKER):
            raise ExtensionError(f"release_notes: refusing to overwrite hand-written page {path}")
        if existing == content:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_pages(app: Sphinx) -> None:
    cfg = app.config
    if not cfg.release_notes_dir:
        return
    repo_root = Path(_git(Path(app.srcdir), "rev-parse", "--show-toplevel").strip())
    _pr_number.cache_clear()
    _note_at_tag.cache_clear()
    out_dir = Path(app.srcdir) / cfg.release_notes_output_dir
    written = set()

    tags = _git(repo_root, "tag", "--list").split()
    release_tags = parse_release_tags(tags, cfg.release_notes_tag_prefix)
    _check_clone(repo_root, release_tags)

    first = tuple(int(part) for part in cfg.release_notes_first_version.split("."))
    notes_releases = sorted(v for v in release_tags if v >= first)

    for version in notes_releases:
        earliest = previous_release(version, notes_releases)
        earliest_tag = None
        if earliest is not None:
            earliest_tag = release_tags[earliest]
            if not _is_ancestor(repo_root, earliest_tag, release_tags[version]):
                raise ExtensionError(
                    f"release_notes: {earliest_tag} is not an ancestor of {release_tags[version]}, "
                    "so the notes for the range cannot be determined."
                )
        title = cfg.release_notes_title.format(version=format_version(version))
        page = build_page(
            repo_root,
            cfg.release_notes_dir,
            release_tags[version],
            earliest_tag,
            title,
            cfg.release_notes_pr_url,
            cfg.release_notes_prolog,
        )
        _write_if_changed(out_dir / f"{format_version(version)}-notes.rst", page)
        written.add(f"{format_version(version)}-notes.rst")

    # The unreleased page lists what came after the newest release that is in the history of
    # HEAD.  Releases on other branches are not (a patch tagged on a maintenance branch is not
    # in the history of main, and docs built on that branch do not contain a newer minor
    # release tagged on main), but their own pages above are built from their own tags.
    reachable = [v for v in notes_releases if _is_ancestor(repo_root, release_tags[v], "HEAD")]
    latest = None
    if reachable:
        latest = release_tags[max(reachable)]
    title = cfg.release_notes_unreleased_title
    page = build_page(
        repo_root, cfg.release_notes_dir, "HEAD", latest, title, cfg.release_notes_pr_url, cfg.release_notes_prolog
    )
    if page is not None:
        _write_if_changed(out_dir / f"{UNRELEASED}-notes.rst", page)
        written.add(f"{UNRELEASED}-notes.rst")

    # Drop generated pages from earlier builds whose tags no longer exist.
    for stale in out_dir.glob("*-notes.rst"):
        if stale.name not in written and stale.read_text(encoding="utf-8").startswith(GENERATED_MARKER):
            stale.unlink()


def setup(app: Sphinx) -> dict:
    app.add_config_value("release_notes_dir", "", "env")
    app.add_config_value("release_notes_tag_prefix", "v", "env")
    app.add_config_value("release_notes_first_version", "", "env")
    app.add_config_value("release_notes_title", "{version} Release notes", "env")
    app.add_config_value("release_notes_prolog", "", "env")
    app.add_config_value("release_notes_unreleased_title", "In development", "env")
    app.add_config_value("release_notes_pr_url", "https://github.com/NVIDIA/cuda-python/pull/{number}", "env")
    app.add_config_value("release_notes_output_dir", "release", "env")
    app.connect("builder-inited", _generate_pages)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
