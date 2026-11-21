# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from ci.tools.check_tagged_release_notes import main, notes_path


def check_rendered(root, tag, component, bindings_package=None):
    args = [
        "--git-tag",
        tag,
        "--component",
        component,
        "--repo-root",
        str(root / "source"),
        "--rendered-docs",
        str(root / "artifacts" / "docs"),
    ]
    if bindings_package is not None:
        args.extend(("--bindings-package", json.dumps(bindings_package)))
    return main(args)


def write_source(root, package, version, content="Release notes."):
    path = root / "source" / notes_path(package, version)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")


def write_html(root, versioned_path, version, href, *, note=True):
    versioned_root = root / "artifacts" / "docs" / versioned_path
    (versioned_root / "release").mkdir(parents=True, exist_ok=True)
    (versioned_root / "release.html").write_text(f'<a href="{href}">Release notes</a>', encoding="utf-8")
    if note:
        (versioned_root / "release" / f"{version}-notes.html").write_text("<h1>Release notes</h1>", encoding="utf-8")


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing-page", "missing rendered release notes"),
        ("wrong-link", "missing relative link"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_required_bindings_page_and_link_must_be_in_the_versioned_tree(tmp_path, capsys, failure, expected_error):
    version = "12.9.11rc1"
    component = "cuda-bindings"
    versioned_path = f"{component}/{version}"
    href = f"release/{version}-notes.html"
    package = {"package_root": "cuda_bindings_12", "release_version": version}
    write_source(tmp_path, "cuda_bindings_12", version)
    write_html(tmp_path, f"{component}/latest", version, href)
    broken_href = f"../latest/{href}" if failure == "wrong-link" else href
    write_html(tmp_path, versioned_path, version, broken_href, note=failure != "missing-page")

    assert check_rendered(tmp_path, f"v{version}", component, package) == 1
    assert expected_error in capsys.readouterr().err

    write_html(tmp_path, versioned_path, version, href)
    assert check_rendered(tmp_path, f"v{version}", component, package) == 0


@pytest.mark.agent_authored(model="gpt-6")
def test_metapackage_payload_is_flat(tmp_path):
    version = "13.5.1"
    write_source(tmp_path, "cuda_python", version)
    write_html(tmp_path, version, version, f"release/{version}-notes.html")

    assert check_rendered(tmp_path, f"v{version}", "cuda-python") == 0


@pytest.mark.agent_authored(model="gpt-6")
def test_all_checks_the_component_selected_by_the_tag(tmp_path):
    version = "1.3.1"
    write_source(tmp_path, "cuda_core", version)
    write_html(tmp_path, f"cuda-core/{version}", version, f"release/{version}-notes.html")

    assert check_rendered(tmp_path, f"cuda-core-v{version}", "all") == 0


@pytest.mark.parametrize("tag", ("cuda-core-v1.3.1a0", "cuda-core-v1.3.1.dev1", "cuda-core-v1.3.1.post1"))
@pytest.mark.agent_authored(model="gpt-6")
def test_tags_without_required_notes_need_no_rendered_page(tmp_path, tag):
    assert check_rendered(tmp_path, tag, "cuda-core") == 0


@pytest.mark.parametrize("content", ("", "Post-release notes."))
@pytest.mark.agent_authored(model="gpt-6")
def test_pathfinder_post_release_requires_rendered_notes_for_nonempty_source(tmp_path, capsys, content):
    version = "1.9.1.post1"
    tag = f"cuda-pathfinder-v{version}"
    write_source(tmp_path, "cuda_pathfinder", version, content)
    if not content:
        assert check_rendered(tmp_path, tag, "all") == 0
        return

    assert check_rendered(tmp_path, tag, "all") == 1
    assert "missing rendered release notes" in capsys.readouterr().err

    write_html(tmp_path, "cuda-pathfinder/1.9.1", version, f"release/{version}-notes.html")
    assert check_rendered(tmp_path, tag, "all") == 0
