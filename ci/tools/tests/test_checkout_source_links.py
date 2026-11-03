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
import checkout_source_links as links


def git(root, *arguments):
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkout(tmp_path):
    git(tmp_path, "init")
    git(
        tmp_path,
        "-c",
        "user.name=Link Test",
        "-c",
        "user.email=link-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-m",
        "Initial test revision",
    )
    return tmp_path


def remap(arguments, url):
    for argument in arguments[1::2]:
        pattern, replacement = argument.split()
        url = re.sub(pattern, replacement.replace("$1", r"\g<1>"), url)
    return url


@pytest.mark.parametrize("view", ["blob", "tree"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_authored_mapping_needs_no_git_or_repository_metadata(tmp_path, monkeypatch, view):
    root = tmp_path / "checkout space$#"
    monkeypatch.setenv("GITHUB_REPOSITORY", "invalid metadata")
    with patch("checkout_source_links.subprocess.run") as run:
        arguments = links.repo_link_remaps(root)
    run.assert_not_called()
    url = f"https://github.com/NVIDIA/cuda-python/{view}/main/docs/a%20guide.md?plain=1#heading"
    assert remap(arguments, url) == f"{root.as_uri()}/docs/a%20guide.md?plain=1#heading"


@pytest.mark.parametrize("view", ["blob", "tree"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_public_rendered_mapping_uses_checkout_commit_and_keeps_suffix(tmp_path, monkeypatch, view):
    root = checkout(tmp_path)
    commit = git(root, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_REPOSITORY", "nvidia/CUDA-PYTHON")
    monkeypatch.setenv("GITHUB_BASE_REF", "12.9.x")
    monkeypatch.setenv("GITHUB_REF_NAME", "123/merge")
    arguments = links.repo_link_remaps(root, rendered=True)
    url = f"https://github.com/NVIDIA/cuda-python/{view}/main/README.md?plain=1#heading"
    assert remap(arguments, url) == f"https://github.com/NVIDIA/cuda-python/{view}/{commit}/README.md?plain=1#heading"
    assert links.github_source_ref(root) == commit


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_other_rendered_checkout_omits_only_remapped_source_fragments(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "Example/project")
    with patch("checkout_source_links.subprocess.run") as run:
        arguments = links.repo_link_remaps(tmp_path, rendered=True)
        assert links.github_source_ref(tmp_path) == "main"
    run.assert_not_called()
    for view in ("blob", "tree"):
        url = f"https://github.com/NVIDIA/cuda-python/{view}/main/docs/guide.md?plain=1#heading"
        assert remap(arguments, url) == f"{tmp_path.as_uri()}/docs/guide.md?plain=1"
    for url in ("guide.html#heading", "file:///docs/guide.html#heading", "https://example.org/guide#heading"):
        assert remap(arguments, url) == url


@pytest.mark.parametrize("rendered", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/cuda-python/blob/12.9.x/README.md#heading",
        "https://github.com/NVIDIA/cuda-python/tree/v13.0.0/cuda_bindings",
        "https://github.com/NVIDIA/cuda-python/blob/0123456789abcdef/README.md",
        "https://github.com/NVIDIA/cuda-python/blob/mainish/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/main",
        "https://github.com/Example/project/blob/main/README.md#heading",
        "https://github.com/NVIDIA/other-repository/tree/main/docs",
        "http://github.com/NVIDIA/cuda-python/blob/main/README.md",
        "https://example.com/redirect?url=https://github.com/NVIDIA/cuda-python/blob/main/README.md",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_unmatched_urls_remain_literal(tmp_path, monkeypatch, rendered, url):
    monkeypatch.setenv("GITHUB_REPOSITORY", "Example/project")
    assert remap(links.repo_link_remaps(tmp_path, rendered=rendered), url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/cuda-python.git",
        "git@github.com:NVIDIA/cuda-python.git",
        "ssh://git@github.com/NVIDIA/cuda-python.git",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_local_rendered_checkout_needs_no_cached_remote_head(tmp_path, monkeypatch, url):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root = checkout(tmp_path)
    git(root, "remote", "add", "upstream", url)
    git(root, "remote", "add", "origin", "https://github.com/Example/project.git")
    assert links.github_source_ref(root) == git(root, "rev-parse", "HEAD")


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_explicit_local_repository_takes_precedence_over_remotes(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root = checkout(tmp_path)
    git(root, "config", "lychee.repository", links.PUBLIC_REPOSITORY)
    git(root, "remote", "add", "upstream", "https://github.com/Example/project.git")
    assert links.github_source_ref(root) == git(root, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_REPOSITORY", "Example/project")
    assert links.github_source_ref(root) == "main"


@pytest.mark.parametrize("identity", ["", "not-a-repository", "OWNER/", "https://github.com/Example/project"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_invalid_environment_identity_fails_explicitly(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("GITHUB_REPOSITORY", identity)
    with pytest.raises(ValueError, match="OWNER/REPOSITORY"):
        links.github_source_ref(tmp_path)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_invalid_explicit_local_identity_fails_instead_of_using_remotes(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root = checkout(tmp_path)
    git(root, "config", "lychee.repository", "invalid")
    git(root, "remote", "add", "origin", "https://github.com/NVIDIA/cuda-python.git")
    with pytest.raises(ValueError, match="OWNER/REPOSITORY"):
        links.github_source_ref(root)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_unidentified_rendered_checkout_explains_explicit_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root = checkout(tmp_path)
    with pytest.raises(ValueError, match="git config lychee.repository OWNER/REPOSITORY"):
        links.github_source_ref(root)
