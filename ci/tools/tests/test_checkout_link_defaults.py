# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import checkout_link_defaults as links


def git(root, *arguments):
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkout(tmp_path):
    root = tmp_path / "checkout space$#"
    root.mkdir()
    git(root, "init")
    git(
        root,
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
    return root


def remote(root, alias, url, branch=None):
    git(root, "remote", "add", alias, url)
    if branch is not None:
        git(root, "update-ref", f"refs/remotes/{alias}/{branch}", "HEAD")
        git(root, "symbolic-ref", f"refs/remotes/{alias}/HEAD", f"refs/remotes/{alias}/{branch}")


def remap(arguments, url):
    for argument in arguments[1::2]:
        pattern, replacement = argument.split()
        url = re.sub(pattern, replacement, url)
    return url


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_local_public_default_uses_repository_url_and_keeps_url_suffix(tmp_path, monkeypatch):
    monkeypatch.delenv(links.METADATA_ENV, raising=False)
    root = checkout(tmp_path)
    remote(root, "canonical", "ssh://git@github.com/NVIDIA/cuda-python.git", "next.toolkit")
    remote(root, "origin", "git@github.com:rwgk/cuda-python.git", "personal")
    git(root, "config", "lychee.repository", links.PUBLIC_REPOSITORY)
    arguments = links.repo_link_remaps(root)
    for view in ("blob", "tree"):
        url = f"https://github.com/NVIDIA/cuda-python/{view}/next.toolkit/docs/a%20guide.md?plain=1#section"
        assert remap(arguments, url) == f"{root.as_uri()}/docs/a%20guide.md?plain=1#section"


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_private_checkout_maps_public_and_private_defaults_only(tmp_path, monkeypatch):
    monkeypatch.delenv(links.METADATA_ENV, raising=False)
    root = checkout(tmp_path)
    remote(root, "upstream", "git@github.com:Example/toolkit-private.git", "next/toolkit")
    remote(root, "public_repo", "https://github.com/NVIDIA/cuda-python.git", "trunk")
    remote(root, "origin", "https://github.com/developer/toolkit-private.git", "fork")
    arguments = links.repo_link_remaps(root)
    for repository, branch in (("Example/toolkit-private", "next/toolkit"), (links.PUBLIC_REPOSITORY, "trunk")):
        url = f"https://github.com/{repository}/tree/{branch}/cuda_bindings"
        assert remap(arguments, url) == f"{root.as_uri()}/cuda_bindings"
    fork_url = "https://github.com/developer/toolkit-private/tree/fork/README.md"
    assert remap(arguments, fork_url) == fork_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/cuda-python/blob/12.9.x/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/v13.0.0/cuda_bindings",
        "https://github.com/NVIDIA/cuda-python/blob/0123456789abcdef/README.md",
        "https://github.com/NVIDIA/cuda-python/blob/trunkish/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/trunk",
        "https://github.com/rwgk/cuda-python/blob/trunk/README.md",
        "https://github.com/NVIDIA/other-repository/tree/trunk/docs",
        "http://github.com/NVIDIA/cuda-python/blob/trunk/README.md",
        "https://example.com/redirect?url=https://github.com/NVIDIA/cuda-python/blob/trunk/README.md",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_remaps_preserve_other_targets(tmp_path, monkeypatch, url):
    monkeypatch.setenv(
        links.METADATA_ENV,
        json.dumps({"repository": links.PUBLIC_REPOSITORY, "defaults": {links.PUBLIC_REPOSITORY: "trunk"}}),
    )
    assert remap(links.repo_link_remaps(tmp_path), url) == url


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_ci_metadata_uses_actual_commit_instead_of_pr_base_or_branch(tmp_path, monkeypatch):
    root = checkout(tmp_path)
    git(
        root,
        "-c",
        "user.name=Link Test",
        "-c",
        "user.email=link-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-m",
        "Test checked-out revision",
    )
    commit = git(root, "rev-parse", "HEAD")
    repository = "Example/toolkit-private"
    metadata = {"repository": repository, "defaults": {repository: "next/toolkit", links.PUBLIC_REPOSITORY: "trunk"}}
    monkeypatch.setenv(links.METADATA_ENV, json.dumps(metadata))
    monkeypatch.setenv("GITHUB_BASE_REF", "12.9.x")
    monkeypatch.setenv("GITHUB_REF_NAME", "123/merge")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    arguments = links.repo_link_remaps(root, rendered=True)
    for source, branch in metadata["defaults"].items():
        url = f"https://github.com/{source}/blob/{branch}/README.md?plain=1#heading"
        assert remap(arguments, url) == f"https://github.com/{repository}/blob/{commit}/README.md?plain=1#heading"


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_cached_head_gives_matching_remote_refresh_instruction(tmp_path, monkeypatch):
    monkeypatch.delenv(links.METADATA_ENV, raising=False)
    root = checkout(tmp_path)
    remote(root, "upstream", "https://github.com/NVIDIA/cuda-python.git")
    with pytest.raises(ValueError, match="git remote set-head upstream --auto"):
        links.repository_defaults(root)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_dangling_cached_head_requires_refresh_instead_of_using_removed_branch(tmp_path, monkeypatch):
    monkeypatch.delenv(links.METADATA_ENV, raising=False)
    root = checkout(tmp_path)
    remote(root, "upstream", "https://github.com/NVIDIA/cuda-python.git")
    git(root, "symbolic-ref", "refs/remotes/upstream/HEAD", "refs/remotes/upstream/removed")
    with pytest.raises(ValueError, match="git remote set-head upstream --auto"):
        links.repository_defaults(root)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_conflicting_cached_heads_fail_instead_of_choosing_arbitrarily(tmp_path, monkeypatch):
    monkeypatch.delenv(links.METADATA_ENV, raising=False)
    root = checkout(tmp_path)
    remote(root, "upstream", "https://github.com/NVIDIA/cuda-python.git", "old")
    remote(root, "public", "git@github.com:NVIDIA/cuda-python.git", "new")
    with pytest.raises(ValueError, match="Cached default branches disagree"):
        links.repository_defaults(root)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_prepare_public_ci_uses_event_default_without_network():
    with patch("checkout_link_defaults.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
        metadata = links.prepare_ci(links.PUBLIC_REPOSITORY, "trunk")
    assert all("ls-remote" not in call.args[0] for call in run.call_args_list)
    assert metadata == {"repository": links.PUBLIC_REPOSITORY, "defaults": {links.PUBLIC_REPOSITORY: "trunk"}}


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_prepare_other_ci_queries_only_public_advertised_head():
    output = "ref: refs/heads/trunk\tHEAD\n" + "a" * 40 + "\tHEAD\n"
    with patch(
        "checkout_link_defaults.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=output)
    ) as run:
        metadata = links.prepare_ci("Example/toolkit-private", "next/toolkit")
    assert metadata["defaults"] == {"Example/toolkit-private": "next/toolkit", links.PUBLIC_REPOSITORY: "trunk"}
    network_calls = [call.args[0] for call in run.call_args_list if "ls-remote" in call.args[0]]
    assert network_calls == [
        [
            "git",
            "ls-remote",
            "--symref",
            "https://github.com/NVIDIA/cuda-python.git",
            "HEAD",
        ]
    ]


@pytest.mark.parametrize(
    "output", ["", "a" * 40 + "\tHEAD\n", "ref: refs/heads/one\tHEAD\nref: refs/heads/two\tHEAD\n"]
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_prepare_rejects_missing_or_ambiguous_public_default(output):
    with (
        patch("checkout_link_defaults.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=output)),
        pytest.raises(ValueError, match="advertised default branch"),
    ):
        links.prepare_ci("Example/toolkit-private", "next/toolkit")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", f"{links.METADATA_ENV} must contain valid JSON"),
        ("[]", f"{links.METADATA_ENV} must contain repository and defaults"),
        ("{}", f"{links.METADATA_ENV} must contain repository and defaults"),
        ('{"repository":null,"defaults":{}}', "OWNER/REPOSITORY"),
        ('{"repository":"NVIDIA/cuda-python","defaults":{}}', "only the checkout and public"),
        ('{"repository":"NVIDIA/cuda-python","defaults":[]}', "only the checkout and public"),
        ('{"repository":"NVIDIA/cuda-python","defaults":{"NVIDIA/cuda-python":""}}', "Git default branch"),
        ('{"repository":"NVIDIA/cuda-python","defaults":{"NVIDIA/cuda-python":null}}', "Git default branch"),
        ('{"repository":"NVIDIA/cuda-python","defaults":{"NVIDIA/cuda-python":"bad..branch"}}', "Git default branch"),
        (
            '{"repository":"NVIDIA/cuda-python","defaults":{"NVIDIA/cuda-python":"bad\\u0000branch"}}',
            "Git default branch",
        ),
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_invalid_ci_metadata_fails_explicitly(tmp_path, monkeypatch, raw, message):
    monkeypatch.setenv(links.METADATA_ENV, raw)
    with pytest.raises(ValueError, match=message):
        links.repository_defaults(tmp_path)


@pytest.mark.parametrize("branch", ["next#toolkit", "next/\u03b2", "next%toolkit", "next$toolkit"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_valid_branch_names_are_url_encoded_before_matching(tmp_path, monkeypatch, branch):
    metadata = {"repository": links.PUBLIC_REPOSITORY, "defaults": {links.PUBLIC_REPOSITORY: branch}}
    monkeypatch.setenv(links.METADATA_ENV, json.dumps(metadata))
    url = f"https://github.com/NVIDIA/cuda-python/blob/{quote(branch, safe='/')}/README.md#heading"
    assert remap(links.repo_link_remaps(tmp_path), url) == f"{tmp_path.as_uri()}/README.md#heading"
