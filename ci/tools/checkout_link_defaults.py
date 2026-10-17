# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve checkout-aware repository links without network access in the hook.

CI prepares repository metadata once. Developer checkouts use cached remote
HEADs; ``lychee.repository`` selects the checkout's repository explicitly, or
the conventional ``upstream`` then ``origin`` remote supplies its identity.
Other remotes are used only to find metadata for those selected repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

PUBLIC_REPOSITORY = "NVIDIA/cuda-python"
METADATA_ENV = "LYCHEE_REPOSITORY_LINKS"


def _git(root: Path, *arguments: str, optional: bool = False) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *arguments],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    if optional and result.returncode == 1:
        return ""
    result.check_returncode()
    return result.stdout.strip()


def _repository(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Expected a GitHub OWNER/REPOSITORY identity, got {value!r}")
    return PUBLIC_REPOSITORY if value.casefold() == PUBLIC_REPOSITORY.casefold() else value


def _branch(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("-")
        or value == "HEAD"
        or re.search(r"[\x00-\x20\x7f]", value)
    ):
        raise ValueError(f"Expected a Git default branch name, got {value!r}")
    result = subprocess.run(  # noqa: S603
        ["git", "check-ref-format", f"refs/heads/{value}"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"Expected a Git default branch name, got {value!r}")
    return value


def _remote_repository(url: str) -> str | None:
    if url.startswith("git@github.com:"):
        path = url.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
            return None
        path = parsed.path.lstrip("/")
    return _repository(path.rstrip("/").removesuffix(".git"))


def _remotes(root: Path) -> dict[str, str]:
    remotes = {}
    for line in _git(root, "config", "--get-regexp", r"^remote\..*\.url$", optional=True).splitlines():
        key, url = line.split(maxsplit=1)
        if repository := _remote_repository(url):
            remotes[key.removeprefix("remote.").removesuffix(".url")] = repository
    return remotes


def _cached_default(root: Path, repository: str, remotes: dict[str, str]) -> str:
    names = [name for name, value in remotes.items() if value.casefold() == repository.casefold()]
    if not names:
        raise ValueError(
            f"No GitHub remote is configured for {repository}. Add and fetch one, then run "
            "git remote set-head <remote> --auto."
        )
    branches = set()
    for name in names:
        target = _git(root, "symbolic-ref", "--quiet", f"refs/remotes/{name}/HEAD", optional=True)
        prefix = f"refs/remotes/{name}/"
        if (
            target.startswith(prefix)
            and target != prefix + "HEAD"
            and _git(root, "rev-parse", "--verify", "--quiet", target, optional=True)
        ):
            branches.add(target.removeprefix(prefix))
    instructions = "; ".join(f"git remote set-head {name} --auto" for name in names)
    if not branches:
        raise ValueError(
            f"The default branch for {repository} is not cached. Run: {instructions}. "
            "Fetch the default branch first if Git requests it."
        )
    if len(branches) != 1:
        raise ValueError(f"Cached default branches disagree for {repository}. Refresh them: {instructions}.")
    return _branch(branches.pop())


def repository_defaults(root: Path) -> tuple[str, dict[str, str]]:
    """Return the checkout repository and only its/public default-branch targets."""
    if raw := os.environ.get(METADATA_ENV):
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{METADATA_ENV} must contain valid JSON") from error
        if not isinstance(metadata, dict) or set(metadata) != {"repository", "defaults"}:
            raise ValueError(f"{METADATA_ENV} must contain repository and defaults")
        repository = _repository(metadata["repository"])
        defaults = metadata["defaults"]
        if not isinstance(defaults, dict) or set(defaults) != {PUBLIC_REPOSITORY, repository}:
            raise ValueError(f"{METADATA_ENV} must describe only the checkout and public repositories")
        for branch in defaults.values():
            _branch(branch)
        return repository, defaults

    remotes = _remotes(root)
    explicit = _git(root, "config", "--get", "lychee.repository", optional=True)
    repository = _repository(explicit) if explicit else remotes.get("upstream", remotes.get("origin"))
    if repository is None:
        raise ValueError(
            "Cannot identify this checkout's GitHub repository. Configure an upstream/origin remote, "
            "or run git config lychee.repository OWNER/REPOSITORY."
        )
    defaults = {
        value: _cached_default(root, value, remotes) for value in dict.fromkeys((PUBLIC_REPOSITORY, repository))
    }
    return repository, defaults


def repo_link_remaps(root: Path, *, rendered: bool = False) -> list[str]:
    """Map default-branch URLs to local sources or GitHub's checked-out commit."""
    repository, defaults = repository_defaults(root)
    if rendered:
        commit = _git(root, "rev-parse", "HEAD")
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        prefix = f"{server}/{repository}"
    else:
        prefix = root.resolve().as_uri()
    arguments = []
    for source, branch in defaults.items():
        for view in ("blob", "tree"):
            replacement = f"{prefix}/{view}/{commit}/" if rendered else f"{prefix}/"
            # Git branch names may contain URL delimiters or non-ASCII text.
            pattern = rf"^https://github[.]com/{re.escape(source)}/{view}/{re.escape(quote(branch, safe='/'))}/"
            arguments.extend(["--remap", f"{pattern} {replacement}"])
    return arguments


def prepare_ci(repository: str, default_branch: str) -> dict[str, object]:
    """Prepare event metadata, querying public HEAD once for other repositories."""
    repository = _repository(repository)
    default_branch = _branch(default_branch)
    defaults = {repository: default_branch}
    if repository != PUBLIC_REPOSITORY:
        result = subprocess.run(  # noqa: S603
            ["git", "ls-remote", "--symref", f"https://github.com/{PUBLIC_REPOSITORY}.git", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        branches = re.findall(r"^ref: refs/heads/(.+)\tHEAD$", result.stdout, flags=re.MULTILINE)
        if len(branches) != 1:
            raise ValueError(f"Could not identify the advertised default branch for {PUBLIC_REPOSITORY}")
        defaults[PUBLIC_REPOSITORY] = _branch(branches[0])
    return {"repository": repository, "defaults": defaults}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--default-branch", required=True)
    parser.add_argument("--github-env", type=Path, required=True)
    arguments = parser.parse_args()
    metadata = prepare_ci(arguments.repository, arguments.default_branch)
    with arguments.github_env.open("a", encoding="utf-8") as stream:
        stream.write(f"{METADATA_ENV}={json.dumps(metadata, separators=(',', ':'))}\n")


if __name__ == "__main__":
    main()
