# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check public main source links against the proposed checkout.

Authored checks use local files without consulting Git metadata. Rendered
checks use the tested public commit for GitHub's fragment handling; other
checkouts validate local source paths and leave GitHub-specific fragments
for their eventual public review.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

PUBLIC_REPOSITORY = "NVIDIA/cuda-python"
SOURCE_URL = r"^https://github[.]com/NVIDIA/cuda-python/(blob|tree)/main/"


def _git(root: Path, *arguments: str, optional: bool = False) -> str | None:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *arguments],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    if optional and result.returncode == 1:
        return None
    result.check_returncode()
    return result.stdout.strip()


def _repository(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Expected a GitHub OWNER/REPOSITORY identity")
    return PUBLIC_REPOSITORY if value.casefold() == PUBLIC_REPOSITORY.casefold() else value


def _remote_repository(url: str) -> str | None:
    if url.startswith("git@github.com:"):
        path = url.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
            return None
        path = parsed.path.lstrip("/")
    return _repository(path.rstrip("/").removesuffix(".git"))


def checkout_repository(root: Path) -> str:
    """Identify a rendered checkout without querying remote default branches."""
    if "GITHUB_REPOSITORY" in os.environ:
        return _repository(os.environ["GITHUB_REPOSITORY"])
    explicit = _git(root, "config", "--get", "lychee.repository", optional=True)
    if explicit is not None:
        return _repository(explicit)
    for name in ("upstream", "origin"):
        url = _git(root, "config", "--get", f"remote.{name}.url", optional=True)
        if url is not None and (repository := _remote_repository(url)) is not None:
            return repository
    raise ValueError(
        "Cannot identify this checkout's GitHub repository. Configure an upstream/origin remote, "
        "or run git config lychee.repository OWNER/REPOSITORY."
    )


def github_source_ref(root: Path) -> str:
    """Use the tested public commit, or main for links checked against local files."""
    if checkout_repository(root) != PUBLIC_REPOSITORY:
        return "main"
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise ValueError("Git did not return a full checkout commit ID")
    return commit


def repo_link_remaps(root: Path, *, rendered: bool = False) -> list[str]:
    """Remap only canonical public main source URLs; keep other targets literal."""
    local = root.resolve().as_uri() + "/"
    if not rendered:
        return ["--remap", f"{SOURCE_URL} {local}"]
    ref = github_source_ref(root)
    if ref != "main":
        return ["--remap", f"{SOURCE_URL} https://github.com/{PUBLIC_REPOSITORY}/$1/{ref}/"]
    # GitHub source fragments do not describe local files. Omit them only on
    # these source links; generated HTML keeps its ordinary full anchor checks.
    pattern = SOURCE_URL.replace("(blob|tree)", "(?:blob|tree)") + r"([^#]*)(?:#.*)?$"
    return ["--remap", f"{pattern} {local}$1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-ref", action="store_true", required=True)
    parser.parse_args()
    print(github_source_ref(Path(__file__).resolve().parents[2]))


if __name__ == "__main__":
    main()
