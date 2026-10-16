# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_lychee import repo_link_remaps


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_remaps_use_actual_checkout_commit(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    readme = checkout / "README.md"
    readme.write_text("# Smoke test\n", encoding="utf-8")
    for arguments in (
        ["init", "--quiet"],
        ["add", "README.md"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Smoke test",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=checkout, check=True)  # noqa: S603, S607
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()  # noqa: S607
    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example-org/cuda-python-mirror")
    args = repo_link_remaps(checkout, rendered=True)
    remaps = args[1::2]
    assert remaps == [
        f"^https://github[.]com/NVIDIA/cuda-python/{view}/main/ "
        f"https://github.com/example-org/cuda-python-mirror/{view}/{head}/"
        for view in ("blob", "tree")
    ]
    for remap in remaps:
        pattern, replacement = remap.split()
        view = "blob" if "/blob/" in pattern else "tree"
        url = f"https://github.com/NVIDIA/cuda-python/{view}/main/new-directory"
        assert re.sub(pattern, replacement, url) == (
            f"https://github.com/example-org/cuda-python-mirror/{view}/{head}/new-directory"
        )
        assert re.search(pattern, url.replace("github.com", "githubXcom")) is None
        assert re.search(pattern, url.replace("/main/", "/mainly/")) is None
        assert re.search(pattern, "https://example.com/" + url) is None
