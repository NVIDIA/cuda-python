# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_action_prepares_remaps_for_actual_checkout_commit(tmp_path):
    action = yaml.safe_load((ROOT / ".github/actions/check-doc-links/action.yml").read_text(encoding="utf-8"))
    prepare = next(step for step in action["runs"]["steps"] if step.get("id") == "inputs")
    checkout = tmp_path / "checkout with spaces"
    helper = checkout / "ci/tools/prepare_lychee_inputs.py"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "ci/tools/prepare_lychee_inputs.py", helper)
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
    output = tmp_path / "action-output"
    input_list = checkout / "lychee-authored-files.txt"
    env = dict(
        os.environ,
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        KIND="authored",
        GITHUB_WORKSPACE=str(checkout),
        RUNNER_TEMP=str(tmp_path / "runner temp"),
        LYCHEE_VERSION="v0.24.2",
        POLICY_HASH="test-policy",
        GITHUB_OUTPUT=str(output),
        LYCHEE_ARGS=f'--files-from "{input_list}" --cache',
        GITHUB_RUN_ID="123",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_SHA="0" * 40,
        GITHUB_SERVER_URL="https://github.com",
        GITHUB_REPOSITORY="example-org/cuda-python-mirror",
    )

    subprocess.run(  # noqa: S603
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", prepare["run"]],  # noqa: S607
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    args = shlex.split(outputs["args"])
    assert args[:3] == ["--files-from", str(input_list), "--cache"]
    remaps = [args[index + 1] for index, argument in enumerate(args) if argument == "--remap"]
    assert remaps == [
        f"^https://github[.]com/NVIDIA/cuda-python/{view}/main/ "
        f"https://github.com/example-org/cuda-python-mirror/{view}/{head}/"
        for view in ("blob", "tree")
    ]
    assert input_list.read_text(encoding="utf-8") == f"{readme}\n"
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
