# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml


@pytest.mark.agent_authored(model="GPT-6")
def test_release_publishers_do_not_execute_repository_or_dependency_code():
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/release.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    docs = yaml.safe_load(workflow.with_name("build-docs.yml").read_text(encoding="utf-8"))["jobs"]["build"]
    for builder in (jobs["prepare-artifacts"], docs):
        assert "write" not in builder["permissions"].values()
    assert "environment" not in jobs["prepare-artifacts"]

    # These jobs can request the trusted-publishing identity. A venv would not
    # isolate dependency installation or repository code from that authority.
    allowed_actions = {"actions/download-artifact", "pypa/gh-action-pypi-publish"}
    for name in ("publish-testpypi", "publish-pypi"):
        for step in jobs[name]["steps"]:
            assert "run" not in step
            assert step["uses"].split("@", 1)[0] in allowed_actions
