# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/woa-main-ci.yml").read_text())


@pytest.mark.agent_authored(model="gpt-6")
def test_entry_allows_reusable_and_manual_dispatch_without_cron():
    assert set(WORKFLOW[True]) == {"workflow_call", "workflow_dispatch"}
    assert WORKFLOW[True]["workflow_call"]["inputs"]["public_run_id"]["required"] is True
    assert WORKFLOW[True]["workflow_dispatch"]["inputs"]["public_run_id"]["required"] is False
    assert WORKFLOW["jobs"]["resolve-public-build"]["if"] == (
        "${{ github.repository == 'NVIDIA/cuda-python' && github.ref == 'refs/heads/main' }}"
    )
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False


@pytest.mark.agent_authored(model="gpt-6")
def test_ci_dispatch_requires_successful_builds_on_main_push():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    dispatch = ci["jobs"]["woa-main-dispatch"]
    assert dispatch["needs"] == ["build-linux-64", "build-windows-arm64"]
    for condition in (
        "github.event_name == 'push'",
        "github.ref == 'refs/heads/main'",
        "needs.build-linux-64.result == 'success'",
        "needs.build-windows-arm64.result == 'success'",
    ):
        assert condition in dispatch["if"]
    assert dispatch["uses"] == "./.github/workflows/woa-main-ci.yml"
    assert dispatch["with"]["public_run_id"] == "${{ github.run_id }}"
    steps = WORKFLOW["jobs"]["resolve-public-build"]["steps"]
    assert "Wait for public WoA wheel artifacts" not in {step["name"] for step in steps}
