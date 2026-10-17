# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from urllib.parse import parse_qs

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


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_selector_requests_historical_checks_for_duplicates_and_baselines():
    selector = (ROOT / "ci/tools/select-woa-batch").read_text()
    queries = [line.split("?", 1)[1].split('"', 1)[0] for line in selector.splitlines() if "/check-runs?" in line]
    assert len(queries) == 2
    assert all(parse_qs(query)["filter"] == ["all"] for query in queries)
