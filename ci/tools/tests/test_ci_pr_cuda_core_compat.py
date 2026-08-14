# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for PR CI wiring of nightly-cuda-core (#2449)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_NIGHTLY_YML = REPO_ROOT / ".github" / "workflows" / "ci-nightly.yml"


def _job_block(workflow_text: str, job_id: str) -> str:
    """Return the YAML block for a top-level job (through the next job header)."""
    pattern = rf"(?ms)^  {re.escape(job_id)}:\n.*?(?=^  \w|^jobs:|\Z)"
    match = re.search(pattern, workflow_text)
    assert match, f"job {job_id!r} not found"
    return match.group(0)


@pytest.fixture(scope="module")
def ci_yml_text() -> str:
    return CI_YML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def test_cuda_core_job(ci_yml_text: str) -> str:
    return _job_block(ci_yml_text, "test-cuda-core-linux-64")


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_pr_ci_defines_released_cuda_core_job(test_cuda_core_job: str) -> None:
    assert "test-mode: nightly-cuda-core" in test_cuda_core_job
    assert "matrix_filter: 'map(select(.ENV.MODE == \"nightly-cuda-core\"))'" in test_cuda_core_job
    assert "build-type: nightly" in test_cuda_core_job
    assert "host-platform: linux-64" in test_cuda_core_job


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_pr_cuda_core_job_uses_pr_artifact_build(test_cuda_core_job: str) -> None:
    # PR CI must consume wheels from this run, not a pinned main run-id/sha.
    assert "run-id:" not in test_cuda_core_job
    assert "sha:" not in test_cuda_core_job
    assert "build-linux-64" in test_cuda_core_job


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_pr_cuda_core_job_gated_on_bindings_changes(test_cuda_core_job: str) -> None:
    assert "detect-changes.outputs.test_bindings" in test_cuda_core_job
    assert "should-skip.outputs.doc-only" in test_cuda_core_job
    assert "should-skip.outputs.skip" in test_cuda_core_job


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_pr_cuda_core_job_skips_standard_core_and_meta_tests(test_cuda_core_job: str) -> None:
    # pathfinder/bindings downloads stay enabled (defaults) so nightly-cuda-core
    # can install PR-built wheels; only skip standard cuda-core/meta pytest.
    assert "test-core: false" in test_cuda_core_job
    assert "test-python: false" in test_cuda_core_job
    assert "test-bindings: false" not in test_cuda_core_job
    assert "test-pathfinder: false" not in test_cuda_core_job


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_checks_aggregator_tracks_cuda_core_job(ci_yml_text: str) -> None:
    checks = _job_block(ci_yml_text, "checks")
    assert "test-cuda-core-linux-64" in checks
    assert 'check_result "test-cuda-core-linux-64"' in checks
    assert "detect-changes.outputs.test_bindings" in checks


@pytest.mark.agent_authored(model="composer-2.5-fast")
def test_nightly_still_defines_cuda_core_jobs() -> None:
    nightly = CI_NIGHTLY_YML.read_text(encoding="utf-8")
    assert _job_block(nightly, "test-cuda-core-linux-64")
    assert _job_block(nightly, "test-cuda-core-windows")
