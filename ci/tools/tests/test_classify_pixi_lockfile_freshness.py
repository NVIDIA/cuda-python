# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from classify_pixi_lockfile_freshness import Classification, LockfileCheck, classify


def _check(*, stale: bool, original: str, repaired: str) -> LockfileCheck:
    return LockfileCheck(stale=stale, original_blob=original, repaired_blob=repaired)


@pytest.mark.agent_authored(model="gpt-5")
def test_fresh_base_is_pr_induced():
    candidate = _check(stale=True, original="candidate-old", repaired="candidate-new")
    base = _check(stale=False, original="base-current", repaired="base-current")

    assert classify(candidate, base) is Classification.PR_INDUCED


@pytest.mark.agent_authored(model="gpt-5")
def test_workspace_missing_from_base_is_pr_induced():
    candidate = _check(stale=True, original="candidate-old", repaired="candidate-new")

    assert classify(candidate, None) is Classification.PR_INDUCED


@pytest.mark.agent_authored(model="gpt-5")
def test_matching_stale_blobs_are_base_maintenance():
    candidate = _check(stale=True, original="shared-old", repaired="shared-new")
    base = _check(stale=True, original="shared-old", repaired="shared-new")

    assert classify(candidate, base) is Classification.BASE_MAINTENANCE


@pytest.mark.parametrize(
    ("base_original", "base_repaired"),
    [
        ("base-old", "shared-new"),
        ("shared-old", "base-new"),
        ("base-old", "base-new"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_different_stale_blobs_are_mixed_or_ambiguous(base_original, base_repaired):
    candidate = _check(stale=True, original="shared-old", repaired="shared-new")
    base = _check(stale=True, original=base_original, repaired=base_repaired)

    assert classify(candidate, base) is Classification.MIXED


@pytest.mark.agent_authored(model="gpt-5")
def test_fresh_candidate_is_rejected():
    candidate = _check(stale=False, original="current", repaired="current")

    with pytest.raises(ValueError, match="candidate lockfile must be stale"):
        classify(candidate, None)
