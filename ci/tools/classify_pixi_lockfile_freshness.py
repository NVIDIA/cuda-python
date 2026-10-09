# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify a stale candidate Pixi lockfile relative to its PR base."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum


class Classification(str, Enum):
    """Possible attributions for a stale candidate lockfile."""

    PR_INDUCED = "pr-induced"
    BASE_MAINTENANCE = "base-maintenance"
    MIXED = "mixed-or-ambiguous"


@dataclass(frozen=True)
class LockfileCheck:
    """Original and post-check Git blobs for one lockfile."""

    stale: bool
    original_blob: str
    repaired_blob: str

    def __post_init__(self) -> None:
        if not self.original_blob or not self.repaired_blob:
            raise ValueError("lockfile blob hashes must be non-empty")
        if not self.stale and self.original_blob != self.repaired_blob:
            raise ValueError("a fresh lockfile cannot have different original and repaired blobs")


def classify(candidate: LockfileCheck, base: LockfileCheck | None) -> Classification:
    """Attribute a stale candidate lockfile using its base check, if any."""
    if not candidate.stale:
        raise ValueError("candidate lockfile must be stale")
    if base is None or not base.stale:
        return Classification.PR_INDUCED
    if base.original_blob == candidate.original_blob and base.repaired_blob == candidate.repaired_blob:
        return Classification.BASE_MAINTENANCE
    return Classification.MIXED


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-original", required=True)
    parser.add_argument("--candidate-repaired", required=True)
    parser.add_argument("--base-result", choices=("missing", "fresh", "stale"), required=True)
    parser.add_argument("--base-original")
    parser.add_argument("--base-repaired")
    return parser.parse_args()


def main() -> int:
    """Print the classification for workflow consumption."""
    args = _parse_args()
    candidate = LockfileCheck(
        stale=True,
        original_blob=args.candidate_original,
        repaired_blob=args.candidate_repaired,
    )

    if args.base_result == "missing":
        if args.base_original or args.base_repaired:
            raise ValueError("a missing base lockfile cannot have blob hashes")
        base = None
    else:
        if not args.base_original or not args.base_repaired:
            raise ValueError("base blob hashes are required for a checked base lockfile")
        base = LockfileCheck(
            stale=args.base_result == "stale",
            original_blob=args.base_original,
            repaired_blob=args.base_repaired,
        )

    print(classify(candidate, base).value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
