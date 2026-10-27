# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fail when the running CPython is older than the fix for the eMudhra/emSign CA compromise.

Print ``sys.version`` and compare the interpreter with the first release of its
minor series whose ``ssl`` module rejects the compromised eMudhra intermediate
keys (CVE-2026-115557). Any release at or after that patch release passes.

Release announcement:
https://discuss.python.org/t/python-3-15-2-3-14-10-3-13-18-3-12-16-and-3-11-18-are-out/109678

This file uses only the standard library because it runs on whichever
interpreter the job is about to use, before any package is installed.
"""

from __future__ import annotations

import sys

# Minor series -> first patch release carrying the fix.
FIXED_MICRO = {
    (3, 11): 18,
    (3, 12): 16,
    (3, 13): 18,
    (3, 14): 10,
    (3, 15): 2,
}


def classify(version_info) -> tuple[str, str]:
    """Return ``(status, detail)`` where status is ``"ok"``, ``"warn"`` or ``"fail"``."""
    series = (version_info.major, version_info.minor)
    found = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    if series in FIXED_MICRO:
        fixed = f"{series[0]}.{series[1]}.{FIXED_MICRO[series]}"
        if version_info.releaselevel != "final":
            return (
                "fail",
                f"Python {found} ({version_info.releaselevel}) is a pre-release; use a final release >= {fixed}.",
            )
        if version_info.micro >= FIXED_MICRO[series]:
            return "ok", f"Python {found} includes the fix (first fixed release of this series: {fixed})."
        return (
            "fail",
            f"Python {found} is older than {fixed}, the first release of this series that rejects the compromised eMudhra intermediate CA keys.",
        )

    if series > max(FIXED_MICRO):
        # Newer than every series we know a fix for (for example 3.16 pre-releases).
        return (
            "warn",
            f"Python {found}: no fixed release is known for the {series[0]}.{series[1]} series yet; update FIXED_MICRO once one exists.",
        )

    return (
        "fail",
        f"Python {series[0]}.{series[1]} has no release that rejects the compromised eMudhra intermediate CA keys.",
    )


def main() -> int:
    print(f"sys.version: {sys.version}")
    print(f"sys.executable: {sys.executable}")
    status, detail = classify(sys.version_info)
    if status == "ok":
        print(detail)
        return 0
    if status == "warn":
        print(f"::warning title=Python patch level unverified::{detail}")
        return 0
    print(f"::error title=Unpatched Python interpreter::{detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
