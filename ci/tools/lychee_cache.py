# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe the URL-result caches shared by pre-commit and documentation CI."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_FILES = (
    ".pre-commit-config.yaml",
    "lychee.toml",
    "ci/tools/run_lychee.py",
    "ci/tools/checkout_link_defaults.py",
    "ci/tools/run_lychee_precommit.py",
    "ci/tools/retry_lychee.py",
    "ci/tools/prepare_lychee_inputs.py",
    "ci/tools/lychee_cache.py",
)


def cache_prefix(kind: str, root: Path = ROOT) -> str:
    # The hook configuration includes the native wheel pin. Separate kinds
    # prevent authored successes from bypassing rendered fragment checks.
    digest = hashlib.sha256()
    for relative in POLICY_FILES:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes() + b"\0")
    return f"lychee-precommit-v1-{kind}-{digest.hexdigest()[:24]}-baseline-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("authored", "rendered"))
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    prefix = cache_prefix(args.kind)
    cache_path = (
        Path(".lycheecache")
        if args.kind == "authored"
        else Path(os.environ.get("LYCHEE_CACHE_DIR", str(ROOT / ".lychee-rendered"))) / ".lycheecache"
    )
    outputs = {
        "prefix": prefix,
        "key": f"{prefix}{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}",
        "path": str(cache_path),
    }
    with args.github_output.open("a", encoding="utf-8") as stream:
        for name, value in outputs.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"Cache output {name} contains a newline")
            stream.write(f"{name}={value}\n")


if __name__ == "__main__":
    main()
