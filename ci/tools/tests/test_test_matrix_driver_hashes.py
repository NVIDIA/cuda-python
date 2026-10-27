# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure custom DRIVER entries in ci/test-matrix.yml carry valid, consistent SHA-256 hashes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

MATRIX_PATH = Path(__file__).resolve().parents[2] / "test-matrix.yml"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREINSTALLED_DRIVERS = {"latest", "earliest"}


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_test_matrix_custom_drivers_have_valid_consistent_sha256() -> None:
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    seen_hashes: dict[tuple[str, str, str, str], str] = {}
    custom_rows = 0

    for platform in ("linux", "windows"):
        for section_name, rows in matrix[platform].items():
            for index, row in enumerate(rows):
                driver = row["DRIVER"]
                if driver in PREINSTALLED_DRIVERS:
                    assert "DRIVER_SHA256" not in row, f"{platform}.{section_name}[{index}] sets DRIVER_SHA256"
                    continue
                custom_rows += 1
                sha256 = row.get("DRIVER_SHA256", "")
                assert SHA256_RE.match(sha256), f"{platform}.{section_name}[{index}] has invalid DRIVER_SHA256"
                key = (platform, str(row["ARCH"]), str(row["GPU"]), str(driver))
                assert seen_hashes.setdefault(key, sha256) == sha256, f"Conflicting DRIVER_SHA256 for {key}"

    assert custom_rows > 0
