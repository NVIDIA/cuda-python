# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure custom DRIVER entries in ci/test-matrix.yml carry verified SHA-256 hashes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = REPO_ROOT / "ci" / "test-matrix.yml"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREINSTALLED_DRIVERS = {"latest", "earliest"}
WINDOWS_DATA_CENTER_GPUS = {"a100", "h100", "l4", "t4", "v100", "rtxa6000", "rtx6000ada"}
WINDOWS_DESKTOP_GPUS = {"rtx2080", "rtx4090", "rtxpro6000"}


def _load_matrix() -> dict[str, object]:
    return yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))


def _iter_rows(matrix: dict[str, object]):
    for platform in ("linux", "windows"):
        sections = matrix.get(platform, {})
        assert isinstance(sections, dict)
        for section_name, rows in sections.items():
            assert isinstance(rows, list)
            for index, row in enumerate(rows):
                assert isinstance(row, dict)
                yield platform, section_name, index, row


def _installer_identity(platform: str, row: dict[str, object]) -> tuple[str, str, str]:
    arch = str(row["ARCH"])
    driver = str(row["DRIVER"])
    if platform == "linux":
        return (platform, arch, driver)
    gpu = str(row["GPU"])
    if gpu in WINDOWS_DATA_CENTER_GPUS:
        variant = "data-center"
    elif gpu in WINDOWS_DESKTOP_GPUS:
        variant = "desktop"
    else:
        raise AssertionError(f"Unknown Windows GPU '{gpu}' in ci/test-matrix.yml")
    return (f"{platform}-{variant}", arch, driver)


def validate_matrix_driver_hashes(matrix: dict[str, object]) -> None:
    seen_hashes: dict[tuple[str, str, str], str] = {}
    custom_rows = 0

    for platform, section_name, index, row in _iter_rows(matrix):
        driver = row.get("DRIVER")
        assert isinstance(driver, str) and driver, f"{platform}.{section_name}[{index}] is missing a non-empty DRIVER"
        if driver in PREINSTALLED_DRIVERS:
            assert "DRIVER_SHA256" not in row, (
                f"{platform}.{section_name}[{index}] sets DRIVER={driver!r} and must not set DRIVER_SHA256"
            )
            continue

        custom_rows += 1
        sha256 = row.get("DRIVER_SHA256")
        assert isinstance(sha256, str) and SHA256_RE.match(sha256), (
            f"{platform}.{section_name}[{index}] (DRIVER={driver!r}) must set "
            "DRIVER_SHA256 to a 64-character lowercase hex SHA-256 digest"
        )

        identity = _installer_identity(platform, row)
        previous = seen_hashes.setdefault(identity, sha256)
        assert sha256 == previous, f"Inconsistent DRIVER_SHA256 for {identity}: {sha256!r} != {previous!r}"

    assert custom_rows > 0, "Expected at least one custom DRIVER row in ci/test-matrix.yml"


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_test_matrix_custom_drivers_have_valid_consistent_sha256() -> None:
    validate_matrix_driver_hashes(_load_matrix())


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_missing_or_invalid_driver_sha256_is_rejected() -> None:
    matrix = _load_matrix()
    for _platform, _section_name, _index, row in _iter_rows(matrix):
        if row["DRIVER"] not in PREINSTALLED_DRIVERS:
            target_row = row
            break
    else:
        pytest.fail("No custom DRIVER row found in ci/test-matrix.yml")

    original = target_row.pop("DRIVER_SHA256")
    with pytest.raises(AssertionError, match="must set DRIVER_SHA256"):
        validate_matrix_driver_hashes(matrix)

    target_row["DRIVER_SHA256"] = "UPPERCASE" + ("0" * 55)
    with pytest.raises(AssertionError, match="64-character lowercase hex"):
        validate_matrix_driver_hashes(matrix)

    target_row["DRIVER_SHA256"] = original


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_conflicting_driver_sha256_for_same_installer_is_rejected() -> None:
    matrix = _load_matrix()
    linux_pr_rows = matrix["linux"]["pull-request"]
    custom_rows = [row for row in linux_pr_rows if row["DRIVER"] == "610.43.02"]
    assert len(custom_rows) >= 2

    custom_rows[1]["DRIVER_SHA256"] = "0" * 64
    with pytest.raises(AssertionError, match="Inconsistent DRIVER_SHA256"):
        validate_matrix_driver_hashes(matrix)
