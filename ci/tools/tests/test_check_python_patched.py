# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the interpreter patch-level gate used by the harden-pip action."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ACTION_DIR = Path(__file__).resolve().parents[3] / ".github" / "actions" / "harden-pip"
SCRIPT_PATH = ACTION_DIR / "check_python_patched.py"

_spec = importlib.util.spec_from_file_location("check_python_patched", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_python_patched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_python_patched)


def _version(major: int, minor: int, micro: int, releaselevel: str = "final") -> SimpleNamespace:
    return SimpleNamespace(major=major, minor=minor, micro=micro, releaselevel=releaselevel)


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.parametrize(
    ("version", "status"),
    [
        # First fixed release and anything later in the same series passes.
        ((3, 11, 18), "ok"),
        ((3, 11, 19), "ok"),
        ((3, 12, 16), "ok"),
        ((3, 13, 18), "ok"),
        ((3, 14, 10), "ok"),
        ((3, 14, 11), "ok"),
        ((3, 15, 2), "ok"),
        ((3, 15, 3), "ok"),
        # The patch release before the fix fails.
        ((3, 11, 17), "fail"),
        ((3, 12, 15), "fail"),
        ((3, 13, 17), "fail"),
        ((3, 14, 9), "fail"),
        ((3, 15, 1), "fail"),
        ((3, 15, 0), "fail"),
        # Series without any fixed release.
        ((3, 10, 99), "fail"),
        ((3, 9, 25), "fail"),
    ],
)
def test_classify_final_releases(version: tuple[int, int, int], status: str) -> None:
    assert check_python_patched.classify(_version(*version))[0] == status


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_classify_prerelease_of_fixed_series_fails() -> None:
    assert check_python_patched.classify(_version(3, 15, 2, "candidate"))[0] == "fail"


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_classify_newer_series_warns() -> None:
    assert check_python_patched.classify(_version(3, 16, 0, "alpha"))[0] == "warn"


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_script_prints_sys_version_and_matches_classification() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True, check=False
    )
    assert f"sys.version: {sys.version}" in result.stdout
    expected_status, _ = check_python_patched.classify(sys.version_info)
    assert (result.returncode == 1) == (expected_status == "fail")


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_harden_pip_action_runs_the_check_before_installing_pip() -> None:
    text = (ACTION_DIR / "action.yml").read_text(encoding="utf-8")
    check = text.index("check_python_patched.py")
    install = text.index("pip install")
    assert check < install
