# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_cuda_bindings_shared_files import DEFAULT_POLICY, REPO_ROOT, main

ROOTS = ("cuda_bindings", "cuda_bindings_12")


def write_policy(tmp_path: Path, shared_paths: object = None) -> Path:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_paths": ["shared.py"] if shared_paths is None else shared_paths,
            }
        ),
        encoding="utf-8",
    )
    return policy


def write_roots(tmp_path: Path, *, second_content: str = "same") -> None:
    for index, root in enumerate(ROOTS):
        path = tmp_path / root / "shared.py"
        path.parent.mkdir()
        path.write_text(second_content if index else "same", encoding="utf-8")


def run_check(tmp_path: Path, policy: Path | None = None) -> int:
    return main(["--repo-root", str(tmp_path), "--policy", str(policy or write_policy(tmp_path))])


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_matching_shared_files_pass(tmp_path):
    write_roots(tmp_path)

    assert run_check(tmp_path) == 0


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_mismatch_identifies_both_roots(tmp_path, capsys):
    write_roots(tmp_path, second_content="different")

    assert run_check(tmp_path) == 1
    error = capsys.readouterr().err
    assert "shared.py: byte mismatch" in error
    assert all(f"{root}=" in error for root in ROOTS)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_missing_file_is_reported(tmp_path, capsys):
    write_roots(tmp_path)
    (tmp_path / ROOTS[1] / "shared.py").unlink()

    assert run_check(tmp_path) == 1
    assert f"shared.py: missing from {ROOTS[1]}" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_unsafe_policy_path_is_rejected(tmp_path, capsys):
    write_roots(tmp_path)

    assert run_check(tmp_path, write_policy(tmp_path, ["../escape.py"])) == 2
    assert "normalized relative path" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_symlink_is_rejected(tmp_path, capsys):
    write_roots(tmp_path)
    link = tmp_path / ROOTS[1] / "shared.py"
    link.unlink()
    link.symlink_to(tmp_path / ROOTS[0] / "shared.py")

    assert run_check(tmp_path) == 1
    assert f"shared.py: symlink in {ROOTS[1]}" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_live_policy_is_clean():
    assert main(["--repo-root", str(REPO_ROOT), "--policy", str(DEFAULT_POLICY)]) == 0
