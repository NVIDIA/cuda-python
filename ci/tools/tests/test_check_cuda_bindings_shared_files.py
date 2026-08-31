# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_cuda_bindings_shared_files import DEFAULT_POLICY, REPO_ROOT, main


def write_policy(tmp_path: Path, **updates: object) -> Path:
    policy = {
        "schema_version": 1,
        "roots": ["cuda_bindings", "cuda_bindings_12"],
        "shared_paths": ["shared.py"],
    }
    policy.update(updates)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def write_roots(tmp_path: Path, *, second_content: str = "same") -> None:
    for root, content in (("cuda_bindings", "same"), ("cuda_bindings_12", second_content)):
        path = tmp_path / root / "shared.py"
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5")
def test_matching_shared_files_pass(tmp_path):
    write_roots(tmp_path)

    assert main(["--repo-root", str(tmp_path), "--policy", str(write_policy(tmp_path))]) == 0


@pytest.mark.agent_authored(model="gpt-5")
def test_byte_mismatch_reports_both_roots(tmp_path, capsys):
    write_roots(tmp_path, second_content="different")

    assert main(["--repo-root", str(tmp_path), "--policy", str(write_policy(tmp_path))]) == 1
    error = capsys.readouterr().err
    assert "shared.py: byte mismatch" in error
    assert "cuda_bindings=" in error
    assert "cuda_bindings_12=" in error


@pytest.mark.agent_authored(model="gpt-5")
def test_missing_file_is_reported(tmp_path, capsys):
    (tmp_path / "cuda_bindings").mkdir()
    (tmp_path / "cuda_bindings_12").mkdir()
    (tmp_path / "cuda_bindings" / "shared.py").write_text("same", encoding="utf-8")

    assert main(["--repo-root", str(tmp_path), "--policy", str(write_policy(tmp_path))]) == 1
    assert "shared.py: missing from cuda_bindings_12" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"schema_version": True}, "schema_version must be 1"),
        ({"schema_version": 1.0}, "schema_version must be 1"),
        ({"roots": ["cuda_bindings"]}, "roots must contain at least 2 entries"),
        ({"shared_paths": ["../escape.py"]}, "normalized relative path"),
        ({"shared_paths": ["C:/escape.py"]}, "drive-qualified"),
        ({"shared_paths": ["C:escape.py"]}, "drive-qualified"),
        ({"shared_paths": ["z.py", "a.py"]}, "sorted"),
        ({"unexpected": True}, "unexpected policy keys"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_invalid_policy_is_configuration_error(tmp_path, capsys, updates, message):
    assert main(["--repo-root", str(tmp_path), "--policy", str(write_policy(tmp_path, **updates))]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5")
def test_live_policy_is_clean():
    assert main(["--repo-root", str(REPO_ROOT), "--policy", str(DEFAULT_POLICY)]) == 0


@pytest.mark.agent_authored(model="gpt-5")
def test_symlink_is_rejected(tmp_path, capsys):
    write_roots(tmp_path)
    target = tmp_path / "target.py"
    target.write_text("same", encoding="utf-8")
    link = tmp_path / "cuda_bindings_12" / "shared.py"
    link.unlink()
    link.symlink_to(target)

    assert main(["--repo-root", str(tmp_path), "--policy", str(write_policy(tmp_path))]) == 1
    assert "shared.py: symlink in cuda_bindings_12" in capsys.readouterr().err
