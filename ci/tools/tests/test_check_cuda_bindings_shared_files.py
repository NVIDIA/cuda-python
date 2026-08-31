# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_cuda_bindings_shared_files import DEFAULT_POLICY, REPO_ROOT, main

DEFAULT_ROOTS = ("cuda_bindings", "cuda_bindings_12")


def write_policy(tmp_path: Path, **updates: object) -> Path:
    policy = {
        "schema_version": 1,
        "shared_paths": ["shared.py"],
    }
    policy.update(updates)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def write_config(
    tmp_path: Path,
    *,
    public_roots: tuple[str, ...] = DEFAULT_ROOTS,
    unreleased_roots: tuple[str, ...] = (),
) -> Path:
    public_specs = (
        ("current", "13.3"),
        ("maintenance-12", "12.9"),
        ("maintenance-11", "11.8"),
    )
    unreleased_specs = (("unreleased-14", "14.0"), ("unreleased-15", "15.0"))
    assert len(public_roots) <= len(public_specs)
    assert len(unreleased_roots) <= len(unreleased_specs)

    lines = {}
    for (line_id, ctk_target), source_dir in zip(public_specs[: len(public_roots)], public_roots, strict=True):
        lines[line_id] = {
            "source_dir": source_dir,
            "ctk_target": ctk_target,
            "toolkit_version": f"{ctk_target}.0",
            "toolkit_channel": "stable",
            "tag_series": f"v{ctk_target}.",
            "allow_alpha_beta_tags": True,
        }
    for (line_id, ctk_target), source_dir in zip(
        unreleased_specs[: len(unreleased_roots)], unreleased_roots, strict=True
    ):
        lines[line_id] = {
            "source_dir": source_dir,
            "ctk_target": ctk_target,
            "toolkit_version": f"{ctk_target}.0",
            "toolkit_channel": "prerelease",
            "tag_series": f"v{ctk_target}.",
            "allow_alpha_beta_tags": True,
        }

    config = {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "lines": lines,
                "roles": {
                    "current": public_specs[0][0],
                    "maintenance": [line_id for line_id, _ in public_specs[1 : len(public_roots)]],
                    "unreleased": [line_id for line_id, _ in unreleased_specs[: len(unreleased_roots)]],
                },
            }
        },
    }
    path = tmp_path / "versions.yml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_roots(
    tmp_path: Path,
    *,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    second_content: str = "same",
) -> None:
    for index, root in enumerate(roots):
        content = second_content if index == 1 else "same"
        path = tmp_path / root / "shared.py"
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")


def run_check(tmp_path: Path, *, policy: Path | None = None, config: Path | None = None) -> int:
    return main(
        [
            "--repo-root",
            str(tmp_path),
            "--policy",
            str(policy or write_policy(tmp_path)),
            "--config",
            str(config or write_config(tmp_path)),
        ]
    )


@pytest.mark.agent_authored(model="gpt-5")
def test_matching_shared_files_pass(tmp_path):
    write_roots(tmp_path)

    assert run_check(tmp_path) == 0


@pytest.mark.agent_authored(model="gpt-5")
def test_byte_mismatch_reports_both_roots(tmp_path, capsys):
    write_roots(tmp_path, second_content="different")

    assert run_check(tmp_path) == 1
    error = capsys.readouterr().err
    assert "shared.py: byte mismatch" in error
    assert "cuda_bindings=" in error
    assert "cuda_bindings_12=" in error


@pytest.mark.agent_authored(model="gpt-5")
def test_missing_file_is_reported(tmp_path, capsys):
    (tmp_path / "cuda_bindings").mkdir()
    (tmp_path / "cuda_bindings_12").mkdir()
    (tmp_path / "cuda_bindings" / "shared.py").write_text("same", encoding="utf-8")

    assert run_check(tmp_path) == 1
    assert "shared.py: missing from cuda_bindings_12" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"schema_version": True}, "schema_version must be 1"),
        ({"schema_version": 1.0}, "schema_version must be 1"),
        ({"shared_paths": []}, "shared_paths must contain at least 1 entries"),
        ({"shared_paths": ["../escape.py"]}, "normalized relative path"),
        ({"shared_paths": ["C:/escape.py"]}, "drive-qualified"),
        ({"shared_paths": ["C:escape.py"]}, "drive-qualified"),
        ({"shared_paths": ["z.py", "a.py"]}, "sorted"),
        ({"unexpected": True}, "unexpected policy keys"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_invalid_policy_is_configuration_error(tmp_path, capsys, updates, message):
    assert run_check(tmp_path, policy=write_policy(tmp_path, **updates)) == 2
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

    assert run_check(tmp_path) == 1
    assert "shared.py: symlink in cuda_bindings_12" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_all_registry_public_roots_are_checked(tmp_path, capsys):
    roots = ("cuda_bindings_13", "cuda_bindings_12", "cuda_bindings_11")
    write_roots(tmp_path, roots=roots[:2])

    assert run_check(tmp_path, config=write_config(tmp_path, public_roots=roots)) == 1
    assert "bindings root is missing or not a directory: cuda_bindings_11" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_unreleased_registry_roots_are_not_checked(tmp_path):
    write_roots(tmp_path)

    config = write_config(tmp_path, unreleased_roots=("cuda_bindings_13_4",))
    assert run_check(tmp_path, config=config) == 0


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_invalid_registry_is_configuration_error(tmp_path, capsys):
    config = tmp_path / "versions.yml"
    config.write_text("{}", encoding="utf-8")

    assert run_check(tmp_path, config=config) == 2
    assert "invalid CUDA bindings release-line registry" in capsys.readouterr().err
