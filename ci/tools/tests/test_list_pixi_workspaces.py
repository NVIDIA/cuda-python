# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import list_pixi_workspaces


def _make_repo(root: Path, manifests: list[str], *, lockfiles: list[str] | None = None) -> None:
    """Create a git repo tracking the given pixi manifests and their lockfiles."""
    if lockfiles is None:
        lockfiles = [str(Path(m).parent / "pixi.lock") for m in manifests]
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603,S607
    for rel in [*manifests, *lockfiles]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)  # noqa: S603,S607


@pytest.mark.agent_authored(model="claude-opus-5")
def test_discover_this_repo():
    workspaces = list_pixi_workspaces.discover()
    by_id = {w["id"]: w for w in workspaces}
    assert by_id["root"] == {
        "id": "root",
        "branch_key": list_pixi_workspaces._branch_key("root"),
        "manifest": ".",
        "lockfile": "pixi.lock",
    }
    for workspace in workspaces:
        assert (list_pixi_workspaces.ROOT / workspace["manifest"] / "pixi.toml").is_file()
        assert (list_pixi_workspaces.ROOT / workspace["lockfile"]).is_file()


@pytest.mark.agent_authored(model="claude-opus-5")
def test_discover_reports_root_and_nested(tmp_path, monkeypatch):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml", "benchmarks/cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    assert list_pixi_workspaces.discover() == [
        {
            "id": "root",
            "branch_key": list_pixi_workspaces._branch_key("root"),
            "manifest": ".",
            "lockfile": "pixi.lock",
        },
        {
            "id": "benchmarks/cuda_core",
            "branch_key": list_pixi_workspaces._branch_key("benchmarks/cuda_core"),
            "manifest": "benchmarks/cuda_core",
            "lockfile": "benchmarks/cuda_core/pixi.lock",
        },
        {
            "id": "cuda_core",
            "branch_key": list_pixi_workspaces._branch_key("cuda_core"),
            "manifest": "cuda_core",
            "lockfile": "cuda_core/pixi.lock",
        },
    ]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_discover_ignores_untracked_manifest(tmp_path, monkeypatch):
    _make_repo(tmp_path, ["pixi.toml"])
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "pixi.toml").write_text("", encoding="utf-8")
    (tmp_path / "scratch" / "pixi.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    assert [w["id"] for w in list_pixi_workspaces.discover()] == ["root"]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_discover_rejects_manifest_without_lockfile(tmp_path, monkeypatch):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"], lockfiles=["pixi.lock"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match=r"cuda_core/pixi.toml has no committed cuda_core/pixi.lock"):
        list_pixi_workspaces.discover()


@pytest.mark.agent_authored(model="claude-opus-5")
def test_discover_rejects_repo_without_manifests(tmp_path, monkeypatch):
    _make_repo(tmp_path, [], lockfiles=[])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="no tracked pixi.toml"):
        list_pixi_workspaces.discover()


@pytest.mark.parametrize("reserved_id", ["all", "root"])
@pytest.mark.agent_authored(model="gpt-5.6")
def test_discover_rejects_reserved_nested_workspace_id(tmp_path, monkeypatch, reserved_id):
    _make_repo(tmp_path, ["pixi.toml", f"{reserved_id}/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match=rf"reserved workspace id '{reserved_id}'"):
        list_pixi_workspaces.discover()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_discover_generates_distinct_ref_safe_branch_keys_for_nested_ids(tmp_path, monkeypatch):
    _make_repo(tmp_path, ["foo/pixi.toml", "foo/bar/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)

    workspaces = list_pixi_workspaces.discover()
    keys = [workspace["branch_key"] for workspace in workspaces]

    assert len(keys) == len(set(keys))
    for key in keys:
        subprocess.run(  # noqa: S603 - fixed command and generated branch name.
            ["git", "check-ref-format", "--branch", f"ci/pixi-lock-refresh/{key}"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.agent_authored(model="gpt-5.6")
def test_discover_rejects_branch_key_collision(tmp_path, monkeypatch):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    monkeypatch.setattr(list_pixi_workspaces, "_branch_key", lambda _workspace_id: "collision")

    with pytest.raises(RuntimeError, match="duplicate refresh branch key 'collision'"):
        list_pixi_workspaces.discover()


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_selects_one_workspace(tmp_path, monkeypatch, capsys):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["list_pixi_workspaces.py", "--select", "cuda_core"])
    assert list_pixi_workspaces.main() == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "id": "cuda_core",
            "branch_key": list_pixi_workspaces._branch_key("cuda_core"),
            "manifest": "cuda_core",
            "lockfile": "cuda_core/pixi.lock",
        }
    ]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_rejects_unknown_workspace(tmp_path, monkeypatch, capsys):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["list_pixi_workspaces.py", "--select", "cuda_kore"])
    assert list_pixi_workspaces.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown pixi workspace 'cuda_kore'" in captured.err
    assert "expected one of: all, root, cuda_core" in captured.err


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_defaults_to_every_workspace(tmp_path, monkeypatch, capsys):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["list_pixi_workspaces.py"])
    assert list_pixi_workspaces.main() == 0
    assert [w["id"] for w in json.loads(capsys.readouterr().out)] == ["root", "cuda_core"]
