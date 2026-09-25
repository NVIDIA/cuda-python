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
    assert workspaces[0] == {
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
            "manifest": ".",
            "lockfile": "pixi.lock",
        },
        {
            "manifest": "benchmarks/cuda_core",
            "lockfile": "benchmarks/cuda_core/pixi.lock",
        },
        {
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
    assert [w["manifest"] for w in list_pixi_workspaces.discover()] == ["."]


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


@pytest.mark.parametrize("directory", ["all", "root"])
@pytest.mark.agent_authored(model="gpt-5.6")
def test_discover_allows_nested_workspace_names(tmp_path, monkeypatch, directory):
    _make_repo(tmp_path, ["pixi.toml", f"{directory}/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    assert list_pixi_workspaces.discover() == [
        {
            "manifest": ".",
            "lockfile": "pixi.lock",
        },
        {
            "manifest": directory,
            "lockfile": f"{directory}/pixi.lock",
        },
    ]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_main_rejects_removed_select_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["list_pixi_workspaces.py", "--select", "cuda_core"])
    with pytest.raises(SystemExit) as exc_info:
        list_pixi_workspaces.main()
    assert exc_info.value.code == 2


@pytest.mark.agent_authored(model="claude-opus-5")
def test_main_prints_every_workspace(tmp_path, monkeypatch, capsys):
    _make_repo(tmp_path, ["pixi.toml", "cuda_core/pixi.toml"])
    monkeypatch.setattr(list_pixi_workspaces, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["list_pixi_workspaces.py"])
    assert list_pixi_workspaces.main() == 0
    assert [w["manifest"] for w in json.loads(capsys.readouterr().out)] == [".", "cuda_core"]
