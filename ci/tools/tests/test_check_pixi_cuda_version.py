# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import check_pixi_cuda_version


def _write_pixi(path: Path, cuda_variant: str, cuda_pin: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""\
[workspace.build-variants]
cuda-version = [\"12.*\", \"{cuda_pin}\"]

[feature.{cuda_variant}.dependencies]
cuda-version = \"{cuda_pin}\"
""",
        encoding="utf-8",
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_main_uses_current_bindings_toolkit_version(tmp_path, monkeypatch, capsys):
    pixi_files = [tmp_path / package / "pixi.toml" for package in ("cuda_bindings_14", "cuda_core")]
    for path in pixi_files:
        _write_pixi(path, "cu14", "14.1.*")

    config = Mock()
    config.line_for_role.return_value = SimpleNamespace(
        source_dir="cuda_bindings_14",
        toolkit_version="14.1.2",
    )
    monkeypatch.setattr(check_pixi_cuda_version.bindings_config, "load_config", lambda: config)
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)

    assert check_pixi_cuda_version.main() == 0
    assert "toolkit_version='14.1.2'" in capsys.readouterr().out
    config.line_for_role.assert_called_once_with("current")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_main_reports_invalid_bindings_config(monkeypatch, capsys):
    def fail_to_load():
        raise check_pixi_cuda_version.bindings_config.BindingsConfigError("broken registry")

    monkeypatch.setattr(check_pixi_cuda_version.bindings_config, "load_config", fail_to_load)

    assert check_pixi_cuda_version.main() == 2
    assert "invalid CUDA bindings configuration: broken registry" in capsys.readouterr().err
