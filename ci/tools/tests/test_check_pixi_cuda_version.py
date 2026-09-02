# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ci.tools import check_pixi_cuda_version


def _write_pixi(path: Path, cuda_pins: dict[str, str]) -> None:
    path.parent.mkdir(parents=True)
    variants = ", ".join(f'"{pin}"' for pin in cuda_pins.values())
    features = "\n".join(
        f"""\
[feature.{cuda_variant}.dependencies]
cuda-version = "{cuda_pin}"
"""
        for cuda_variant, cuda_pin in cuda_pins.items()
    )
    path.write_text(
        f"""\
[workspace.build-variants]
cuda-version = [{variants}]

{features}
""",
        encoding="utf-8",
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_main_checks_every_registered_bindings_line(tmp_path, monkeypatch, capsys):
    lines = (
        SimpleNamespace(
            line_id="released-12",
            source_dir="cuda_bindings_12",
            toolkit_version="12.9.1",
            cuda_variant="cu12",
        ),
        SimpleNamespace(
            line_id="released-14",
            source_dir="cuda_bindings_14",
            toolkit_version="14.1.2",
            cuda_variant="cu14",
        ),
    )
    _write_pixi(tmp_path / "cuda_bindings_12" / "pixi.toml", {"cu12": "12.*"})
    _write_pixi(tmp_path / "cuda_bindings_14" / "pixi.toml", {"cu14": "14.1.*"})
    _write_pixi(tmp_path / "cuda_core" / "pixi.toml", {"cu12": "12.*", "cu14": "14.1.*"})

    monkeypatch.setattr(
        check_pixi_cuda_version.bindings_config,
        "load_config",
        lambda: SimpleNamespace(lines=lines),
    )
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)

    assert check_pixi_cuda_version.main() == 0
    output = capsys.readouterr().out
    assert "cuda_bindings_12/pixi.toml: released-12" in output
    assert "cuda_bindings_14/pixi.toml: released-14" in output
    assert "cuda_core/pixi.toml: released-12" in output
    assert "cuda_core/pixi.toml: released-14" in output


@pytest.mark.agent_authored(model="gpt-5.6")
def test_main_reports_maintenance_bindings_pin_drift(tmp_path, monkeypatch, capsys):
    line = SimpleNamespace(
        line_id="released-12",
        source_dir="cuda_bindings_12",
        toolkit_version="12.9.1",
        cuda_variant="cu12",
    )
    _write_pixi(tmp_path / "cuda_bindings_12" / "pixi.toml", {"cu12": "13.*"})
    _write_pixi(tmp_path / "cuda_core" / "pixi.toml", {"cu12": "12.*"})
    monkeypatch.setattr(
        check_pixi_cuda_version.bindings_config,
        "load_config",
        lambda: SimpleNamespace(lines=(line,)),
    )
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)

    assert check_pixi_cuda_version.main() == 1
    error = capsys.readouterr().err
    assert "cuda_bindings_12/pixi.toml" in error
    assert "does not cover registered line 'released-12' toolkit_version='12.9.1'" in error


@pytest.mark.agent_authored(model="gpt-5.6")
def test_main_reports_core_variant_pin_drift(tmp_path, monkeypatch, capsys):
    line = SimpleNamespace(
        line_id="released-14",
        source_dir="cuda_bindings_14",
        toolkit_version="14.1.2",
        cuda_variant="cu14",
    )
    _write_pixi(tmp_path / "cuda_bindings_14" / "pixi.toml", {"cu14": "14.1.*"})
    _write_pixi(tmp_path / "cuda_core" / "pixi.toml", {"cu14": "14.2.*"})
    monkeypatch.setattr(
        check_pixi_cuda_version.bindings_config,
        "load_config",
        lambda: SimpleNamespace(lines=(line,)),
    )
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)

    assert check_pixi_cuda_version.main() == 1
    error = capsys.readouterr().err
    assert "cuda_core/pixi.toml" in error
    assert "does not cover registered line 'released-14' toolkit_version='14.1.2'" in error


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_main_reports_invalid_bindings_config(monkeypatch, capsys):
    def fail_to_load():
        raise check_pixi_cuda_version.bindings_config.BindingsConfigError("broken registry")

    monkeypatch.setattr(check_pixi_cuda_version.bindings_config, "load_config", fail_to_load)

    assert check_pixi_cuda_version.main() == 2
    assert "invalid CUDA bindings configuration: broken registry" in capsys.readouterr().err
