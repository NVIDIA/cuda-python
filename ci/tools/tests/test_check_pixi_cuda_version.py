# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ci.tools import bindings_config, check_pixi_cuda_version


def _write_pixi(path: Path, pins: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = ", ".join(f'"{pin}"' for pin in pins.values())
    features = "".join(f'[feature.{variant}.dependencies]\ncuda-version = "{pin}"\n' for variant, pin in pins.items())
    path.write_text(f"[workspace.build-variants]\ncuda-version = [{variants}]\n{features}", encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_pin_check_catches_drift_in_a_maintenance_root(tmp_path, monkeypatch, capsys):
    config = bindings_config.BindingsConfig(
        2,
        (
            bindings_config.BindingsPackage("cuda_bindings_12", (12, 9, 1), "maintenance"),
            bindings_config.BindingsPackage("cuda_bindings_14", (14, 1, 2), "current"),
        ),
    )
    for package in config.package_roots:
        _write_pixi(tmp_path / package.package_root / "pixi.toml", {package.cuda_variant: f"{package.ctk_target}.*"})
    _write_pixi(tmp_path / "cuda_core/pixi.toml", {"cu12": "12.*", "cu14": "14.1.*"})
    monkeypatch.setattr(bindings_config, "load_config", lambda: config)
    monkeypatch.setattr(check_pixi_cuda_version, "ROOT", tmp_path)
    assert check_pixi_cuda_version.main() == 0

    _write_pixi(tmp_path / "cuda_bindings_12/pixi.toml", {"cu12": "13.*"})

    assert check_pixi_cuda_version.main() == 1
    assert "cuda_bindings_12/pixi.toml" in capsys.readouterr().err
