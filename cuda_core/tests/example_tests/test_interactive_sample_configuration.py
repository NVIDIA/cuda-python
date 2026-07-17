# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("test_args.json")


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


@pytest.mark.parametrize("sample_name", ("glInteropFluid", "glInteropMipmapLod", "glInteropPlasma"))
@pytest.mark.agent_authored(model="gpt-5")
def test_interactive_gl_sample_exposes_frame_limit(sample_name: str) -> None:
    sample = REPO_ROOT / "samples" / "cuda_core" / sample_name / f"{sample_name}.py"

    result = subprocess.run(  # noqa: S603 - executes a repository-owned sample
        [sys.executable, str(sample), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--frames" in result.stdout


@pytest.mark.parametrize("sample_name", ("glInteropFluid", "glInteropMipmapLod", "glInteropPlasma"))
@pytest.mark.agent_authored(model="gpt-5")
def test_interactive_gl_sample_has_bounded_test_configuration(sample_name: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    args = config[f"cuda_core/{sample_name}"]["python"]["args"]

    assert args[0] == "--frames"
    assert int(args[1]) > 0


@pytest.mark.parametrize("sample_name", ("glInteropFluid", "glInteropMipmapLod", "glInteropPlasma"))
@pytest.mark.agent_authored(model="gpt-5")
def test_frame_limit_exits_after_current_draw_before_closing_window(sample_name: str) -> None:
    sample = REPO_ROOT / "samples" / "cuda_core" / sample_name / f"{sample_name}.py"
    tree = ast.parse(sample.read_text(encoding="utf-8"))
    on_draw = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "on_draw")
    calls = {_dotted_name(node.func) for node in ast.walk(on_draw) if isinstance(node, ast.Call)}

    assert "pyglet.app.exit" in calls
    assert "window.close" not in calls
