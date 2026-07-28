# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("test_args.json")
SAMPLE_KEY = "cuda_bindings/extra/isoFdModelling"


@pytest.mark.agent_authored(model="gpt-5")
def test_iso_fd_modelling_exposes_no_display_option() -> None:
    sample = REPO_ROOT / "samples" / "cuda_bindings" / "extra" / "isoFdModelling" / "isoFdModelling.py"

    result = subprocess.run(  # noqa: S603 - executes a repository-owned sample
        [sys.executable, str(sample), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--no-display" in result.stdout


@pytest.mark.agent_authored(model="gpt-5")
def test_iso_fd_modelling_has_headless_test_configuration() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config[SAMPLE_KEY]["min_gpus"] == 2
    assert config[SAMPLE_KEY]["python"]["args"] == ["--no-display"]
