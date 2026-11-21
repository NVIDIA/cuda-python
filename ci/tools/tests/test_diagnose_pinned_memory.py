# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import venv
from pathlib import Path

import pytest

from ci.tools import diagnose_pinned_memory


@pytest.fixture
def diagnostic_environment(tmp_path):
    # A CPU-only stand-in for an installed wheel, alongside a source checkout
    # that must never be used. Keep real pytest and its dependencies available.
    builder = venv.EnvBuilder(with_pip=False)
    env_dir = tmp_path / "env"
    builder.create(env_dir)
    context = builder.ensure_directories(env_dir)
    python = context.env_exec_cmd
    site = Path(
        subprocess.check_output(  # noqa: S603 - fixture-owned interpreter and fixed argv.
            [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    (site / "pytest-dependencies.pth").write_text(str(Path(pytest.__file__).resolve().parents[1]) + "\n")
    core = site / "cuda/core"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("from ._version import __version__\n")
    (core / "_version.py").write_text('__version__ = "1.0"\n')
    metadata = site / "cuda_core-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: cuda-core\nVersion: 1.0\n")

    checkout = tmp_path / "cuda_core"
    source = checkout / "cuda/core"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('__version__ = "source-checkout"\n')
    tests = checkout / "tests"
    tests.mkdir()
    (checkout / "pytest.ini").write_text("[pytest]\npythonpath = tests\n")
    (tests / "diagnostic_helper.py").write_text("VALUE = 42\n")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "", "PYTHONPATH": str(checkout)}
    return python, checkout, core, env


@pytest.mark.parametrize("skip", [False, True], ids=["executes-wheel-test", "rejects-all-skipped"])
@pytest.mark.agent_authored(model="gpt-6")
def test_diagnostics_use_installed_wheel_and_require_executed_tests(diagnostic_environment, tmp_path, skip):
    python, checkout, core, env = diagnostic_environment
    body = "pytest.skip('GPU capability unavailable')" if skip else "assert cuda.core.__version__ == '1.0'"
    (checkout / "tests/test_probe.py").write_text(
        "import pytest\nimport cuda.core\nfrom diagnostic_helper import VALUE\n"
        f"def test_probe():\n    assert VALUE == 42\n    {body}\n"
    )
    report_path = tmp_path / "pytest-report.json"
    result = subprocess.run(  # noqa: S603 - fixture-owned interpreter and repository script.
        [python, "-I", diagnose_pinned_memory.__file__, "--pytest-child", str(report_path), "-q", "tests"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == int(skip), result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["completed"]
    assert report["counts"] == {"passed": int(not skip), "failed": 0, "skipped": int(skip)}
    assert Path(report["wheel"]["origins"]["cuda.core"]) == core / "__init__.py"
