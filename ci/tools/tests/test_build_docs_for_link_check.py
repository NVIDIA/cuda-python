# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BUILD_HELPER = Path(__file__).resolve().parents[1] / "build_docs_for_link_check.sh"


@pytest.mark.parametrize("checkout_name", ["checkout", "checkout with spaces"])
@pytest.mark.agent_authored(model="gpt-6")
def test_checking_build_uses_final_local_output_as_docs_domain(tmp_path, monkeypatch, checkout_name):
    checkout = tmp_path / checkout_name
    helper = checkout / "ci/tools/build_docs_for_link_check.sh"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(BUILD_HELPER, helper)

    # Skip package installation/import checks while running the helper's actual
    # environment setup and artifact copy. Sphinx is not a CI-tools dependency.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == '-m' && "$2" == 'pip' ]]; then
    exit 0
elif [[ "$1" == '-' ]]; then
    cat >/dev/null
    exit 0
fi
exec "$TEST_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python.chmod(0o755)

    docs = checkout / "cuda_python/docs"
    docs.mkdir(parents=True)
    (docs / "build_all_docs.sh").write_text(
        """set -euo pipefail
[[ "$1" == 'latest-only' ]]
"$TEST_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output = Path("build/html")
output.mkdir(parents=True)
settings = {key: os.environ[key] for key in (
    "CUDA_PYTHON_DOCS_DOMAIN", "BUILD_LATEST", "BUILD_PREVIEW",
)}
(output / "build-settings.json").write_text(json.dumps(settings), encoding="utf-8")
PY
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TEST_PYTHON", sys.executable)
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "docs")
    monkeypatch.setenv("CUDA_PYTHON_DOCS_GITHUB_REF", "test-ref")
    monkeypatch.setenv("CUDA_PYTHON_DOCS_DOMAIN", "https://nvidia.github.io/cuda-python")
    monkeypatch.setenv("BUILD_PREVIEW", "1")

    subprocess.run(["bash", str(helper)], cwd=tmp_path, check=True)  # noqa: S603, S607

    output = checkout / "artifacts/docs"
    settings = json.loads((output / "build-settings.json").read_text(encoding="utf-8"))
    assert settings == {
        "CUDA_PYTHON_DOCS_DOMAIN": output.as_uri(),
        "BUILD_LATEST": "1",
        "BUILD_PREVIEW": "0",
    }
