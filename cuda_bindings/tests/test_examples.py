# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import glob
import os
import subprocess
import sys

import pytest

from cuda.bindings._test_helpers.pep723 import has_package_requirements_or_skip

examples_path = os.path.join(os.path.dirname(__file__), "..", "examples")
examples_files = glob.glob(os.path.join(examples_path, "**/*.py"), recursive=True)


BROKEN_EXAMPLES = {"numba_emm_plugin.py"}


@pytest.mark.parametrize("example", examples_files)
def test_example(example):
    if os.path.basename(example) in BROKEN_EXAMPLES:
        pytest.skip(f"Skipping broken example: {example}")

    has_package_requirements_or_skip(example)

    env = os.environ.copy()
    env["CUDA_BINDINGS_SKIP_EXAMPLE"] = "100"

    process = subprocess.run([sys.executable, example], capture_output=True, env=env)  # noqa: S603
    # returncode is a special value used in the examples to indicate that system requirements are not met.
    if process.returncode == 100:
        pytest.skip(process.stderr.decode(errors="replace").strip())
    elif process.returncode != 0:
        if process.stdout:
            print(process.stdout.decode(errors="replace"))
        if process.stderr:
            print(process.stderr.decode(errors="replace"), file=sys.stderr)
        raise AssertionError(f"`{example}` failed ({process.returncode})")
