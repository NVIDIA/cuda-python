# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest
from cuda_python_test_helpers.pep723 import has_package_requirements_or_skip

examples_path = Path(__file__).parents[1] / "examples"
examples_files = list(examples_path.glob("**/*.py"))


# ``ids=str`` keeps the test IDs as the example's path, the way they read when
# the parameters were plain strings.
@pytest.mark.parametrize("example", examples_files, ids=str)
def test_example(example):
    has_package_requirements_or_skip(example)

    env = os.environ.copy()
    env["CUDA_BINDINGS_SKIP_EXAMPLE"] = "100"
    env["MPLBACKEND"] = "Agg"  # avoid plt.show() from blocking

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
