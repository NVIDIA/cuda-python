# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# If we have subcategories of examples in the future, this file can be split along those lines

from pathlib import Path

import pytest

from .utils import run_example

# not dividing, but navigating into the "examples" directory.
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# recursively glob for test files in examples directory, sort for deterministic
# test runs. Relative paths offer cleaner output when tests fail.
SAMPLE_FILES = sorted([str(p.relative_to(EXAMPLES_DIR)) for p in EXAMPLES_DIR.glob("**/*.py")])


@pytest.mark.parametrize("example_rel_path", SAMPLE_FILES)
class TestExamples:
    # deinit_cuda is defined in conftest.py and pops the cuda context automatically.
    def test_example(self, example_rel_path: str, deinit_cuda) -> None:
        run_example(str(EXAMPLES_DIR), example_rel_path)
