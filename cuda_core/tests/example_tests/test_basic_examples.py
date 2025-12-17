# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# If we have subcategories of examples in the future, this file can be split along those lines

import glob
import os

import pytest
from cuda.core import Device

from .utils import run_example

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)


@pytest.mark.parametrize("example", sample_files)
class TestExamples:
    def test_example(self, example, deinit_cuda):
        run_example(samples_path, example)
        if Device().device_id != 0:
            Device(0).set_current()
