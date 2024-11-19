# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# If we have subcategories of examples in the future, this file can be split along those lines

from .utils import run_example
import os
import glob
import pytest

samples_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'examples')
sample_files = glob.glob(samples_path+'**/*.py', recursive=True)
@pytest.mark.parametrize(
    'example', sample_files
)
class TestExamples:
    def test_example(self, example, deinit_cuda):
        filename = os.path.basename(example)
        run_example(samples_path, example)
