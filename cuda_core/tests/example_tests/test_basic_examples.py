# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# If we have subcategories of examples in the future, this file can be split along those lines

from utils import run_example

def test_basic_examples():
    run_example("../examples", "saxpy.py")
    run_example("../examples", "vector_add.py")
