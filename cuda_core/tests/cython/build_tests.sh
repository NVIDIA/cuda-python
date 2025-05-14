#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

SCRIPTPATH=$(dirname $(realpath "$0"))
CPLUS_INCLUDE_PATH=$SCRIPTPATH/../../cuda/core/experimental/include:$CUDA_HOME/include:$CPLUS_INCLUDE_PATH cythonize -3 -i $(dirname "$0")/test_*.pyx
