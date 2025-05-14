#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH cythonize -3 -i $(dirname "$0")/test_*.pyx
