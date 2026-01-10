# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom build hooks for cuda-pathfinder.

This module validates git tags are available before setuptools-scm runs,
ensuring proper version detection during pip install. All PEP 517 build
hooks are delegated to setuptools.build_meta.
"""

# Import and re-export all PEP 517 hooks from setuptools.build_meta
from setuptools.build_meta import *  # noqa: F403
