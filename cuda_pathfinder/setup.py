# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from setuptools import setup

# setuptools.build_meta does not put the package directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _build_shared import make_egg_info_command

setup(cmdclass={"egg_info": make_egg_info_command("cuda-pathfinder-v")})
