# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize(
    ("package", "tag", "version"),
    (
        ("cuda_core", "cuda-core-v1.2.3.post1", "v1.2.3.post1"),
        ("cuda_pathfinder", "cuda-pathfinder-v1.2.3.post1", "v1.2.3.post1"),
        ("cuda_bindings", "v13.3.1.post1", "v13.3.1.post1"),
        ("cuda_bindings_12", "v12.9.8.post1", "v12.9.8.post1"),
    ),
)
def test_release_package_scm_regex_preserves_post_suffix(package, tag, version):
    with (REPO_ROOT / package / "pyproject.toml").open("rb") as stream:
        pattern = tomllib.load(stream)["tool"]["setuptools_scm"]["tag_regex"]

    match = re.match(pattern, tag)

    assert match is not None
    assert match.group("version") == version
