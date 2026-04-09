# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import importlib.metadata
import os
import re

import pytest


def has_package_requirements_or_skip(example):
    example_name = os.path.basename(example)

    with open(example, encoding="utf-8") as f:
        content = f.read()

    # The canonical regex as defined in PEP 723
    pep723 = re.search(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$", content)
    if not pep723:
        raise ValueError(f"PEP 723 metadata not found in {example_name}")

    metadata = {}
    for line in pep723.group("content").splitlines():
        line = line.lstrip("# ").rstrip()
        if not line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        metadata[key] = value

    if "dependencies" not in metadata:
        raise ValueError(f"PEP 723 dependencies not found in {example_name}")

    missing_dependencies = []
    dependencies = eval(metadata["dependencies"])  # noqa: S307
    for dependency in dependencies:
        name = re.match("[a-zA-Z0-9_-]+", dependency)
        try:
            importlib.metadata.distribution(name.group(0))
        except importlib.metadata.PackageNotFoundError:
            missing_dependencies.append(name.string)

    if missing_dependencies:
        pytest.skip(f"Skipping {example} due to missing package requirement: {', '.join(missing_dependencies)}")
