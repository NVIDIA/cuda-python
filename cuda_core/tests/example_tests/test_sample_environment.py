# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest


@pytest.mark.parametrize(
    ("distribution_name", "expected_source", "expected_version"),
    (
        ("cuda-core", "cuda_core", "1.1.0"),
        ("cuda-bindings", "cuda_bindings", "13.3.1"),
        ("cuda-pathfinder", "cuda_pathfinder", "1.5.6"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5")
def test_samples_environment_uses_one_local_cuda_distribution(
    distribution_name: str,
    expected_source: str,
    expected_version: str,
) -> None:
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != "samples":
        pytest.skip("local package provenance is specific to the Pixi samples environment")

    distributions = list(importlib.metadata.distributions(name=distribution_name))
    assert len(distributions) == 1, f"expected one {distribution_name} distribution, found {len(distributions)}"

    direct_url = json.loads(distributions[0].read_text("direct_url.json"))
    parsed_source = urlparse(direct_url["url"])
    source_path = Path(url2pathname(parsed_source.path)).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    assert parsed_source.scheme == "file"
    assert direct_url["dir_info"]["editable"] is True
    assert source_path == repository_root / expected_source

    record_paths = list((Path(sys.prefix) / "conda-meta").glob(f"{distribution_name}-*.json"))
    assert len(record_paths) == 1, f"expected one Conda record for {distribution_name}, found {record_paths}"

    record = json.loads(record_paths[0].read_text(encoding="utf-8"))
    assert record["version"] == expected_version
    assert record["channel"] is None
    assert urlparse(record["url"]).scheme == "file"
