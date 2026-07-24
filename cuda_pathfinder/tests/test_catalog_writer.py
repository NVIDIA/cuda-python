# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import runpy

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG
from toolshed._catalog_writer import render_catalog


@pytest.mark.agent_authored(model="gpt-5")
def test_catalog_writer_round_trips_windows_search_dirs(tmp_path):
    generated_catalog = tmp_path / "descriptor_catalog.py"
    generated_catalog.write_text(render_catalog(DESCRIPTOR_CATALOG), encoding="utf-8")

    rendered_specs = runpy.run_path(str(generated_catalog))["DESCRIPTOR_CATALOG"]

    assert tuple(dataclasses.asdict(spec) for spec in rendered_specs) == tuple(
        dataclasses.asdict(spec) for spec in DESCRIPTOR_CATALOG
    )
