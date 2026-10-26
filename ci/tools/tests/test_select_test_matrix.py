# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ci.tools.select_test_matrix import select_test_matrix

PACKAGES = [
    {
        "package_root": "bindings12",
        "toolkit_version": "12.9.1",
        "cuda_variant": "cu12",
        "release_status": "maintenance",
    },
    {"package_root": "bindings13", "toolkit_version": "13.4.2", "cuda_variant": "cu13", "release_status": "current"},
]


def rows(*toolkits):
    return [{"CUDA_VER": toolkit, "DRIVER": "latest"} for toolkit in toolkits]


def plan(*, bindings=False, core=True, meta=False):
    return {
        "jobs": {"test_cuda_majors": {"cu12": True, "cu13": False}},
        "sources": {"pathfinder": "artifact"},
        "modules": {
            "pathfinder": {"needs_test": False},
            "bindings": {"package_roots": {package["package_root"]: {"needs_test": bindings} for package in PACKAGES}},
            "core": {"cuda_majors": {"cu12": {"needs_test": core}, "cu13": {"needs_test": core}}},
            "python": {"package_roots": {package["package_root"]: {"needs_test": meta} for package in PACKAGES}},
        },
    }


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_exact_toolkit_rows_use_local_bindings_and_older_toolkits_use_published_floors():
    matrix = select_test_matrix(rows("12.9.1", "12.6.3", "13.4.2", "13.0.2"), PACKAGES)

    assert [(row["CUDA_VARIANT"], row["BINDINGS_SOURCE"], row["BINDINGS_SOURCE_DIR"]) for row in matrix["include"]] == [
        ("cu12", "local", "bindings12"),
        ("cu12", "floor", ""),
        ("cu13", "local", "bindings13"),
        ("cu13", "floor", ""),
    ]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_workplan_retains_only_the_required_cuda_major_and_suites():
    matrix = select_test_matrix(
        rows("12.9.1", "12.6.3", "13.4.2"), PACKAGES, workplan=plan(bindings=True, core=False, meta=True)
    )

    # Without Core or Pathfinder tests the older-toolkit row has no applicable
    # suites. The required maintenance bindings/metapackage row must remain.
    assert [(row["CUDA_VER"], row["BINDINGS_SOURCE_DIR"]) for row in matrix["include"]] == [("12.9.1", "bindings12")]


@pytest.mark.parametrize("component", ("bindings", "meta"))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_filter_cannot_silently_drop_required_local_wheel_coverage(component):
    with pytest.raises(ValueError, match="omits required local-wheel coverage: bindings12"):
        select_test_matrix(rows("12.6.3", "13.4.2"), PACKAGES, workplan=plan(**{component: True}))


@pytest.mark.parametrize("test_mode", ("nightly-pytorch", "standard"))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_nightly_subsets_may_use_published_floor_bindings(test_mode):
    matrix = select_test_matrix(rows("12.6.3"), PACKAGES, build_type="nightly", test_mode=test_mode)

    assert [row["BINDINGS_SOURCE"] for row in matrix["include"]] == ["floor"]
