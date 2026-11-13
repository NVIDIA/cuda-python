# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ci.tools.select_test_matrix import select_test_matrix

PACKAGES = [
    {
        "package_root": "bindings12",
        "toolkit_version": "12.9.1",
        "ctk_target": "12.9",
        "cuda_variant": "cu12",
        "release_status": "maintenance",
    },
    {
        "package_root": "bindings13",
        "toolkit_version": "13.4.2",
        "ctk_target": "13.4",
        "cuda_variant": "cu13",
        "release_status": "current",
    },
]


def rows(*toolkits):
    return [{"CUDA_VER": toolkit, "DRIVER": "latest"} for toolkit in toolkits]


def plan(*, bindings=False, core=True, meta=False):
    return {
        "jobs": {"test_cuda_majors": {"cu12": True, "cu13": False}},
        "sources": {"pathfinder": "artifact"},
        "modules": {
            "pathfinder": {"needs_test": False},
            "bindings": {
                "needs_test": bindings,
                "package_roots": {package["package_root"]: {"needs_test": bindings} for package in PACKAGES},
            },
            "core": {"needs_test": core, "cuda_majors": {"cu12": {"needs_test": core}, "cu13": {"needs_test": core}}},
            "python": {
                "needs_test": meta,
                "package_roots": {package["package_root"]: {"needs_test": meta} for package in PACKAGES},
            },
        },
    }


@pytest.mark.agent_authored(model="gpt-6")
def test_exact_toolkit_rows_use_local_bindings_and_older_toolkits_use_published_floors():
    matrix = select_test_matrix(rows("12.9.1", "12.6.3", "13.4.2", "13.0.2"), PACKAGES)

    assert [
        (row["BINDINGS_SOURCE"], row["BINDINGS_SOURCE_DIR"], row["BINDINGS_BUILD_VERSION"]) for row in matrix["include"]
    ] == [
        ("local", "bindings12", "12.9.1"),
        ("floor", "bindings13", "13.4.2"),
        ("local", "bindings13", "13.4.2"),
        ("floor", "bindings13", "13.4.2"),
    ]


@pytest.mark.agent_authored(model="gpt-6")
def test_workplan_retains_only_the_required_cuda_major_and_suites():
    matrix = select_test_matrix(
        rows("12.9.1", "12.6.3", "13.4.2"), PACKAGES, workplan=plan(bindings=True, core=False, meta=True)
    )

    # Without Core or Pathfinder tests the older-toolkit row has no applicable
    # suites. The required maintenance bindings/metapackage row must remain.
    assert [(row["CUDA_VER"], row["BINDINGS_SOURCE_DIR"]) for row in matrix["include"]] == [("12.9.1", "bindings12")]
    row = matrix["include"][0]
    assert row["TEST_BINDINGS"] == row["TEST_PYTHON"] == "true"
    assert row["TEST_PATHFINDER"] == row["TEST_CORE"] == "false"
    assert row["SUITES"] == "bindings12+meta12"


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


@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_published_floor_rows_never_select_local_wheel_suites():
    row = select_test_matrix(rows("12.6.3"), PACKAGES, build_type="nightly")["include"][0]

    # Workflows gate local-wheel downloads and suites on these flags alone.
    assert row["TEST_BINDINGS"] == row["TEST_PYTHON"] == "false"


@pytest.mark.agent_authored(model="gpt-6")
def test_toolkit_patch_updates_keep_nightly_bindings_coverage():
    packages = [dict(package) for package in PACKAGES]
    packages[1]["toolkit_version"] = "13.4.3"

    row = select_test_matrix(rows("13.4.2"), packages, build_type="nightly")["include"][0]

    assert row["BINDINGS_SOURCE"] == "local"
    assert row["BINDINGS_BUILD_VERSION"] == "13.4.3"
    assert {"bindings13", "meta13"} <= set(row["SUITES"].split("+"))


@pytest.mark.agent_authored(model="gpt-6")
def test_core_only_workplan_retains_floor_tests_without_enabling_local_binding_tests():
    matrix = select_test_matrix(rows("12.9.1", "12.6.3"), PACKAGES, workplan=plan())

    assert [row["BINDINGS_SOURCE"] for row in matrix["include"]] == ["local", "floor"]
    for row in matrix["include"]:
        assert row["TEST_CORE"] == "true"
        assert row["TEST_BINDINGS"] == row["TEST_PYTHON"] == "false"
        assert row["SUITES"] == "core12"
