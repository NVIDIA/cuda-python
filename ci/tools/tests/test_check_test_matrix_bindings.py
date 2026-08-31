# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from ci.tools.check_test_matrix_bindings import (
    MatrixBindingsError,
    check_test_matrix_bindings,
    main,
)


def bindings_config(*lines):
    return {
        "schema_version": 2,
        "lines": list(lines),
        "roles": {
            "current": ["released-13"],
            "maintenance": ["released-12"],
            "unreleased": [],
        },
    }


def line(line_id, toolkit_version, role, *, cuda_variant=None):
    major = toolkit_version.partition(".")[0]
    return {
        "line_id": line_id,
        "toolkit_version": toolkit_version,
        "cuda_variant": cuda_variant or f"cu{major}",
        "roles": [role],
    }


@pytest.mark.agent_authored(model="gpt-5.6-sol")
class TestCheckTestMatrixBindings:
    def test_accepts_exact_rows_for_enabled_public_lines(self):
        config = bindings_config(
            line("released-12", "12.9.1", "maintenance"),
            line("released-13", "13.3.0", "current"),
        )

        check_test_matrix_bindings(
            config,
            {"cu12": True, "cu13": True},
            [{"CUDA_VER": "12.9.1"}, {"CUDA_VER": "13.0.2"}, {"CUDA_VER": "13.3.0"}],
        )

    def test_requires_each_enabled_same_major_line(self):
        config = bindings_config(
            line("released-13-4", "13.4.0", "maintenance"),
            line("released-13-5", "13.5.0", "current"),
        )

        with pytest.raises(MatrixBindingsError, match=r"released-13-5 .*CUDA 13\.5\.0"):
            check_test_matrix_bindings(config, {"cu13": True}, [{"CUDA_VER": "13.4.0"}])

    def test_ignores_disabled_and_unreleased_lines(self):
        config = bindings_config(
            line("released-12", "12.9.1", "maintenance"),
            line("released-13", "13.3.0", "current"),
            line("future-13-4", "13.4.0", "unreleased"),
        )

        check_test_matrix_bindings(config, {"cu12": False, "cu13": True}, [{"CUDA_VER": "13.3.0"}])

    @pytest.mark.parametrize(
        ("config", "enabled", "matrix", "message"),
        [
            ([], {"cu13": True}, [], "bindings config must be a JSON object"),
            (
                bindings_config(line("released-13", "13.3.0", "current")),
                {"cu13": "true"},
                [],
                "enabled CUDA variants must be a JSON object of boolean values",
            ),
            (
                bindings_config(line("released-13", "13.3.0", "current")),
                {"cu14": True},
                [],
                "enabled CUDA variants are absent from the public bindings registry: cu14",
            ),
            (
                bindings_config(line("released-13", "13.3.0", "current")),
                {"cu13": True},
                [{"PY_VER": "3.12"}],
                "test matrix row 0 has invalid CUDA_VER",
            ),
            (
                bindings_config(line("released-13", "13.3.0", "current", cuda_variant="cu12")),
                {"cu13": True},
                [],
                "does not match toolkit_version",
            ),
        ],
    )
    def test_rejects_malformed_inputs(self, config, enabled, matrix, message):
        with pytest.raises(MatrixBindingsError, match=message):
            check_test_matrix_bindings(config, enabled, matrix)

    def test_cli_reports_invalid_json(self, capsys):
        result = main(
            [
                "--bindings-config",
                "not-json",
                "--enabled-cuda-variants",
                "{}",
                "--test-matrix",
                "[]",
            ]
        )

        assert result == 2
        assert "bindings config is not valid JSON" in capsys.readouterr().err

    def test_cli_accepts_valid_inputs(self):
        config = bindings_config(line("released-13", "13.3.0", "current"))

        assert (
            main(
                [
                    "--bindings-config",
                    json.dumps(config),
                    "--enabled-cuda-variants",
                    '{"cu13":true}',
                    "--test-matrix",
                    '[{"CUDA_VER":"13.3.0"}]',
                ]
            )
            == 0
        )
