# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest

from ci.tools.bindings_config import check_package_metadata, parse_config


def _registry(maintenance_root="cuda_bindings_12", current_root="cuda_bindings_13"):
    return {
        "schema_version": 2,
        "python": {"supported": ["3.11", "3.14t", "3.15"]},
        "cuda": {
            "bindings": {
                "package_roots": {
                    maintenance_root: {"toolkit_version": "12.9.1", "release_status": "maintenance"},
                    current_root: {"toolkit_version": "13.4.1", "release_status": "current"},
                }
            }
        },
    }


def _write_package_metadata(root):
    config = parse_config(_registry())
    for package in config.package_roots:
        path = root / package.package_root / "pyproject.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            '[tool.setuptools_scm]\ngit_describe_command = ["git", "describe", "--match", '
            f'"v{package.ctk_target}.*"]\n',
            encoding="utf-8",
        )
    meta = root / "cuda_python" / "pyproject.toml"
    meta.parent.mkdir()
    meta.write_text(
        '[tool.cuda-python.scm-describe-match-by-major]\n"12" = "v12.9.*"\n"13" = "v13.4.*"\n',
        encoding="utf-8",
    )
    core = root / "cuda_core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncu12 = ["cuda-bindings>=12.9.0,<13"]\ncu13 = ["cuda-bindings>=13.4.0,<14"]\n',
        encoding="utf-8",
    )
    floor_reader = Path("cuda_core/cuda/core/_bindings_floor.py")
    (root / floor_reader).parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[3] / floor_reader, root / floor_reader)
    return config


@pytest.mark.parametrize(
    "changed_paths",
    (
        ("cuda_bindings_13/pyproject.toml",),
        ("cuda_python/pyproject.toml",),
        ("cuda_bindings_13/pyproject.toml", "cuda_python/pyproject.toml"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_metadata_check_catches_selectors_drifting_from_the_registered_line(tmp_path, changed_paths):
    config = _write_package_metadata(tmp_path)
    check_package_metadata(config, tmp_path)
    for relative in changed_paths:
        path = tmp_path / relative
        path.write_text(path.read_text().replace("v13.4.*", "v13.5.*"), encoding="utf-8")

    with pytest.raises(ValueError):
        check_package_metadata(config, tmp_path)


@pytest.mark.agent_authored(model="gpt-6")
def test_registered_toolkit_below_core_floor_is_rejected(tmp_path):
    config = _write_package_metadata(tmp_path)
    core_metadata = tmp_path / "cuda_core" / "pyproject.toml"
    core_metadata.write_text(core_metadata.read_text().replace(">=12.9.0", ">=12.10.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="below.*floor"):
        check_package_metadata(config, tmp_path)
