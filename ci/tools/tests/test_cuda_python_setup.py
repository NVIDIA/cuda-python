# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import tarfile
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement


@pytest.mark.agent_authored(model="gpt-6")
def test_tagged_maintenance_sdist_keeps_version_and_dependencies_without_selector(tmp_path):
    source = Path(__file__).resolve().parents[3] / "cuda_python"
    checkout = tmp_path / "checkout"
    package = checkout / "cuda_python"
    package.mkdir(parents=True)
    for filename in ("setup.py", "pyproject.toml", "DESCRIPTION.rst", "LICENSE"):
        shutil.copyfile(source / filename, package / filename)

    env = os.environ.copy()
    env.pop("CUDA_PYTHON_BUILD_MAJOR", None)
    for args in (("init", "--quiet"), ("add", "."), ("commit", "--quiet", "-m", "Release"), ("tag", "v12.9.11")):
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "-c",
                "user.name=CI test",
                "-c",
                "user.email=ci-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
        )

    subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
        cwd=package,
        env=env,
        check=True,
    )
    unpacked = tmp_path / "unpacked"
    with tarfile.open(next((package / "dist").glob("*.tar.gz"))) as archive:
        archive.extractall(unpacked, filter="data")
    standalone = next(unpacked.iterdir())
    subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=standalone,
        env=env,
        check=True,
    )
    with ZipFile(next((standalone / "dist").glob("*.whl"))) as wheel:
        metadata = message_from_bytes(
            wheel.read(next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")))
        )
    assert metadata["Version"] == "12.9.11"
    assert {Requirement(value).name for value in metadata.get_all("Requires-Dist", [])} == {"cuda-bindings"}
