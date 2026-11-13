# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

import pytest
import setuptools
from packaging.requirements import Requirement


@pytest.mark.parametrize("major, version", [("12", "12.9.11"), ("13", "13.5.4")])
@pytest.mark.agent_authored(model="gpt-6")
def test_doubly_tagged_sdist_keeps_selected_version_and_dependencies_without_selector(tmp_path, major, version):
    source = Path(__file__).resolve().parents[3] / "cuda_python"
    checkout = tmp_path / "checkout"
    package = checkout / "cuda_python"
    package.mkdir(parents=True)
    for filename in ("setup.py", "pyproject.toml", "DESCRIPTION.rst", "LICENSE"):
        shutil.copyfile(source / filename, package / filename)

    env = os.environ.copy()
    env["CUDA_PYTHON_BUILD_MAJOR"] = major
    for args in (
        ("init", "--quiet"),
        ("add", "."),
        ("commit", "--quiet", "-m", "Release"),
        ("tag", "v12.9.11"),
        ("tag", "v13.5.4"),
    ):
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
    env.pop("CUDA_PYTHON_BUILD_MAJOR")
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
    assert metadata["Version"] == version
    dependencies = {Requirement(value).name for value in metadata.get_all("Requires-Dist", [])}
    expected = {"cuda-bindings"}
    if major == "13":
        expected.update({"cuda-core", "cuda-pathfinder"})
    assert dependencies == expected


@pytest.mark.agent_authored(model="gpt-6")
def test_archive_rejects_resolved_version_from_other_cuda_major(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / "cuda_python"
    package = tmp_path / "cuda_python"
    package.mkdir()
    for filename in ("setup.py", "pyproject.toml"):
        shutil.copyfile(source / filename, package / filename)
    (tmp_path / ".git_archival.txt").write_text(
        "node: 481d12c848bdf70386ba70cd601bcfaef52cda7f\n"
        "node-date: 2026-11-10T09:44:41-08:00\n"
        "ref-names: HEAD -> master, tag: v13.5.4, tag: v12.9.11\n"
    )
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    monkeypatch.setenv("CUDA_PYTHON_BUILD_MAJOR", "12")
    monkeypatch.chdir(package)

    with pytest.raises(ValueError, match=r"CUDA_PYTHON_BUILD_MAJOR=12 does not match resolved version 13\.5\.4"):
        runpy.run_path(str(package / "setup.py"))
    assert not metadata
