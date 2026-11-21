# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "components",
    [("cuda-bindings",), ("", "cuda-bindings", "cuda-core", "cuda-pathfinder")],
    ids=["bindings", "all"],
)
@pytest.mark.agent_authored(model="GPT-6")
def test_release_docs_preserve_shared_files(tmp_path, components):
    payload = tmp_path / "payload"
    pages = tmp_path / "pages"
    shared = ("objects.inv", "nv-versions.json", "versions.json", "latest/index.html")
    versioned = ("12.9.11/objects.inv", "12.9.11/release/12.9.11-notes.html")

    for component in components:
        for name in shared:
            path = pages / component / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("main", encoding="utf-8")
        for name in (*shared, *versioned):
            path = payload / component / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("release", encoding="utf-8")

    tool = Path(__file__).resolve().parents[1] / "remove_shared_release_docs.sh"
    subprocess.run(["bash", str(tool), str(payload)], check=True)  # noqa: S603, S607

    # Simulate the Pages deploy's overlay of the staged files.
    shutil.copytree(payload, pages, dirs_exist_ok=True)
    for component in components:
        for name in shared:
            path = pages / component / name
            assert path.read_text(encoding="utf-8") == "main", path.relative_to(pages)
        for name in versioned:
            assert (pages / component / name).read_text(encoding="utf-8") == "release"
