# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_lychee_inputs import authored_inputs, rendered_inputs, write_inputs


@pytest.mark.agent_authored(model="gpt-6")
def test_authored_inputs_use_tracked_documents_and_exclude_qa(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)  # noqa: S603, S607
    tracked = ["README.md", "docs/a guide.rst", "qa/README.md", "src/worker.py", "gone.md"]
    for name in tracked:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
    subprocess.run(["git", "add", "--", *tracked], cwd=tmp_path, check=True)  # noqa: S603, S607
    (tmp_path / "gone.md").unlink()
    (tmp_path / "untracked.md").write_text("content", encoding="utf-8")

    assert authored_inputs(tmp_path) == [tmp_path / "README.md", tmp_path / "docs/a guide.rst"]


@pytest.mark.agent_authored(model="gpt-6")
def test_authored_inputs_skip_tracked_symlinks_and_keep_their_real_target(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)  # noqa: S603, S607
    target = tmp_path / "README.md"
    target.write_text("[Guide](docs/guide.rst)\n", encoding="utf-8")
    package = tmp_path / "cuda_python"
    package.mkdir()
    (package / "README.md").symlink_to("../README.md")
    subprocess.run(["git", "add", "--", "README.md", "cuda_python/README.md"], cwd=tmp_path, check=True)  # noqa: S607

    assert authored_inputs(tmp_path) == [target]


@pytest.mark.agent_authored(model="gpt-6")
def test_rendered_inputs_include_all_components_and_exclude_static_assets(tmp_path):
    expected = ["cuda-bindings/latest/api.html", "cuda-core/latest/guide.html", "latest/index.html"]
    for name in [*expected, "cuda-core/latest/_static/theme.html", "_static/vendor/embed.html", "versions.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
    (tmp_path / "directory.html").mkdir()

    assert rendered_inputs(tmp_path) == [tmp_path / name for name in expected]


@pytest.mark.agent_authored(model="gpt-6")
def test_write_inputs_keeps_spaces_and_writes_absolute_paths(tmp_path):
    output = tmp_path / "lists/files.txt"
    paths = [tmp_path / "docs/a guide.rst", tmp_path / "README.md"]

    write_inputs(paths, output)

    assert output.read_text(encoding="utf-8") == "".join(f"{path}\n" for path in paths)


@pytest.mark.agent_authored(model="gpt-6")
def test_write_inputs_rejects_empty_inputs_without_publishing_list(tmp_path):
    output = tmp_path / "files.txt"

    with pytest.raises(ValueError, match="No documentation inputs found"):
        write_inputs([], output)

    assert not output.exists()


@pytest.mark.agent_authored(model="gpt-6")
def test_write_inputs_rejects_paths_that_would_split_into_multiple_inputs(tmp_path):
    output = tmp_path / "files.txt"

    with pytest.raises(ValueError, match="must not contain newlines"):
        write_inputs([tmp_path / "two\ninputs.md"], output)

    assert not output.exists()
