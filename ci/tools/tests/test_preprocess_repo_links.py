# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess_repo_links import preprocess_links

SCRIPT = Path(__file__).resolve().parents[1] / "preprocess_repo_links.py"


@pytest.mark.parametrize(
    "template",
    [
        "[Directory]({directory})\n[File]({file})\n[Relative](guide.rst)\n",
        "`Directory <{directory}>`_\n`File <{file}>`_\n`Relative <guide.rst>`_\n",
        '<a href="{directory}">Directory</a><a href="{file}">File</a><a href="guide.rst">Relative</a>\n',
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_preprocess_links_preserves_format_relative_links_and_url_encoding(tmp_path, template):
    root = tmp_path / "checkout space$#"
    content = template.format(
        directory="https://github.com/NVIDIA/cuda-python/tree/main/new_directory/",
        file="https://github.com/NVIDIA/cuda-python/blob/main/docs/a%20guide.md?plain=1#some-heading",
    )

    assert preprocess_links(content, root) == template.format(
        directory=f"{root.as_uri()}/new_directory/",
        file=f"{root.as_uri()}/docs/a%20guide.md?plain=1#some-heading",
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/cuda-python/blob/12.9.x/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/v13.0.0/cuda_bindings",
        "https://github.com/NVIDIA/cuda-python/blob/0123456789abcdef/README.md",
        "https://github.com/NVIDIA/cuda-python/blob/mainly/README.md",
        "https://github.com/NVIDIA/cuda-python/tree/main",
        "https://github.com/rwgk/cuda-python/blob/main/README.md",
        "https://github.com/NVIDIA/other-repository/tree/main/docs",
        "http://github.com/NVIDIA/cuda-python/blob/main/README.md",
        "https://raw.githubusercontent.com/NVIDIA/cuda-python/main/README.md",
        "https://nvidia.github.io/cuda-python/latest/",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_preprocess_links_leaves_other_targets_unchanged(tmp_path, url):
    assert preprocess_links(f"[Link]({url})\n", tmp_path) == f"[Link]({url})\n"


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_uses_its_checkout_root_and_leaves_input_unchanged(tmp_path):
    checkout = tmp_path / "checkout space$#"
    script = checkout / "ci" / "tools" / SCRIPT.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    source = tmp_path / "source guide.rst"
    content = b"`Repository <https://github.com/NVIDIA/cuda-python/tree/main/new_directory>`_\r\n"
    source.write_bytes(content)

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script), str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == f"`Repository <{checkout.as_uri()}/new_directory>`_\n"
    assert result.stderr == ""
    assert source.read_bytes() == content


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_missing_input_fails_without_success_output(tmp_path):
    source = tmp_path / "missing.md"

    result = subprocess.run([sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True)  # noqa: S603

    assert result.returncode != 0
    assert result.stdout == ""
    assert "FileNotFoundError" in result.stderr
    assert not source.exists()
