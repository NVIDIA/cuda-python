# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_hooks.py: the CUDA header check that runs before cythonize.

A cuda-bindings source tree is generated from one CUDA header set and compiles
only against a toolkit of that major.minor; against another minor the C++
compile fails with redefinition errors that do not name the cause. The check
reads both versions and fails early with a message that does. No GPU needed.

build_hooks.py is a PEP 517 backend, not an installed module, so it is loaded
from source. It imports setuptools at the top; the ``test`` extra provides it.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import setuptools  # noqa: F401

# Don't call .resolve(): a symlinked checkout would make parents[1] point elsewhere.
BUILD_HOOKS = Path(__file__).parents[1] / "build_hooks.py"


@pytest.fixture(scope="module")
def build_hooks():
    if not BUILD_HOOKS.is_file():
        pytest.skip(f"{BUILD_HOOKS} is not in this tree; these tests need the source checkout")
    spec = importlib.util.spec_from_file_location("cuda_bindings_build_hooks", BUILD_HOOKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cuda_h(tmp_path, cuda_version):
    include = tmp_path / "include"
    include.mkdir(exist_ok=True)
    (include / "cuda.h").write_text(f"#define CUDA_VERSION {cuda_version}\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_generated_header_version_is_read_from_cydriver_pxd(build_hooks):
    generated = build_hooks._generated_cuda_version()
    assert generated // 1000 in (12, 13)
    assert build_hooks._major_minor(13040) == "13.4"
    assert build_hooks._major_minor(12090) == "12.9"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_a_header_of_the_generated_major_minor_passes(build_hooks, tmp_path):
    generated = build_hooks._generated_cuda_version()
    build_hooks._check_cuda_headers(_write_cuda_h(tmp_path, generated))
    # Only major.minor matters; the last digit (13041) is a toolkit patch.
    build_hooks._check_cuda_headers(_write_cuda_h(tmp_path, generated + 1))


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("delta", [-10, 10, -1000, 1000])
def test_another_header_fails_and_names_both_versions(build_hooks, tmp_path, delta):
    generated = build_hooks._generated_cuda_version()
    cuda_path = _write_cuda_h(tmp_path, generated + delta)
    with pytest.raises(RuntimeError) as excinfo:
        build_hooks._check_cuda_headers(cuda_path)
    message = str(excinfo.value)
    needed, found = build_hooks._major_minor(generated), build_hooks._major_minor(generated + delta)
    assert message.startswith(f"This cuda-bindings source tree needs CUDA {needed} headers, but ")
    assert os.path.realpath(os.path.join(cuda_path, "include", "cuda.h")) in message  # the resolved path
    assert f" is CUDA {found}. This is a build-time requirement only" in message
    assert build_hooks._INSTALL_URL in message
    assert message.endswith(
        f"Point CUDA_PATH or CUDA_HOME at a CUDA {needed} toolkit, or build from cuda-bindings {found}.x sources."
    )


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_an_unreadable_cuda_h_is_a_clear_error(build_hooks, tmp_path):
    with pytest.raises(RuntimeError, match=r"Cannot read CUDA_VERSION from .*cuda\.h"):
        build_hooks._check_cuda_headers(str(tmp_path))  # no include/cuda.h
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "cuda.h").write_text("/* no CUDA_VERSION macro */\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"Cannot read CUDA_VERSION from .*cuda\.h"):
        build_hooks._check_cuda_headers(str(tmp_path))


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_the_build_checks_the_header_before_it_touches_the_tree(build_hooks, tmp_path, monkeypatch):
    pytest.importorskip("Cython")  # _build_cuda_bindings imports it first
    cuda_path = _write_cuda_h(tmp_path, build_hooks._generated_cuda_version() + 10)
    monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: cuda_path)

    def not_reached():
        raise AssertionError("the header check must run before the source tree is modified")

    monkeypatch.setattr(build_hooks, "_rename_architecture_specific_files", not_reached)
    with pytest.raises(RuntimeError, match="source tree needs CUDA .* headers"):
        build_hooks._build_cuda_bindings()
