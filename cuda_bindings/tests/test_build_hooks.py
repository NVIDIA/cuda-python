# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_hooks.py build infrastructure.

These cover the build-configuration identity that keys generated sources and
decides when build_ext must be forced. They do NOT require cuda.bindings to be
built, since they only exercise build-time infrastructure.

Mirrors cuda_core/tests/test_build_hooks.py; keep the two in sync.
"""

import importlib.util
from pathlib import Path

import pytest

from cuda.pathfinder import get_cuda_path_or_home

pytest.importorskip("setuptools")


def _load_build_hooks():
    """Load build_hooks.py from source without polluting sys.path.

    build_hooks.py is a PEP 517 build backend, not an installed module, and the
    directory holding it also contains the cuda/bindings/ sources that would
    shadow the installed package.
    """
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    spec = importlib.util.spec_from_file_location("cuda_bindings_build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_hooks = _load_build_hooks()


@pytest.fixture
def build_tree(tmp_path, monkeypatch):
    """Run the identity helpers against a scratch source tree with a fake cuda.h.

    The stamp path is relative because PEP 517 hooks always run with the
    package directory as the working directory.
    """
    toolkit = tmp_path / "toolkit"
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "include" / "cuda.h").write_text("#define CUDA_VERSION 13000\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_hooks, "_build_identity", None)
    monkeypatch.setattr(build_hooks, "force_build_ext", False)
    build_hooks._get_cuda_path.cache_clear()
    build_hooks._determine_cuda_major_version.cache_clear()
    get_cuda_path_or_home.cache_clear()
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PYTHON_COVERAGE", raising=False)
    return tmp_path


def _set_cuda_version(build_tree, cuda_version):
    (build_tree / "toolkit" / "include" / "cuda.h").write_text(f"#define CUDA_VERSION {cuda_version}\n")
    build_hooks._determine_cuda_major_version.cache_clear()


def _write_stamp(build_tree, identity):
    stamp = build_tree / build_hooks._BUILD_IDENTITY_STAMP
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(identity + "\n")


class TestDetermineCudaMajorVersion:
    @pytest.mark.agent_authored(model="claude-opus-5")
    @pytest.mark.parametrize(
        ("cuda_version", "expected_major"),
        [(11080, "11"), (12000, "12"), (12080, "12"), (13000, "13"), (13010, "13")],
    )
    def test_parses_cuda_h(self, build_tree, cuda_version, expected_major):
        _set_cuda_version(build_tree, cuda_version)
        assert build_hooks._determine_cuda_major_version() == expected_major

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_missing_macro_raises(self, build_tree):
        (build_tree / "toolkit" / "include" / "cuda.h").write_text("/* no version here */\n")
        build_hooks._determine_cuda_major_version.cache_clear()
        with pytest.raises(RuntimeError, match="Cannot determine CUDA major version"):
            build_hooks._determine_cuda_major_version()


class TestBuildIdentity:
    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_identity_encodes_configuration_axes(self, build_tree, monkeypatch):
        assert build_hooks._resolve_build_identity(debug=False) == "cu13"
        assert build_hooks._resolve_build_identity(debug=True) == "cu13-debug"
        monkeypatch.setenv("CUDA_PYTHON_COVERAGE", "1")
        assert build_hooks._resolve_build_identity(debug=False) == "cu13-coverage"

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_identity_tracks_cuda_major(self, build_tree):
        _set_cuda_version(build_tree, 12080)
        assert build_hooks._resolve_build_identity(debug=False) == "cu12"

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_first_build_does_not_force(self, build_tree):
        build_hooks._resolve_build_identity(debug=False)
        assert build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_same_configuration_does_not_force(self, build_tree):
        _write_stamp(build_tree, "cu13")
        build_hooks._resolve_build_identity(debug=False)
        assert build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="claude-opus-5")
    @pytest.mark.parametrize("previous", ["cu12", "cu13-debug", "cu13-coverage"])
    def test_changed_configuration_forces_rebuild(self, build_tree, previous):
        _write_stamp(build_tree, previous)
        build_hooks._resolve_build_identity(debug=False)
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_record_writes_stamp(self, build_tree):
        build_hooks._resolve_build_identity(debug=True)
        build_hooks.record_build_identity()
        assert (build_tree / build_hooks._BUILD_IDENTITY_STAMP).read_text().strip() == "cu13-debug"

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_record_is_noop_without_a_build(self, build_tree):
        """A failed build must not advertise outputs it never produced."""
        _write_stamp(build_tree, "cu12")
        build_hooks.record_build_identity()
        assert (build_tree / build_hooks._BUILD_IDENTITY_STAMP).read_text().strip() == "cu12"
