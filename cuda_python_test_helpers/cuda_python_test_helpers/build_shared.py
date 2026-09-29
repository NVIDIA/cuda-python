# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tests for the toolchain / build-key / stamp helpers in build_hooks.py.

The helpers themselves live in ``cuda_bindings/_build_shared.py`` (with
``cuda_core/_build_shared.py`` a symlink into it), so their behavior is
identical no matter which backend loads them. These mixin classes capture
the tests that assert that shared behavior; each package's
``tests/test_build_hooks.py`` mixes them in against its own loaded
``build_hooks`` module.

Kept out of these mixins on purpose:

- ``test_llvm_sets_env_and_flags`` / ``test_gnu_sets_env_and_flags``: each
  package's ``_resolve_toolchain()`` assembles a different flag set (bindings
  adds ``-fpermissive``/``-fno-var-tracking-assignments`` on gnu, core does
  not), so the flag-set assertions live in each package's file.
- ``TestBuildToolchainStamp`` / ``TestBuildConfigStamp``: bindings stamps the
  toolchain name, core stamps a composite ``cu{major}-{toolchain}-...`` key.
  Only the ``_abi_stamp_path`` mechanics are shared here.
"""

import os
import shutil
import sys
import sysconfig

import pytest


class ResolveToolchainSharedMixin:
    """Common ``_resolve_toolchain`` assertions that don't depend on the flag set.

    Subclasses set ``build_hooks`` (the loaded build_hooks module). The
    default-toolchain happy paths (env not touched, external CC preserved,
    gnu flag set applied) aren't asserted here — a successful wheel build
    on any Linux CI worker already exercises them. Only behavior that a
    passing wheel build would not surface is kept: error paths and the
    llvm-override case (llvm has no CI, so it's not covered by wheels
    either).
    """

    build_hooks = None

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_case_insensitive(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "LLVM")
        name, _cc, _cxx, _cargs, _largs = self.build_hooks._resolve_toolchain()
        assert name == "llvm"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "icc")
        with pytest.raises(RuntimeError, match="not supported"):
            self.build_hooks._resolve_toolchain()

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_overrides_external_cc(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        # An explicit non-default toolchain governs the compiler, so a stale
        # external CC (e.g. "sccache cc") is replaced, not kept.
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", "sccache cc")
        _name, _cc, _cxx, _cargs, _largs = self.build_hooks._resolve_toolchain()
        assert os.environ["CC"] == "clang"


class CheckToolchainAvailableSharedMixin:
    """Common ``_check_toolchain_available`` assertions.

    Subclasses set ``build_hooks`` (the loaded build_hooks module). The
    default-is-noop happy path is covered implicitly by any successful
    wheel build; only the error message and the llvm-tools-present case
    (llvm has no CI) are kept.
    """

    build_hooks = None

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_missing_tool_lists_install_hint(self, monkeypatch):
        def fake_which(name):
            return None if name in ("clang", "clang++", "ld.lld") else "/bin/" + name

        monkeypatch.setattr(shutil, "which", fake_which)
        with pytest.raises(RuntimeError, match="clang and lld"):
            self.build_hooks._check_toolchain_available("llvm")

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_present_passes(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: "/bin/true")
        self.build_hooks._check_toolchain_available("llvm")


class AbiStampPathMixin:
    """``_abi_stamp_path`` scopes stamp files by Python's EXT_SUFFIX.

    Subclasses set ``build_hooks`` (the loaded build_hooks module). The test
    passes an arbitrary stem — each package uses its own stem at runtime
    (``.build-toolchain`` for cuda.bindings, ``.build-config`` for cuda.core),
    but the mechanism is shared.
    """

    build_hooks = None

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_stamp_path_is_scoped_to_extension_abi(self, monkeypatch):
        monkeypatch.setattr(sysconfig, "get_config_var", lambda _name: ".cpython-310-x86_64-linux-gnu.so")
        python_310 = self.build_hooks._abi_stamp_path(".build-test")
        monkeypatch.setattr(sysconfig, "get_config_var", lambda _name: ".cpython-311-x86_64-linux-gnu.so")
        python_311 = self.build_hooks._abi_stamp_path(".build-test")

        assert python_310 != python_311
        assert python_310.name == ".build-test.cpython-310-x86_64-linux-gnu.so"
        assert python_311.name == ".build-test.cpython-311-x86_64-linux-gnu.so"
