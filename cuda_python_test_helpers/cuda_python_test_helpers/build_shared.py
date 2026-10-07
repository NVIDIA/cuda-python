# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tests for the helpers in ``_build_shared.py``.

Both backends load the same ``_build_shared.py`` implementation through
package-local paths, so their behavior is identical. The mixins here hold the
tests of that shared behavior. Each package's ``tests/test_build_hooks.py``
mixes them in against the ``_build_shared`` module it loaded, by setting the
class attribute
``build_shared``.

Kept out of the mixins on purpose:

- What each package decides: the C++ standard, ``tweak`` flags, whether
  warnings are errors, and the key it stamps (the toolchain for cuda-bindings;
  CUDA major, toolchain, debug and coverage for cuda-core). Those tests stay
  in the package's own file.
- Anything that a normal wheel build already proves, such as the platform
  default toolchain resolving, or the llvm preflight being a no-op for the
  default. A regression there breaks every wheel build.

Tests that change ``force_build_ext`` do so on ``build_shared`` itself.
``build_hooks`` re-exports the flag through a module ``__getattr__``, so
``monkeypatch.setattr(build_hooks, "force_build_ext", ...)`` would, on teardown,
leave a plain attribute on ``build_hooks`` that shadows the re-export for every
later test.
"""

import os
import shlex
import sys
import sysconfig

import pytest
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


def _fake_sysconfig(monkeypatch, build_shared, **values):
    """Pin sysconfig.get_config_var so linker-command assertions are exact."""
    monkeypatch.setattr(build_shared.sysconfig, "get_config_var", lambda name: values.get(name))


class ResolveToolchainMixin:
    """``resolve_toolchain`` behavior that does not depend on a package's choices.

    Subclasses set ``build_shared`` (the loaded ``_build_shared`` module). The
    ``cxx_std`` passed here is arbitrary: these tests exercise the environment,
    name and shared flag mechanics. Each package's own tests assert the
    standard and the tweaks that it chooses.
    """

    build_shared = None
    cxx_std = 17

    def _resolve(self, **kwargs):
        return self.build_shared.resolve_toolchain(cxx_std=self.cxx_std, **kwargs)

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_does_not_touch_env(self, monkeypatch):
        # The default toolchain must leave the compiler environment alone: an
        # externally-set compiler (e.g. the sccache wrapper in CI) survives.
        for name in ("CUDA_PYTHON_TOOLCHAIN", "CC", "CXX", "LDSHARED", "LDCXXSHARED"):
            monkeypatch.delenv(name, raising=False)
        self._resolve()
        for name in ("CC", "CXX", "LDSHARED", "LDCXXSHARED"):
            assert name not in os.environ

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_default_preserves_existing_cc(self, monkeypatch):
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        self._resolve()
        assert os.environ["CC"] == "sccache cc"
        assert os.environ["CXX"] == "sccache c++"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_case_insensitive(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "LLVM")
        name, _cc, _cxx, _cargs, _largs = self._resolve()
        assert name == "llvm"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "icc")
        with pytest.raises(RuntimeError, match="not supported"):
            self._resolve()

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_sets_env_and_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        _fake_sysconfig(monkeypatch, self.build_shared, LDCXXSHARED="g++ -shared -Wl,-O1")
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        name, cc, cxx, cargs, largs = self._resolve()
        assert name == "llvm"
        assert (cc, cxx) == ("clang", "clang++")
        assert os.environ["CC"] == "clang"
        assert os.environ["CXX"] == "clang++"
        assert os.environ["LDCXXSHARED"] == "clang++ -shared -Wl,-O1"
        assert "LDSHARED" not in os.environ
        assert "-fuse-ld=lld" in largs
        # clang rejects the gcc-only flags that gnu used to use; they must be absent.
        assert "-fpermissive" not in cargs
        assert "-fno-var-tracking-assignments" not in cargs

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_gnu_sets_env_and_flags(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "gnu")
        _fake_sysconfig(monkeypatch, self.build_shared, LDCXXSHARED="x86_64-linux-gnu-g++ -shared -Wl,-O1")
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        monkeypatch.delenv("LDSHARED", raising=False)
        name, cc, cxx, cargs, _largs = self._resolve()
        assert name == "gnu"
        assert (cc, cxx) == ("gcc", "g++")
        assert os.environ["CC"] == "gcc"
        assert os.environ["CXX"] == "g++"
        assert os.environ["LDCXXSHARED"] == "g++ -shared -Wl,-O1"
        assert "LDSHARED" not in os.environ
        assert "-fpermissive" not in cargs
        assert "-fno-var-tracking-assignments" not in cargs

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_llvm_keeps_sccache_prefix(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        _fake_sysconfig(monkeypatch, self.build_shared, LDCXXSHARED="g++ -shared -Wl,-O1")
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        self._resolve()
        assert os.environ["CC"] == "sccache clang"
        assert os.environ["CXX"] == "sccache clang++"
        assert "LDSHARED" not in os.environ
        # The launcher prefixes CC/CXX only; the shared linker command is the bare compiler.
        assert os.environ["LDCXXSHARED"] == "clang++ -shared -Wl,-O1"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_explicit_toolchain_prefers_env_ldcxxshared(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu/llvm only valid on Linux")
        _fake_sysconfig(monkeypatch, self.build_shared, LDCXXSHARED="g++ -shared -Wl,-O1")
        monkeypatch.setenv("LDCXXSHARED", "g++ -shared -Wl,-rpath,/user/lib")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        self._resolve()
        assert os.environ["LDCXXSHARED"] == "clang++ -shared -Wl,-rpath,/user/lib"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_explicit_toolchain_falls_back_to_ldshared_then_shared_flag(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("gnu/llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        _fake_sysconfig(monkeypatch, self.build_shared, LDSHARED="gcc -shared -Wl,-z,relro")
        self._resolve()
        assert os.environ["LDCXXSHARED"] == "clang++ -shared -Wl,-z,relro"
        _fake_sysconfig(monkeypatch, self.build_shared)
        os.environ.pop("LDCXXSHARED")
        self._resolve()
        assert os.environ["LDCXXSHARED"] == "clang++ -shared"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_linux_opt_flag_set(self, monkeypatch):
        """The one Linux opt flag set: -std, -g0 -O2, stripped link; no -O3 or gcc-only flags."""
        if sys.platform == "win32":
            pytest.skip("Linux flags only")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, largs = self._resolve(debug=False)
        assert f"-std=c++{self.cxx_std}" in cargs
        assert "-g0" in cargs
        assert "-O2" in cargs
        assert "-O3" not in cargs
        assert "-fpermissive" not in cargs
        assert "-fno-var-tracking-assignments" not in cargs
        assert "-Wl,--strip-all" in largs

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_msvc_opt_flag_set(self, monkeypatch):
        """Modern setuptools no longer forces /Ox, so /O2 must be emitted explicitly."""
        if sys.platform != "win32":
            pytest.skip("MSVC flags only on Windows")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, _largs = self._resolve(debug=False)
        assert f"/std:c++{self.cxx_std}" in cargs
        assert "/O2" in cargs

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_warnings_as_errors_is_opt_in(self, monkeypatch):
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, _largs = self._resolve()
        assert "-Werror" not in cargs
        assert "/WX" not in cargs

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_warnings_as_errors_flags_per_platform(self, monkeypatch):
        # A wheel build passes without these flags, so a refactor could drop
        # them and silently disable the Werror gate that CI relies on.
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        _name, _cc, _cxx, cargs, _largs = self._resolve(warnings_as_errors=True)
        if sys.platform == "win32":
            assert {"/WX", "/wd4551", "/wd4244"} <= set(cargs)
        else:
            assert "-Werror" in cargs


class WithSccacheMixin:
    """``_with_sccache``: keep a leading sccache token, swap the compiler."""

    build_shared = None

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_sccache_and_swaps_compiler(self):
        assert self.build_shared._with_sccache("sccache cc", "clang") == "sccache clang"

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_absolute_sccache_path(self):
        assert (
            self.build_shared._with_sccache("/host/usr/local/bin/sccache cc", "clang")
            == "/host/usr/local/bin/sccache clang"
        )

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_bare_or_unrelated_cc_returns_compiler(self):
        assert self.build_shared._with_sccache("", "clang") == "clang"
        assert self.build_shared._with_sccache("gcc", "clang") == "clang"
        assert self.build_shared._with_sccache("ccache gcc", "clang") == "clang"


class WithCompilerMixin:
    """``_with_compiler``: replace the compiler executable, keep the following flags."""

    build_shared = None

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_keeps_flags_that_were_part_of_sysconfig_cxx(self):
        assert (
            self.build_shared._with_compiler("g++ -pthread -B /compat -shared -Wl,-rpath,/lib", "clang++")
            == "clang++ -pthread -B /compat -shared -Wl,-rpath,/lib"
        )

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_compiler_only_command_returns_compiler(self):
        assert self.build_shared._with_compiler("g++", "clang++") == "clang++"
        assert self.build_shared._with_compiler("", "clang++") == "clang++"
        assert self.build_shared._with_compiler(None, "clang++") == "clang++"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_keeps_quoted_arguments_intact(self):
        result = self.build_shared._with_compiler("g++ -Wl,-rpath='/a  b' -shared", "clang++")
        assert shlex.split(result) == ["clang++", "-Wl,-rpath=/a  b", "-shared"]

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_drops_launcher_before_compiler(self):
        # setuptools takes the launcher from CXX; keeping a second copy here would
        # leave a stray compiler argument on the link line.
        assert self.build_shared._with_compiler("ccache g++ -shared", "clang++") == "clang++ -shared"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_keeps_env_prefix(self):
        assert (
            self.build_shared._with_compiler("env LIBRARY_PATH=/custom/lib g++ -shared", "clang++")
            == "env LIBRARY_PATH=/custom/lib clang++ -shared"
        )
        # env long options (--unset=VAR) also treated as prefix, same as setuptools' _split_env
        assert (
            self.build_shared._with_compiler("env --unset=LD_LIBRARY_PATH g++ -shared", "clang++")
            == "env --unset=LD_LIBRARY_PATH clang++ -shared"
        )


class DistutilsLinkerIntegrationMixin:
    """The env set by ``resolve_toolchain``, as setuptools' distutils consumes it.

    ``ResolveToolchainMixin`` checks os.environ; this checks the linker commands
    distutils derives from it, which is what reaches the C++ link step.
    """

    build_shared = None
    cxx_std = 17

    def _resolve(self):
        return self.build_shared.resolve_toolchain(cxx_std=self.cxx_std)

    @staticmethod
    def _customized_compiler():
        compiler = new_compiler()
        customize_compiler(compiler)
        if not hasattr(compiler, "linker_so_cxx"):
            pytest.skip("this setuptools' distutils has no linker_so_cxx")
        return compiler

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_linker_so_cxx_swaps_compiler_and_keeps_sysconfig_flags(self, monkeypatch):
        if sys.platform != "linux":
            pytest.skip("gnu/llvm only valid on Linux")
        sysconfig_ld = sysconfig.get_config_var("LDCXXSHARED")
        if not sysconfig_ld:
            pytest.skip("this Python has no LDCXXSHARED")
        # Everything from the first flag on, including operands such as ``-B /path``.
        tokens = shlex.split(sysconfig_ld)
        first_flag = next((i for i, tok in enumerate(tokens) if tok.startswith("-")), len(tokens))
        expected_tail = tokens[first_flag:]
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        self._resolve()
        linker = self._customized_compiler().linker_so_cxx
        assert linker[0] == "clang++"
        assert linker[1 : 1 + len(expected_tail)] == expected_tail

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_linker_so_cxx_keeps_split_option_operands(self, monkeypatch):
        if sys.platform != "linux":
            pytest.skip("gnu/llvm only valid on Linux")
        monkeypatch.setenv("LDCXXSHARED", "g++ -pthread -B /path/to/python_compiler_compat -shared -Wl,-rpath,/lib")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        self._resolve()
        linker = self._customized_compiler().linker_so_cxx
        assert linker[:6] == [
            "clang++",
            "-pthread",
            "-B",
            "/path/to/python_compiler_compat",
            "-shared",
            "-Wl,-rpath,/lib",
        ]

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_linker_so_cxx_keeps_env_prefix(self, monkeypatch):
        if sys.platform != "linux":
            pytest.skip("gnu/llvm only valid on Linux")
        monkeypatch.setenv("LDCXXSHARED", "env LIBRARY_PATH=/custom/lib g++ -shared")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        self._resolve()
        linker = self._customized_compiler().linker_so_cxx
        assert linker[:3] == ["env", "LIBRARY_PATH=/custom/lib", "clang++"]
        assert "g++" not in linker

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_sccache_does_not_duplicate_compiler_in_cxx_linker(self, monkeypatch):
        if sys.platform != "linux":
            pytest.skip("gnu/llvm only valid on Linux")
        monkeypatch.setenv("CUDA_PYTHON_TOOLCHAIN", "llvm")
        monkeypatch.setenv("CC", "sccache cc")
        monkeypatch.setenv("CXX", "sccache c++")
        self._resolve()
        compiler = self._customized_compiler()
        assert compiler.compiler_cxx[:2] == ["sccache", "clang++"]
        linker = compiler.linker_so_cxx
        assert linker[0] == "clang++"
        assert linker.count("clang++") == 1


class CheckToolchainAvailableMixin:
    """``_check_toolchain_available``: a helpful failure when an llvm tool is missing.

    The platform default never preflights, and a wheel build proves that, so
    only the error message and the llvm-tools-present case are tested.
    """

    build_shared = None

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_missing_tool_lists_install_hint(self, monkeypatch):
        def fake_which(name):
            return None if name in ("clang", "clang++", "ld.lld") else "/bin/" + name

        monkeypatch.setattr(self.build_shared.shutil, "which", fake_which)
        with pytest.raises(RuntimeError, match="clang and lld"):
            self.build_shared._check_toolchain_available("llvm")

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_llvm_present_passes(self, monkeypatch):
        monkeypatch.setattr(self.build_shared.shutil, "which", lambda name: "/bin/" + name)
        self.build_shared._check_toolchain_available("llvm")


class AbiStampPathMixin:
    """``_abi_stamp_path`` scopes stamp files by Python's EXT_SUFFIX.

    The stem is arbitrary: each package uses its own at run time
    (``.build-toolchain`` for cuda-bindings, ``.build-config`` for cuda-core),
    but the mechanism is shared.
    """

    build_shared = None

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_stamp_path_is_scoped_to_extension_abi(self, monkeypatch):
        monkeypatch.setattr(
            self.build_shared.sysconfig, "get_config_var", lambda _name: ".cpython-310-x86_64-linux-gnu.so"
        )
        python_310 = self.build_shared._abi_stamp_path(".build-test")
        monkeypatch.setattr(
            self.build_shared.sysconfig, "get_config_var", lambda _name: ".cpython-311-x86_64-linux-gnu.so"
        )
        python_311 = self.build_shared._abi_stamp_path(".build-test")

        assert python_310 != python_311
        assert python_310.name == ".build-test.cpython-310-x86_64-linux-gnu.so"
        assert python_311.name == ".build-test.cpython-311-x86_64-linux-gnu.so"

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_missing_ext_suffix_is_an_error(self, monkeypatch):
        monkeypatch.setattr(self.build_shared.sysconfig, "get_config_var", lambda _name: None)
        with pytest.raises(RuntimeError, match="EXT_SUFFIX"):
            self.build_shared._abi_stamp_path(".build-test")


class BuildKeyStampMixin:
    """``check_build_key`` and ``record_build_key``: the stamp-and-force protocol.

    The tests use a scratch stamp, so they do not depend on what key a package
    stamps. Those keys are tested in the package's own file.
    """

    build_shared = None

    @pytest.fixture(autouse=True)
    def _reset_force_build_ext(self, monkeypatch):
        monkeypatch.setattr(self.build_shared, "force_build_ext", False)

    @pytest.fixture
    def stamp_file(self, tmp_path):
        return tmp_path / "build" / ".build-test"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_missing_stamp_forces_rebuild(self, stamp_file):
        self.build_shared.check_build_key(stamp_file, "gnu", "Toolchain")
        assert self.build_shared.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_same_key_does_not_force(self, stamp_file):
        self.build_shared.record_build_key(stamp_file, "gnu")
        self.build_shared.check_build_key(stamp_file, "gnu", "Toolchain")
        assert self.build_shared.force_build_ext is False

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_key_forces_rebuild(self, stamp_file):
        self.build_shared.record_build_key(stamp_file, "gnu")
        self.build_shared.check_build_key(stamp_file, "llvm", "Toolchain")
        assert self.build_shared.force_build_ext is True

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_a_later_match_does_not_clear_the_flag(self, stamp_file):
        # Once one check asks for a full rebuild, a second, unchanged check must not retract it.
        self.build_shared.check_build_key(stamp_file, "gnu", "Toolchain")
        self.build_shared.record_build_key(stamp_file, "gnu")
        self.build_shared.check_build_key(stamp_file, "gnu", "Toolchain")
        assert self.build_shared.force_build_ext is True

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_message_names_the_previous_and_the_new_key(self, stamp_file, capsys):
        self.build_shared.record_build_key(stamp_file, "gnu")
        self.build_shared.check_build_key(stamp_file, "llvm", "Toolchain")
        out = capsys.readouterr().out
        assert "Toolchain of last build: gnu (building llvm); forcing a full rebuild" in out

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_record_creates_the_directory_and_writes_the_key(self, stamp_file):
        assert not stamp_file.parent.exists()
        self.build_shared.record_build_key(stamp_file, "cu13-gnu-opt")
        assert stamp_file.read_text(encoding="utf-8") == "cu13-gnu-opt\n"


class ForceBuildExtReexportMixin:
    """``build_hooks.force_build_ext`` is a live view of ``build_shared.force_build_ext``.

    setup.py reads the flag as ``build_hooks.force_build_ext``. The flag is
    owned by ``_build_shared``; ``build_hooks`` re-exports it. Subclasses set
    both ``build_hooks`` and ``build_shared``.
    """

    build_hooks = None
    build_shared = None

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_build_hooks_follows_the_shared_flag(self, monkeypatch):
        monkeypatch.setattr(self.build_shared, "force_build_ext", True)
        assert self.build_hooks.force_build_ext is True
        monkeypatch.setattr(self.build_shared, "force_build_ext", False)
        assert self.build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_build_hooks_holds_no_copy_of_the_flag(self):
        # A plain attribute would shadow the module __getattr__ and go stale.
        assert "force_build_ext" not in vars(self.build_hooks)

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_other_names_still_raise_attribute_error(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            self.build_hooks.no_such_name  # noqa: B018
