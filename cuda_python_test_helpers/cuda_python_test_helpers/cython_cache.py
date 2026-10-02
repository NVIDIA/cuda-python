# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tests for the Cython cache helpers in build_hooks.py.

Provides:

- ``cython_cache_miss_then_hit``: a real-cythonize miss/hit smoke test.
- ``CythonCachePathMixin``: the common unit tests for ``_cython_cache_path``.
- ``CythonAliasMixin``: the common unit tests for ``_stable_cython_alias``.
- ``POSIX_ONLY_CACHE``: shared skip marker for tests that need caching/aliasing
  actually enabled (both are unconditionally disabled on win32).
- ``WINDOWS_ONLY_CACHE``: shared skip marker for the complementary Windows
  disable-path tests (warn + return None when the cache dir is set).

These are used by ``cuda_bindings/tests/test_build_hooks.py``
and ``cuda_core/tests/test_build_hooks.py``. Drift between the two vendored
helper copies is enforced by ``toolshed/check_build_hooks_sync.py``, not by a
runtime test.

Cython is imported inside the functions that need it so this module does not
force a Cython dependency on the ``cuda-python-test-helpers`` package.
"""

import os
import shutil
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cuda_python_test_helpers.subprocess_runner import run_python_snippet

# `_cython_cache_path()` unconditionally returns None on win32 (symlink
# aliasing needs elevated privileges/Developer Mode there), so any test that
# sets CUDA_PYTHON_CYTHON_CACHE_DIR and expects a real cache path/alias, or
# that calls `_stable_cython_alias()` directly, would fail deterministically
# on a Windows runner. Share one marker/reason so it's applied consistently.
POSIX_ONLY_CACHE = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Cython caching and symlink aliasing are POSIX-only (disabled on win32)",
)
WINDOWS_ONLY_CACHE = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only: Cython caching is intentionally disabled on win32",
)


def cython_cache_miss_then_hit(cache_path, tmp_path, capsys):
    """Run a real Cython cache miss/hit through ``cache_path``.

    Uses ``from libc.stdint cimport uint64_t`` to exercise a stdlib
    dependency resolved through Cython's include path.  The cold run
    generates ``mod.c`` and populates the cache; the warm run (same alias,
    same contents) restores ``mod.c`` from the cache without regenerating.

    ``cache_path`` is a directory path produced by the build hook's
    ``_cython_cache_path`` helper.  ``tmp_path`` and ``capsys`` are
    standard pytest fixtures.
    """
    from Cython.Build import cythonize
    from setuptools import Extension

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mod.pyx").write_text("from libc.stdint cimport uint64_t\ndef get_value() -> uint64_t:\n    return 42\n")

    ext = Extension("mod", sources=[str(src_dir / "mod.pyx")], language="c")

    # Cold run: no .c exists, cythonize generates mod.c and populates the cache.
    capsys.readouterr()
    cythonize([ext], cache=cache_path, quiet=False)
    out = capsys.readouterr().out
    assert "Cythonizing" in out, "cold run should cythonize"
    assert "Found compiled" not in out
    gen_c = src_dir / "mod.c"
    assert gen_c.exists(), "cold run did not generate mod.c"
    cold_bytes = gen_c.read_bytes()

    # Warm run: delete the .c so cythonize reconsiders; cache hit restores it.
    gen_c.unlink()
    capsys.readouterr()
    cythonize([ext], cache=cache_path, quiet=False)
    out = capsys.readouterr().out
    assert "Found compiled" in out, "warm run should hit cache"
    assert gen_c.exists(), "warm run did not restore mod.c from cache"
    assert gen_c.read_bytes() == cold_bytes, "warm run changed mod.c"


class CythonAliasMixin:
    """Common unit tests for ``_stable_cython_alias``.

    Subclasses set ``build_hooks`` (the loaded build_hooks module). Tests pass
    an *absolute* alias path inside the per-test ``tmp_path`` sandbox, so the
    helper's relative-alias anchoring (package directory) is not exercised
    here; the production build hooks cover that path.
    """

    build_hooks = None

    def _alias(self, tmp_path, name=".cython-stdlib"):
        # An absolute alias lands in the per-test sandbox (tmp_path), not the
        # package directory, so a crash can never litter the source tree.
        return tmp_path / name

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_alias_created_and_removed_on_success(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        alias = self._alias(tmp_path)
        with self.build_hooks._stable_cython_alias(target, alias):
            assert alias.is_symlink()
            assert alias.resolve() == target.resolve()
        assert not alias.exists()
        assert not alias.is_symlink()

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_alias_removed_on_exception(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        alias = self._alias(tmp_path)
        with pytest.raises(RuntimeError, match="boom"), self.build_hooks._stable_cython_alias(target, alias):
            raise RuntimeError("boom")
        assert not alias.is_symlink()

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_stale_dangling_symlink_atomically_replaced(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        old_target = tmp_path / "gone"  # does not exist => dangling
        alias = self._alias(tmp_path)
        alias.symlink_to(old_target)
        assert not alias.exists()
        assert alias.is_symlink()
        with self.build_hooks._stable_cython_alias(target, alias) as rel:
            assert alias.resolve() == target.resolve()
            assert rel  # non-empty relative path
        assert not alias.is_symlink()

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_real_directory_raises_without_clobbering(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        alias = self._alias(tmp_path)
        alias.mkdir(exist_ok=True)  # real directory, not a symlink
        with (
            pytest.raises(RuntimeError, match="already exists"),
            self.build_hooks._stable_cython_alias(target, alias),
        ):
            pass
        assert alias.is_dir()
        assert not alias.is_symlink()

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_no_filesystem_changes_when_cache_disabled(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CUDA_PYTHON_CYTHON_CACHE_DIR", raising=False)
        # When cache is disabled _cython_cache_path returns None, so the
        # build hook skips alias creation entirely.
        cache = self.build_hooks._cython_cache_path("cuda-bindings")
        assert cache is None
        alias = self._alias(tmp_path)
        assert not alias.exists(), f"unexpected alias at {alias}"

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_yielded_value_is_relative_path(self, tmp_path, monkeypatch):
        target = tmp_path / "real"
        target.mkdir()
        alias = self._alias(tmp_path)
        monkeypatch.chdir(tmp_path)
        with self.build_hooks._stable_cython_alias(target, alias) as rel:
            assert not os.path.isabs(rel), f"expected relative path, got {rel!r}"
            assert Path(rel).resolve() == target.resolve()

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_tmp_sibling_cleaned_on_replace_failure(self, tmp_path, monkeypatch):
        """The temporary atomic-replacement sibling is removed even if os.replace fails."""
        target = tmp_path / "real"
        target.mkdir()
        alias = self._alias(tmp_path)

        def failing_replace(_src, _dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated"), self.build_hooks._stable_cython_alias(target, alias):
            pass
        tmp_files = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert tmp_files == [], f"leftover tmp siblings: {tmp_files}"

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_stable_alias_gives_cache_hit_across_isolated_envs(self, tmp_path):
        """Fingerprint stays stable when the symlink target moves to a new random path.

        Simulates what happens across two consecutive PEP 517 builds: the
        Cython stdlib (libc/stdint.pxd) is installed under a different temporary
        prefix but the stable alias stays at the same worktree-relative path.
        The second subprocess should print "Found compiled" (cache hit) rather
        than "Cythonizing".
        """
        import Cython

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        real_includes = Path(Cython.__file__).parent / "Includes"
        env_a = tmp_path / "env_a" / "Includes"
        env_b = tmp_path / "env_b" / "Includes"
        shutil.copytree(real_includes, env_a)
        shutil.copytree(real_includes, env_b)

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "mod.pyx").write_text("from libc.stdint cimport uint64_t\ndef f() -> uint64_t:\n    return 1\n")

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        alias = pkg_dir / ".cython-stdlib"

        def _run(target):
            script = textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                from Cython.Build import cythonize
                from setuptools import Extension
                target = Path({str(target)!r})
                alias = Path({str(alias)!r})
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to(target, target_is_directory=True)
                rel_alias = os.path.relpath(alias)
                ext = Extension("mod", sources=[{str(src_dir / "mod.pyx")!r}], language="c")
                gen_c = Path({str(src_dir / "mod.c")!r})
                if gen_c.exists():
                    gen_c.unlink()
                cythonize([ext], cache={str(cache_dir)!r}, include_path=[".", rel_alias], quiet=False)
                alias.unlink()
                """
            )
            return run_python_snippet(script, cwd=tmp_path)

        r1 = _run(env_a)
        assert "Cythonizing" in r1.stdout, "expected cold Cythonizing"

        r2 = _run(env_b)
        assert "Found compiled" in r2.stdout, f"expected cache hit but got:\n{r2.stdout}"


class CythonCachePathMixin:
    """Common unit tests for the ``_cython_cache_path`` build-hook helper.

    Subclasses set the class attributes ``build_hooks`` (the loaded
    ``build_hooks`` module) and ``package`` (the cache namespace label,
    e.g. ``"cuda-core"``) and inherit the shared tests. Per-package
    extras (e.g. ``compile_time_env`` for cuda_core) live on the
    subclass.
    """

    build_hooks = None
    package = None

    def _set_env(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("CUDA_PYTHON_CYTHON_CACHE_DIR", raising=False)
        else:
            monkeypatch.setenv("CUDA_PYTHON_CYTHON_CACHE_DIR", value)

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_unset_env_returns_none(self, monkeypatch):
        """An unset ``CUDA_PYTHON_CYTHON_CACHE_DIR`` disables caching (returns None)."""
        self._set_env(monkeypatch, None)
        assert self.build_hooks._cython_cache_path(self.package) is None

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_set_env_returns_namespaced_path(self, monkeypatch, tmp_path):
        """A set cache root yields a path namespaced by package and config."""
        self._set_env(monkeypatch, str(tmp_path))
        path = self.build_hooks._cython_cache_path(self.package)
        assert path is not None
        assert path.startswith(str(tmp_path))
        assert f"{self.package}-" in path

    @WINDOWS_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_windows_cache_dir_warns_and_returns_none(self, monkeypatch, tmp_path):
        """A set ``CUDA_PYTHON_CYTHON_CACHE_DIR`` warns and returns None on Windows."""
        self._set_env(monkeypatch, str(tmp_path))
        with pytest.warns(UserWarning, match="not supported on Windows"):
            assert self.build_hooks._cython_cache_path(self.package) is None

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_windows_disable_path_warns_and_returns_none(self, monkeypatch, tmp_path):
        """The win32 branch of ``_cython_cache_path`` warns and returns None.

        Complements ``test_windows_cache_dir_warns_and_returns_none`` so Linux
        CI still covers the disable path (that test is Windows-only).
        """
        self._set_env(monkeypatch, str(tmp_path))
        with (
            mock.patch.object(self.build_hooks.sys, "platform", "win32"),
            pytest.warns(UserWarning, match="not supported on Windows"),
        ):
            assert self.build_hooks._cython_cache_path(self.package) is None

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_package_namespaces_are_distinct(self, monkeypatch, tmp_path):
        """Different package names map to different cache directories."""
        self._set_env(monkeypatch, str(tmp_path))
        a = self.build_hooks._cython_cache_path("cuda-bindings")
        b = self.build_hooks._cython_cache_path("cuda-core")
        assert a != b

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_identical_settings_produce_same_path(self, monkeypatch, tmp_path):
        """Dict insertion order does not affect the namespace (sorting makes it stable)."""
        self._set_env(monkeypatch, str(tmp_path))
        directives = {"embedsignature": True, "linetrace": True}
        p1 = self.build_hooks._cython_cache_path(
            self.package,
            compiler_directives=directives,
            language_level=3,
            cplus=True,
            debug=False,
        )
        p2 = self.build_hooks._cython_cache_path(
            self.package,
            compiler_directives=dict(reversed(list(directives.items()))),
            language_level=3,
            cplus=True,
            debug=False,
        )
        assert p1 == p2

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_changed_directives_change_namespace(self, monkeypatch, tmp_path):
        """Different ``compiler_directives`` map to different namespaces (#7532 workaround)."""
        self._set_env(monkeypatch, str(tmp_path))
        base = dict(embedsignature=True, freethreading_compatible=True)
        with_linetrace = dict(base, linetrace=True)
        p1 = self.build_hooks._cython_cache_path(self.package, compiler_directives=base)
        p2 = self.build_hooks._cython_cache_path(self.package, compiler_directives=with_linetrace)
        assert p1 != p2

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_path_has_no_workspace_component(self, monkeypatch, tmp_path):
        """The namespace tail is just ``<package>-<hex>``, with no checkout path."""
        self._set_env(monkeypatch, str(tmp_path))
        path = self.build_hooks._cython_cache_path(
            self.package,
            compiler_directives={"linetrace": True},
            language_level=3,
            cplus=True,
        )
        tail = path[len(str(tmp_path)) + 1 :]
        assert tail.count(os.sep) == 0, f"namespace tail has path separators: {tail!r}"
        assert tail.startswith(f"{self.package}-"), f"unexpected namespace tail: {tail!r}"

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_different_python_versions_map_to_distinct_paths(self, monkeypatch, tmp_path):
        """Different Python interpreter versions map to different namespaces.

        Cython's generated C code is Python-version-specific (3.14
        switches to zstd-compressed string literals via ``CYTHON_COMPRESS_STRINGS``,
        while 3.12/3.13 use zlib), so the interpreter major.minor
        must partition the cache namespace.
        """
        self._set_env(monkeypatch, str(tmp_path))
        with mock.patch.object(sys, "version_info", SimpleNamespace(major=3, minor=12)):
            p312 = self.build_hooks._cython_cache_path(self.package)
        with mock.patch.object(sys, "version_info", SimpleNamespace(major=3, minor=13)):
            p313 = self.build_hooks._cython_cache_path(self.package)
        with mock.patch.object(sys, "version_info", SimpleNamespace(major=3, minor=14)):
            p314 = self.build_hooks._cython_cache_path(self.package)
        assert p312 != p313
        assert p312 != p314
        assert p313 != p314
