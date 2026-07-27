# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_hooks.py build infrastructure.

These tests verify the CUDA version detection logic used during builds,
particularly the _determine_cuda_major_version() function which derives the
CUDA major version from headers.

Note: These tests do NOT require cuda.core to be built/installed since they
test build-time infrastructure. Run with --noconftest to avoid loading
conftest.py which imports cuda.core modules:

    pytest tests/test_build_hooks.py -v --noconftest

These tests require Cython to be installed (build_hooks.py imports it).
"""

import builtins
import importlib.util
import os
import sys
import tempfile
import threading
import types
from distutils.ccompiler import CCompiler
from pathlib import Path
from unittest import mock

# build_hooks.py imports Cython and setuptools at the top level; both are
# declared test dependencies, so a missing install must surface as an
# ImportError at collection time rather than being hidden by importorskip.
import Cython  # noqa: F401
import pytest
import setuptools  # noqa: F401

from cuda.pathfinder import get_cuda_path_or_home


def _load_module(name, path, *, register=False):
    """Load a module from source without permanently modifying sys.path.

    build_hooks.py and _build_shared.py are PEP 517 backend files, not
    installed modules. We use importlib to load them directly from source to
    avoid polluting sys.path with the package directory (which contains
    cuda/ source that could shadow the installed package). With ``register``
    the module is also entered into sys.modules, which is how build_hooks.py's
    ``from _build_shared import ...`` finds this copy.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the modules once at import time; _build_shared must come first.
_PACKAGE_ROOT = Path(__file__).parent.parent
_build_shared = _load_module("_build_shared", _PACKAGE_ROOT / "_build_shared.py", register=True)
build_hooks = _load_module("build_hooks", _PACKAGE_ROOT / "build_hooks.py")


@pytest.fixture(autouse=True)
def _isolate_toolchain_env():
    names = ("CUDA_PYTHON_TOOLCHAIN", "CC", "CXX", "LDSHARED", "LDCXXSHARED", "CUDA_PYTHON_CYTHON_CACHE_DIR")
    original = {name: os.environ[name] for name in names if name in os.environ}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(original)


@pytest.mark.agent_authored(model="grok-4.6")
def test_get_cccl_include_dirs_requires_initialized_submodule(tmp_path, monkeypatch):
    monkeypatch.setattr(build_hooks, "_CCCL_SUBMODULE_DIR", tmp_path / "cccl")
    build_hooks._get_cccl_include_dirs.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="CCCL submodule"):
            build_hooks._get_cccl_include_dirs()
    finally:
        build_hooks._get_cccl_include_dirs.cache_clear()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cuda_path_is_resolved_before_importing_bindings(monkeypatch):
    """PEP 517 namespace repair runs before cuda.bindings is imported."""
    events = []

    class StopBuildError(Exception):
        pass

    def get_cuda_path():
        events.append("cuda-path")
        return "/cuda"

    original_import = builtins.__import__

    def stop_at_bindings_import(name, *args, **kwargs):
        if name == "cuda.bindings":
            events.append("cuda-bindings")
            raise StopBuildError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(build_hooks, "_get_cuda_path", get_cuda_path)
    monkeypatch.setattr(builtins, "__import__", stop_at_bindings_import)

    with pytest.raises(StopBuildError):
        build_hooks._build_cuda_core()

    assert events == ["cuda-path", "cuda-bindings"]


def _check_version_detection(
    cuda_version, expected_major, *, use_cuda_path=True, use_cuda_home=False, cuda_core_build_major=None
):
    """Test version detection with a mock cuda.h.

    Args:
        cuda_version: CUDA_VERSION to write in mock cuda.h (e.g., 12080)
        expected_major: Expected return value (e.g., "12")
        use_cuda_path: If True, set CUDA_PATH to the mock headers directory
        use_cuda_home: If True, set CUDA_HOME to the mock headers directory
        cuda_core_build_major: If set, override with this CUDA_CORE_BUILD_MAJOR env var
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        include_dir = Path(tmpdir) / "include"
        include_dir.mkdir()
        cuda_h = include_dir / "cuda.h"
        cuda_h.write_text(f"#define CUDA_VERSION {cuda_version}\n")

        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()

        mock_env = {
            k: v
            for k, v in {
                "CUDA_CORE_BUILD_MAJOR": cuda_core_build_major,
                "CUDA_PATH": tmpdir if use_cuda_path else None,
                "CUDA_HOME": tmpdir if use_cuda_home else None,
            }.items()
            if v is not None
        }

        with mock.patch.dict(os.environ, mock_env, clear=True):
            result = build_hooks._determine_cuda_major_version()
            assert result == expected_major


class TestGetCudaMajorVersion:
    """Tests for _determine_cuda_major_version()."""

    @pytest.mark.parametrize("version", ["11", "12", "13", "14"])
    def test_env_var_override(self, version):
        """CUDA_CORE_BUILD_MAJOR env var override works with various versions."""
        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()
        with mock.patch.dict(os.environ, {"CUDA_CORE_BUILD_MAJOR": version}, clear=False):
            result = build_hooks._determine_cuda_major_version()
            assert result == version

    @pytest.mark.parametrize(
        ("cuda_version", "expected_major"),
        [
            (11000, "11"),  # CUDA 11.0
            (11080, "11"),  # CUDA 11.8
            (12000, "12"),  # CUDA 12.0
            (12020, "12"),  # CUDA 12.2
            (12080, "12"),  # CUDA 12.8
            (13000, "13"),  # CUDA 13.0
            (13010, "13"),  # CUDA 13.1
        ],
        ids=["11.0", "11.8", "12.0", "12.2", "12.8", "13.0", "13.1"],
    )
    def test_cuda_headers_parsing(self, cuda_version, expected_major):
        """CUDA_VERSION is correctly parsed from cuda.h headers."""
        _check_version_detection(cuda_version, expected_major)

    def test_cuda_home_fallback(self):
        """CUDA_HOME is used if CUDA_PATH is not set."""
        _check_version_detection(12050, "12", use_cuda_path=False, use_cuda_home=True)

    def test_env_var_takes_priority_over_headers(self):
        """Env var override takes priority even when headers exist."""
        _check_version_detection(12080, "11", cuda_core_build_major="11")

    def test_missing_cuda_path_raises_error(self):
        """RuntimeError is raised when CUDA_PATH/CUDA_HOME not set and no env var override."""
        build_hooks._get_cuda_path.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        get_cuda_path_or_home.cache_clear()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match="CUDA_PATH or CUDA_HOME"),
        ):
            build_hooks._determine_cuda_major_version()


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    """Redirect the build-config stamp to a scratch path.

    _BUILD_CONFIG_STAMP is anchored to build_hooks.py rather than the
    working directory, so it has to be replaced outright; chdir would
    not move it, and record_build_config() would write into the
    real source tree.
    """
    scratch = tmp_path / "build" / ".build-config"
    monkeypatch.setattr(build_hooks, "_BUILD_CONFIG_STAMP", scratch)
    monkeypatch.setattr(_build_shared, "force_build_ext", False)
    build_hooks._get_cuda_path.cache_clear()
    build_hooks._determine_cuda_major_version.cache_clear()
    get_cuda_path_or_home.cache_clear()
    monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
    monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
    monkeypatch.delenv("CUDA_PYTHON_COVERAGE", raising=False)
    return scratch


def _write_stamp(stamp, config_key):
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(config_key + "\n")


class TestBuildConfigStamp:
    """Tests for _check_build_config() and record_build_config()."""

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_missing_stamp_forces_rebuild(self, stamp):
        # No stamp means the last build's config is unknown, so rebuild.
        cuda_major, key = build_hooks._check_build_config("gnu", False, False)
        assert cuda_major == "13"
        assert key == "cu13-gnu-opt"
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_same_config_does_not_force(self, stamp):
        _write_stamp(stamp, "cu13-gnu-opt")
        cuda_major, key = build_hooks._check_build_config("gnu", False, False)
        assert cuda_major == "13"
        assert build_hooks.force_build_ext is False

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_cuda_major_forces_rebuild(self, stamp):
        _write_stamp(stamp, "cu12-gnu-opt")
        cuda_major, _ = build_hooks._check_build_config("gnu", False, False)
        assert cuda_major == "13"
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_toolchain_forces_rebuild(self, stamp):
        _write_stamp(stamp, "cu13-gnu-opt")
        build_hooks._check_build_config("llvm", False, False)
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_debug_forces_rebuild(self, stamp):
        _write_stamp(stamp, "cu13-gnu-opt")
        build_hooks._check_build_config("gnu", True, False)
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_changed_coverage_forces_rebuild(self, stamp):
        _write_stamp(stamp, "cu13-gnu-opt")
        build_hooks._check_build_config("gnu", False, True)
        assert build_hooks.force_build_ext is True

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_record_writes_stamp(self, stamp):
        build_hooks.record_build_config("cu13-gnu-debug")
        assert stamp.read_text().strip() == "cu13-gnu-debug"


class TestBuildHookStamping:
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_wheel_records_exact_prepared_config_after_success(self, monkeypatch):
        events = []

        def prepare(debug):
            events.append(("prepare", debug))
            return "cu13-gnu-debug"

        def build(wheel_directory, config_settings, metadata_directory):
            events.append("build")
            return "cuda_core.whl"

        monkeypatch.setattr(build_hooks, "_build_cuda_core", prepare)
        monkeypatch.setattr(build_hooks._build_meta, "build_wheel", build)
        monkeypatch.setattr(build_hooks, "record_build_config", lambda key: events.append(("record", key)))

        wheel_name = build_hooks.build_wheel("dist", {"debug": True}, "metadata")

        assert wheel_name == "cuda_core.whl"
        assert events == [("prepare", True), "build", ("record", "cu13-gnu-debug")]

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_editable_records_default_config_after_patch(self, monkeypatch):
        events = []
        expected_debug = sys.platform != "win32"
        toolchain = "msvc" if sys.platform == "win32" else "gnu"
        expected_key = f"cu13-{toolchain}-{'debug' if expected_debug else 'opt'}"

        def prepare(debug):
            events.append(("prepare", debug))
            return expected_key

        monkeypatch.setattr(build_hooks, "_build_cuda_core", prepare)
        monkeypatch.setattr(
            build_hooks._build_meta,
            "build_editable",
            lambda *_args: events.append("build") or "cuda_core.whl",
        )
        monkeypatch.setattr(
            build_hooks,
            "_add_cython_include_paths_to_pth",
            lambda wheel_path: events.append(("patch", wheel_path)),
        )
        monkeypatch.setattr(build_hooks, "record_build_config", lambda key: events.append(("record", key)))

        wheel_name = build_hooks.build_editable("dist")

        assert wheel_name == "cuda_core.whl"
        assert events == [
            ("prepare", expected_debug),
            "build",
            ("patch", os.path.join("dist", "cuda_core.whl")),
            ("record", expected_key),
        ]

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_failed_wheel_build_does_not_record_config(self, monkeypatch):
        monkeypatch.setattr(
            build_hooks,
            "_build_cuda_core",
            lambda debug: f"cu13-gnu-{'debug' if debug else 'opt'}",
        )

        def fail(*args):
            raise RuntimeError("wheel build failed")

        monkeypatch.setattr(build_hooks._build_meta, "build_wheel", fail)
        monkeypatch.setattr(
            build_hooks,
            "record_build_config",
            lambda _key: pytest.fail("failed build must not be stamped"),
        )

        with pytest.raises(RuntimeError, match="wheel build failed"):
            build_hooks.build_wheel("dist", {"debug": True}, "metadata")

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_failed_editable_patch_does_not_record_config(self, monkeypatch):
        monkeypatch.setattr(
            build_hooks,
            "_build_cuda_core",
            lambda debug: f"cu13-gnu-{'debug' if debug else 'opt'}",
        )
        monkeypatch.setattr(build_hooks._build_meta, "build_editable", lambda *_args: "cuda_core.whl")

        def fail(wheel_path):
            raise RuntimeError("editable patch failed")

        monkeypatch.setattr(build_hooks, "_add_cython_include_paths_to_pth", fail)
        monkeypatch.setattr(
            build_hooks,
            "record_build_config",
            lambda _key: pytest.fail("unpatched editable build must not be stamped"),
        )

        with pytest.raises(RuntimeError, match="editable patch failed"):
            build_hooks.build_editable("dist", {"debug": True}, "metadata")


def _capture_cythonize_kwargs(monkeypatch, cuda_major):
    """Run the cythonize setup for one CUDA major and report its keyword arguments.

    cythonize() is replaced, so nothing is generated or compiled: this only
    observes how the build was configured.
    """
    captured = {}

    def fake_cythonize(ext_modules, **kwargs):
        captured.update(kwargs)
        return []

    # Builds resolve the CTK for include dirs; stub it so the test runs
    # where no toolkit is installed (e.g. the wheels CI jobs). The CCCL
    # submodule is also absent from those checkouts.
    monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: "/nonexistent-cuda")
    # The configuration check reads that header and the installed cuda-bindings.
    # TestBuildConfigurationCheck covers it.
    monkeypatch.setattr(build_hooks, "_check_build_configuration", lambda *_: None)
    monkeypatch.setattr(build_hooks, "_get_cccl_include_dirs", lambda: ["/nonexistent-cccl"])
    monkeypatch.setattr(build_hooks, "cythonize", fake_cythonize)
    monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", cuda_major)
    build_hooks._determine_cuda_major_version.cache_clear()
    # _build_cuda_core() globs cuda/core/**/*.pyx relative to the cwd.
    monkeypatch.chdir(Path(__file__).parent.parent)
    # It also prepends cuda_bindings/ to sys.path; swap in a copy so the
    # mutation lands there and the real list is restored on teardown.
    monkeypatch.setattr(sys, "path", list(sys.path))

    build_hooks._build_cuda_core()
    return captured


def _capture_cythonize_build_dir(monkeypatch, cuda_major):
    return Path(_capture_cythonize_kwargs(monkeypatch, cuda_major)["build_dir"])


class TestGeneratedSourceDirIsKeyed:
    """Generated C++ must not be shared between build configurations.

    Cython's up-to-date check does not hash compile_time_env or the
    extension flags, so without a per-config directory a cu13-gnu build's generated sources are handed to a cu13-llvm compiler (and vice versa).
    """

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_majors_use_different_dirs(self, monkeypatch):
        dir_12 = _capture_cythonize_build_dir(monkeypatch, "12")
        dir_13 = _capture_cythonize_build_dir(monkeypatch, "13")

        assert dir_12 != dir_13
        toolchain = "msvc" if sys.platform == "win32" else "gnu"
        assert dir_12.name == f"cu12-{toolchain}-opt"
        assert dir_13.name == f"cu13-{toolchain}-opt"

    @pytest.mark.agent_authored(model="glm-5.2")
    def test_dir_is_anchored_not_relative_to_cwd(self, monkeypatch):
        # Anchored to build_hooks.py, so it must agree with the stamp
        # regardless of where the build was invoked from.
        build_dir = _capture_cythonize_build_dir(monkeypatch, "13")

        assert build_dir.is_absolute()
        assert build_dir.parent.parent == build_hooks._BUILD_CONFIG_STAMP.parent


class TestSetuptoolsSourcePaths:
    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_absolute_sources_are_made_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        generated = tmp_path / "build" / "cython" / "cu13-gnu-opt" / "cuda" / "core" / "_device.cpp"
        relative = "cuda/core/_cpp/helper.cpp"
        extension = build_hooks.Extension("cuda.core._device", [str(generated), relative])

        build_hooks._relativize_extension_sources([extension])

        assert extension.sources == [os.path.relpath(generated, start=tmp_path), relative]


def _load_setup_py(monkeypatch):
    """Import setup.py for its command classes.

    Importing rather than running is only possible because setup() is guarded
    by __name__ == "__main__"; setuptools invokes the file as a script, so the
    guard does not affect real builds.

    setup.py does a bare ``import build_hooks``, which resolves to
    cuda_bindings' copy if that directory is on sys.path. Pin cuda_core's, so
    the flag the test sets is the one setup.py reads.
    """
    monkeypatch.setitem(sys.modules, "build_hooks", build_hooks)
    setup_path = Path(__file__).parent.parent / "setup.py"
    spec = importlib.util.spec_from_file_location("cuda_core_setup", setup_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestForceReachesBuildExt:
    """The rebuild decision must actually be handed to setuptools.

    _check_build_config() only sets a flag; if build_ext does not read it, a
    stale extension is silently kept because its mtime looks newer than the
    regenerated sources.
    """

    @staticmethod
    def _finalized_build_ext(force_flag, monkeypatch):
        from setuptools.dist import Distribution

        setup_py = _load_setup_py(monkeypatch)
        assert setup_py.build_hooks is build_hooks
        monkeypatch.setattr(_build_shared, "force_build_ext", force_flag)

        cmd = setup_py.build_ext(Distribution({"name": "cuda-core", "version": "0"}))
        cmd.finalize_options()
        return cmd

    def test_flag_set_forces_rebuild(self, monkeypatch):
        assert self._finalized_build_ext(True, monkeypatch).force

    def test_flag_clear_leaves_default(self, monkeypatch):
        assert not self._finalized_build_ext(False, monkeypatch).force


class TestExtensionSources:
    """_extension_sources: a directory of .cpp files, a single legacy .cpp, or nothing."""

    @pytest.fixture
    def tree(self, tmp_path, monkeypatch):
        core = tmp_path / "cuda" / "core"
        cpp = core / "_cpp"
        (cpp / "a" / "nested").mkdir(parents=True)
        (cpp / "d").mkdir()
        for name in ("_a.pyx", "_b.pyx", "_c.pyx", "_d.pyx"):
            (core / name).write_text("")
        for name in ("a/x.cpp", "a/y.cpp", "a/nested/z.cpp", "a/notes.md", "b.cpp"):
            (cpp / name).write_text("")
        monkeypatch.chdir(tmp_path)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_directory_of_sources(self, tree):
        a = os.path.join("cuda", "core", "_cpp", "a")
        assert build_hooks._extension_sources("_a") == [
            "cuda/core/_a.pyx",
            os.path.join(a, "nested", "z.cpp"),
            os.path.join(a, "x.cpp"),
            os.path.join(a, "y.cpp"),
        ]

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_legacy_single_file_and_no_cpp(self, tree):
        assert build_hooks._extension_sources("_b") == [
            "cuda/core/_b.pyx",
            os.path.join("cuda", "core", "_cpp", "b.cpp"),
        ]
        assert build_hooks._extension_sources("_c") == ["cuda/core/_c.pyx"]

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_empty_directory_is_an_error(self, tree):
        with pytest.raises(RuntimeError, match="no .cpp files"):
            build_hooks._extension_sources("_d")


class TestExtensionDepends:
    """_extension_depends: every header under a directory-form module's
    _cpp/<stem>/, the same list for every extension (see its docstring)."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_headers_under_module_directories_only(self, tmp_path, monkeypatch):
        cpp = tmp_path / "cuda" / "core" / "_cpp"
        (cpp / "a" / "nested").mkdir(parents=True)
        for name in ("a/x.hpp", "a/nested/y.h", "a/z.cpp", "a/notes.md", "top.hpp", "b.cpp"):
            (cpp / name).write_text("")
        monkeypatch.chdir(tmp_path)
        a = os.path.join("cuda", "core", "_cpp", "a")
        assert build_hooks._extension_depends() == [os.path.join(a, "nested", "y.h"), os.path.join(a, "x.hpp")]


class TestParallelSourceCompilation:
    """setup.py compiles an extension's sources through one shared thread pool."""

    class FakeCompiler(CCompiler):
        """Uses the stock CCompiler.compile(), like the Unix compilers."""

        executables = {}

        def __init__(self, fail_on=None):
            super().__init__()
            self.compiled = []
            self.fail_on = fail_on
            self.lock = threading.Lock()

        def _setup_compile(self, outdir, macros, incdirs, sources, depends, extra):
            extra = [] if extra is None else extra  # as distutils does
            objects = [source + ".o" for source in sources]
            return macros, objects, extra, ["-Dpp"], {obj: (src, ".cpp") for obj, src in zip(objects, sources)}

        def _get_cc_args(self, pp_opts, debug, before):
            return ["-c", *pp_opts]

        def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src == self.fail_on:
                raise RuntimeError(f"{src} failed")
            with self.lock:
                self.compiled.append((obj, src, ext, tuple(cc_args), tuple(extra_postargs), tuple(pp_opts)))

    class MsvcLikeCompiler(FakeCompiler):
        """Overrides compile() wholesale, like MSVCCompiler."""

        def compile(self, *args, **kwargs):
            return "stock"

    def _build_ext(self, monkeypatch, nthreads, compiler):
        from setuptools.dist import Distribution

        setup_py = _load_setup_py(monkeypatch)
        monkeypatch.setattr(setup_py, "nthreads", nthreads)
        cmd = setup_py.build_ext(Distribution({"name": "cuda-core", "version": "0"}))
        cmd.compiler = compiler
        return cmd

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_every_source_compiles_once_and_the_object_order_is_kept(self, monkeypatch):
        cmd = self._build_ext(monkeypatch, 4, self.FakeCompiler())
        sources = [f"rt/{name}.cpp" for name in "abcdef"]
        with cmd._parallel_source_compilation():
            objects = cmd.compiler.compile(sources, output_dir="tmp", extra_postargs=["-O2"], depends=["x.hpp"])
        assert objects == [source + ".o" for source in sources]
        assert sorted(entry[0] for entry in cmd.compiler.compiled) == sorted(objects)
        assert {entry[2:] for entry in cmd.compiler.compiled} == {(".cpp", ("-c", "-Dpp"), ("-O2",), ("-Dpp",))}
        assert cmd.compiler.compile.__func__ is CCompiler.compile  # restored on exit

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_a_failing_source_fails_the_extension(self, monkeypatch):
        cmd = self._build_ext(monkeypatch, 4, self.FakeCompiler(fail_on="rt/c.cpp"))
        with cmd._parallel_source_compilation(), pytest.raises(RuntimeError, match="rt/c.cpp failed"):
            cmd.compiler.compile([f"rt/{name}.cpp" for name in "abcdef"])

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_serial_builds_and_compilers_without_the_hook_keep_the_stock_path(self, monkeypatch):
        cmd = self._build_ext(monkeypatch, 1, self.FakeCompiler())
        with cmd._parallel_source_compilation():
            assert cmd.compiler.compile.__func__ is CCompiler.compile
        cmd = self._build_ext(monkeypatch, 4, self.MsvcLikeCompiler())
        with cmd._parallel_source_compilation():
            assert cmd.compiler.compile(["a.cpp"]) == "stock"


def _fake_bindings(monkeypatch, version, cuda_version=None):
    """Make the build see an installed cuda-bindings of `version`, generated from the header
    `cuda_version`. None for `version` means not installed. `cuda_version` defaults to the
    header of the version's major.minor."""

    def installed_cuda_bindings():
        if version is None:
            raise ModuleNotFoundError("No module named 'cuda.bindings'", name="cuda.bindings")
        if cuda_version is not None:
            return version, cuda_version
        major, minor = (int(part) for part in version.split(".")[:2])
        return version, major * 1000 + minor * 10

    monkeypatch.setattr(build_hooks, "_installed_cuda_bindings", installed_cuda_bindings)


def _floor_str(major):
    return ".".join(str(part) for part in build_hooks._bindings_floors()[major])


def _write_cuda_h(tmp_path, cuda_version):
    include = tmp_path / "include"
    include.mkdir(exist_ok=True)
    (include / "cuda.h").write_text(f"#define CUDA_VERSION {cuda_version}\n")
    return str(tmp_path)


class TestBuildConfigurationCheck:
    """_check_build_configuration() accepts exactly one configuration per CUDA
    major: cuda-bindings at or above the floor, and a cuda.h of the same
    major.minor as that cuda-bindings. Anything else is a build error that
    names what the check found and what it requires."""

    FLOOR = build_hooks._bindings_floors()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("major", [12, 13])
    def test_floor_bindings_and_matching_header_pass(self, tmp_path, monkeypatch, major):
        floor = self.FLOOR[major]
        header = floor[0] * 1000 + floor[1] * 10
        _fake_bindings(monkeypatch, f"{floor[0]}.{floor[1]}.{floor[2] + 1}.dev3+gabcdef0")
        cuda_path = _write_cuda_h(tmp_path, header)
        # Returns the header cuda-bindings was generated from, for the versions.hpp cross-check.
        assert build_hooks._check_build_configuration(cuda_path, str(major)) == header

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_bindings_below_the_floor_fail(self, tmp_path, monkeypatch):
        floor = self.FLOOR[13]
        _fake_bindings(monkeypatch, f"{floor[0]}.{floor[1]}.{floor[2] - 1}" if floor[2] else "13.0.0")
        cuda_path = _write_cuda_h(tmp_path, 13040)
        with pytest.raises(RuntimeError, match=r"requires cuda-bindings >= 13\.\d+\.\d+ for CUDA 13"):
            build_hooks._check_build_configuration(cuda_path, "13")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_bindings_of_another_major_fail(self, tmp_path, monkeypatch):
        _fake_bindings(monkeypatch, _floor_str(13))
        cuda_path = _write_cuda_h(tmp_path, 12090)
        with pytest.raises(
            RuntimeError,
            match=f"This cuda.core build is for CUDA 12, but the installed cuda-bindings is {_floor_str(13)}",
        ):
            build_hooks._check_build_configuration(cuda_path, "12")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("header", [13030, 13050, 12090])
    def test_header_minor_must_match_bindings(self, tmp_path, monkeypatch, header):
        floor = self.FLOOR[13]
        _fake_bindings(monkeypatch, _floor_str(13))
        cuda_path = _write_cuda_h(tmp_path, header)
        with pytest.raises(RuntimeError) as excinfo:
            build_hooks._check_build_configuration(cuda_path, "13")
        message = str(excinfo.value)
        assert message.startswith(
            f"cuda.core needs CUDA {floor[0]}.{floor[1]} headers to build with the installed cuda-bindings {_floor_str(13)}, but "
        )
        assert os.path.realpath(os.path.join(cuda_path, "include", "cuda.h")) in message  # the resolved path
        assert f"is CUDA {header // 1000}.{header // 10 % 100}." in message
        assert "This is a build-time requirement only" in message
        assert f"Point CUDA_PATH or CUDA_HOME at a CUDA {floor[0]}.{floor[1]} toolkit" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_header_is_read_even_when_the_major_override_is_set(self, tmp_path, monkeypatch):
        # CUDA_CORE_BUILD_MAJOR skips header detection of the major, not this check.
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        _fake_bindings(monkeypatch, _floor_str(13))
        cuda_path = _write_cuda_h(tmp_path, 13030)
        with pytest.raises(RuntimeError, match=r"needs CUDA 13\.\d+ headers .* is CUDA 13\.3\."):
            build_hooks._check_build_configuration(cuda_path, "13")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_development_bindings_pass_on_their_generated_header(self, tmp_path, monkeypatch):
        """The header rule compares headers, not version strings: a cuda-bindings built from
        main after a toolkit bump still carries the previous release's version string."""
        floor = self.FLOOR[13]
        new_header = floor[0] * 1000 + (floor[1] + 1) * 10
        _fake_bindings(monkeypatch, f"{floor[0]}.{floor[1]}.{floor[2]}.dev5+gabcdef0", cuda_version=new_header)
        cuda_path = _write_cuda_h(tmp_path, new_header)
        assert build_hooks._check_build_configuration(cuda_path, "13") == new_header

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_missing_bindings_is_a_build_error(self, tmp_path, monkeypatch):
        _fake_bindings(monkeypatch, None)  # no cuda-bindings in the build environment
        cuda_path = _write_cuda_h(tmp_path, 13040)
        with pytest.raises(RuntimeError, match="requires cuda-bindings to build"):
            build_hooks._check_build_configuration(cuda_path, "13")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_unparseable_bindings_version_is_a_build_error(self, tmp_path, monkeypatch):
        _fake_bindings(monkeypatch, "0.1.dev1+g0d22cb444")  # a shallow clone of cuda-bindings
        cuda_path = _write_cuda_h(tmp_path, 13040)
        with pytest.raises(RuntimeError, match="Cannot parse the installed cuda-bindings version"):
            build_hooks._check_build_configuration(cuda_path, "13")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_unsupported_major_is_a_build_error(self, tmp_path, monkeypatch):
        _fake_bindings(monkeypatch, "14.0.0")
        cuda_path = _write_cuda_h(tmp_path, 14000)
        with pytest.raises(RuntimeError, match="does not support CUDA 14"):
            build_hooks._check_build_configuration(cuda_path, "14")


class TestBindingsFloorsFromPyproject:
    """_bindings_floors() reads the cu<major> extras of pyproject.toml. A malformed extra fails the build."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_malformed_extra_is_a_build_error(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "cuda-core"\n[project.optional-dependencies]\ncu13 = ["cuda-bindings>=13.4"]\n'
        )
        monkeypatch.setattr(build_hooks, "_PYPROJECT_PATH", pyproject)
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        build_hooks._bindings_floors.cache_clear()
        build_hooks._determine_cuda_major_version.cache_clear()
        try:
            for call in (
                lambda: build_hooks._check_build_configuration(str(tmp_path), "13"),
                build_hooks._get_cuda_bindings_require,
                lambda: build_hooks._build_define_macros("13"),
            ):
                with pytest.raises(RuntimeError, match=r"pyproject\.toml: the 'cu13' extra must pin cuda-bindings"):
                    call()
        finally:
            build_hooks._bindings_floors.cache_clear()
            build_hooks._determine_cuda_major_version.cache_clear()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_checkout_declares_both_majors(self):
        floors = build_hooks._bindings_floors()
        assert list(floors) == [12, 13]
        assert all(floor[0] == major for major, floor in floors.items())


class TestBuildRequirement:
    """get_requires_for_build_wheel pins cuda-bindings for isolated builds. It pins the floor.
    When cuda.h is readable, it also pins the header's minor, so pip cannot pick a newer minor
    than the toolkit."""

    @staticmethod
    def _no_cuda_path():
        raise RuntimeError("no CUDA")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("major", ["12", "13"])
    def test_pins_the_floor_and_the_major_without_a_header(self, monkeypatch, major):
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", major)
        monkeypatch.setattr(build_hooks, "_get_cuda_path", self._no_cuda_path)
        build_hooks._determine_cuda_major_version.cache_clear()
        floor = build_hooks._bindings_floors()[int(major)]
        (requirement,) = build_hooks._get_cuda_bindings_require()
        assert requirement == f"cuda-bindings>={floor[0]}.{floor[1]}.{floor[2]},<{int(major) + 1}"

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_caps_at_the_headers_minor_when_cuda_h_is_readable(self, tmp_path, monkeypatch):
        from packaging.specifiers import SpecifierSet

        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        build_hooks._determine_cuda_major_version.cache_clear()
        floor = build_hooks._bindings_floors()[13]
        cuda_path = _write_cuda_h(tmp_path, floor[0] * 1000 + (floor[1] + 1) * 10)
        monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: cuda_path)
        (requirement,) = build_hooks._get_cuda_bindings_require()
        assert requirement == f"cuda-bindings>={floor[0]}.{floor[1]}.{floor[2]},<13.{floor[1] + 2}"
        specifiers = SpecifierSet(requirement.removeprefix("cuda-bindings"))
        # The dev cuda-bindings built alongside a toolkit bump still carries the old minor's version string.
        assert specifiers.contains(f"13.{floor[1]}.{floor[2] + 1}.dev133", prereleases=True)
        assert specifiers.contains(f"13.{floor[1] + 1}.0")
        assert not specifiers.contains(f"13.{floor[1] + 2}.0")  # a newer minor on PyPI stays out

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_requests_the_floor_alone_when_the_header_is_of_another_major(self, tmp_path, monkeypatch):
        # CUDA_CORE_BUILD_MAJOR=13 with a CUDA 12 toolkit: the configuration check reports the mismatch.
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        build_hooks._determine_cuda_major_version.cache_clear()
        floor = build_hooks._bindings_floors()[13]
        cuda_path = _write_cuda_h(tmp_path, 12090)
        monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: cuda_path)
        (requirement,) = build_hooks._get_cuda_bindings_require()
        assert requirement == f"cuda-bindings>={floor[0]}.{floor[1]}.{floor[2]},<14"

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_requests_the_floor_alone_when_the_header_is_below_it(self, tmp_path, monkeypatch):
        # The configuration check reports this mismatch in its own words.
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        build_hooks._determine_cuda_major_version.cache_clear()
        floor = build_hooks._bindings_floors()[13]
        cuda_path = _write_cuda_h(tmp_path, floor[0] * 1000 + (floor[1] - 1) * 10)
        monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: cuda_path)
        (requirement,) = build_hooks._get_cuda_bindings_require()
        assert requirement == f"cuda-bindings>={floor[0]}.{floor[1]}.{floor[2]},<14"

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_unsupported_major_names_the_supported_ones(self, monkeypatch):
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "11")
        build_hooks._determine_cuda_major_version.cache_clear()
        with pytest.raises(RuntimeError, match="does not support CUDA 11.*12, 13"):
            build_hooks._get_cuda_bindings_require()


class TestDefineMacros:
    """The C++ learns the build decision through three macros. See _cpp/rt/versions.hpp."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("major", ["12", "13"])
    def test_major_and_floor_header_version(self, major):
        floor = build_hooks._bindings_floors()[int(major)]
        assert build_hooks._build_define_macros(major) == [
            ("CUDA_CORE_BUILD_MAJOR", major),
            ("CUDA_CORE_MIN_CUDA_VERSION", str(floor[0] * 1000 + floor[1] * 10)),
        ]

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_bindings_header_is_a_third_macro(self):
        # versions.hpp compares the compiler's cuda.h with it by major.minor.
        assert build_hooks._build_define_macros("13", 13040)[2] == ("CUDA_CORE_BINDINGS_CUDA_VERSION", "13040")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_floor_reaches_the_cython_compile_time_environment(self, monkeypatch):
        # cuda/core/_build_info.pyx records it next to the header the compiler resolved.
        captured = _capture_cythonize_kwargs(monkeypatch, "13")
        assert captured["compile_time_env"] == {
            "CUDA_CORE_BUILD_MAJOR": 13,
            "CUDA_CORE_BINDINGS_FLOOR": build_hooks._bindings_floors()[13],
        }

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_extensions_receive_the_macros(self, monkeypatch):
        captured = {}

        def fake_cythonize(ext_modules, **kwargs):
            captured["macros"] = {tuple(ext.define_macros) for ext in ext_modules}
            return []

        monkeypatch.setattr(build_hooks, "_get_cuda_path", lambda: "/nonexistent-cuda")
        monkeypatch.setattr(build_hooks, "_check_build_configuration", lambda *_: None)
        monkeypatch.setattr(build_hooks, "cythonize", fake_cythonize)
        monkeypatch.setenv("CUDA_CORE_BUILD_MAJOR", "13")
        build_hooks._determine_cuda_major_version.cache_clear()
        monkeypatch.chdir(Path(__file__).parent.parent)
        monkeypatch.setattr(sys, "path", list(sys.path))
        build_hooks._build_cuda_core()
        assert captured["macros"] == {tuple(build_hooks._build_define_macros("13"))}


# ---------------------------------------------------------------------------
# Cython cache path helper (workaround for cython/cython#7532)
#
# These tests cover the configuration-digest workaround in build_hooks.py.
# They can be deleted together with the `_cython_cache_path` helper once
# cython/cython#7532 is resolved in a released Cython version and
# cuda-python's minimum Cython version includes the fix.
# See https://github.com/cython/cython/issues/7532


_test_helpers_root = Path(__file__).parents[2] / "cuda_python_test_helpers"
if _test_helpers_root.is_dir() and str(_test_helpers_root) not in sys.path:
    sys.path.insert(0, str(_test_helpers_root))

from cuda_python_test_helpers.build_shared import (
    AbiStampPathMixin,
    BuildKeyStampMixin,
    CheckToolchainAvailableMixin,
    DistutilsLinkerIntegrationMixin,
    ForceBuildExtReexportMixin,
    ResolveToolchainMixin,
    WithCompilerMixin,
    WithSccacheMixin,
)
from cuda_python_test_helpers.cython_cache import POSIX_ONLY_CACHE, CythonAliasMixin, CythonCachePathMixin


class TestResolveToolchainShared(ResolveToolchainMixin):
    build_shared = _build_shared


class TestWithSccache(WithSccacheMixin):
    build_shared = _build_shared


class TestWithCompiler(WithCompilerMixin):
    build_shared = _build_shared


class TestDistutilsLinkerIntegration(DistutilsLinkerIntegrationMixin):
    build_shared = _build_shared


class TestCheckToolchainAvailable(CheckToolchainAvailableMixin):
    build_shared = _build_shared


class TestAbiStampPath(AbiStampPathMixin):
    build_shared = _build_shared


class TestBuildKeyStamp(BuildKeyStampMixin):
    build_shared = _build_shared


class TestForceBuildExtReexport(ForceBuildExtReexportMixin):
    build_hooks = build_hooks
    build_shared = _build_shared


class TestResolveToolchain:
    """What cuda.core chooses in its ``_resolve_toolchain`` wrapper."""

    @pytest.mark.agent_authored(model="claude-sonnet-4-6")
    def test_linux_flag_set(self, monkeypatch):
        """c++17 (structured bindings and if constexpr in cuda/core/_cpp/); warnings are not errors by default."""
        if sys.platform == "win32":
            pytest.skip("Linux flags only")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.setattr(build_hooks, "WARNINGS_AS_ERRORS", False)
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain(debug=False)
        assert "-std=c++17" in cargs
        assert "-Werror" not in cargs

    @pytest.mark.agent_authored(model="claude-sonnet-4-6")
    def test_msvc_flag_set(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("MSVC flags only on Windows")
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.setattr(build_hooks, "WARNINGS_AS_ERRORS", False)
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain(debug=False)
        assert "/std:c++17" in cargs
        assert "/WX" not in cargs

    @pytest.mark.agent_authored(model="claude-sonnet-5.5")
    def test_cuda_python_werror_makes_warnings_errors(self, monkeypatch):
        """CUDA_PYTHON_WERROR=1 (read into WARNINGS_AS_ERRORS) reaches the shared flag set."""
        monkeypatch.delenv("CUDA_PYTHON_TOOLCHAIN", raising=False)
        monkeypatch.setattr(build_hooks, "WARNINGS_AS_ERRORS", True)
        _name, _cc, _cxx, cargs, _largs = build_hooks._resolve_toolchain(debug=False)
        assert ("/WX" if sys.platform == "win32" else "-Werror") in cargs


class TestCudaCoreCythonIncludePath:
    """How `_build_cuda_core` assembles `include_path` for cythonize().

    `_stable_cython_alias` itself (creation, cleanup, cross-env cache hits) is
    covered generically by `TestCythonAlias` below. These tests cover only
    `_build_cuda_core`'s own wiring: which aliases it adds, in what order, and
    the cache-disabled / bindings-unavailable fallbacks.
    """

    @staticmethod
    def _fake_bindings(monkeypatch):
        bindings = types.ModuleType("cuda.bindings")
        bindings.__file__ = "/random-build-env/cuda/bindings/__init__.py"
        bindings.__version__ = "13.0"
        monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
        monkeypatch.setattr(sys.modules["cuda"], "bindings", bindings, raising=False)

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_cache_enabled_includes_bindings_and_stdlib_aliases(self, monkeypatch, tmp_path):
        self._fake_bindings(monkeypatch)
        monkeypatch.setenv("CUDA_PYTHON_CYTHON_CACHE_DIR", str(tmp_path))

        captured = _capture_cythonize_kwargs(monkeypatch, "13")

        assert captured["include_path"] == [".", ".cython-bindings", ".cython-stdlib"]

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_cache_enabled_without_bindings_omits_bindings_alias(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "cuda.bindings", None)  # forces ImportError
        monkeypatch.setenv("CUDA_PYTHON_CYTHON_CACHE_DIR", str(tmp_path))

        captured = _capture_cythonize_kwargs(monkeypatch, "13")

        assert captured["include_path"] == [".", ".cython-stdlib"]

    @pytest.mark.agent_authored(model="grok-4.6")
    def test_cache_disabled_skips_aliasing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cuda.bindings", None)  # forces ImportError
        monkeypatch.delenv("CUDA_PYTHON_CYTHON_CACHE_DIR", raising=False)

        captured = _capture_cythonize_kwargs(monkeypatch, "13")

        assert captured["include_path"] == ["."]


class TestCythonCachePath(CythonCachePathMixin):
    """`_cython_cache_path` tests specific to cuda.core.

    Inherits the common tests from CythonCachePathMixin; the mixin
    covers the package-agnostic behavior. cuda.core passes ``compile_time_env``
    and ``cuda_major``, so those partitions are tested here.
    """

    build_hooks = build_hooks
    package = "cuda-core"

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_changed_compile_time_env_changes_namespace(self, monkeypatch, tmp_path):
        """Different ``compile_time_env`` values map to different namespaces."""
        self._set_env(monkeypatch, str(tmp_path))
        p1 = build_hooks._cython_cache_path("cuda-core", compile_time_env={"CUDA_CORE_BUILD_MAJOR": 12})
        p2 = build_hooks._cython_cache_path("cuda-core", compile_time_env={"CUDA_CORE_BUILD_MAJOR": 13})
        assert p1 != p2

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_changed_debug_or_cuda_major_changes_namespace(self, monkeypatch, tmp_path):
        """``debug`` and ``cuda_major`` each partition the namespace."""
        self._set_env(monkeypatch, str(tmp_path))
        p1 = build_hooks._cython_cache_path("cuda-core", debug=False, cuda_major="12")
        p2 = build_hooks._cython_cache_path("cuda-core", debug=True, cuda_major="12")
        p3 = build_hooks._cython_cache_path("cuda-core", debug=False, cuda_major="13")
        assert p1 != p2
        assert p1 != p3
        assert p2 != p3


class TestCythonCacheSmokeTest:
    """Real Cython cache miss/hit through `_cython_cache_path`.

    The actual cythonize exercise lives in
    ``cuda_python_test_helpers.cython_cache`` so it is shared with
    ``cuda_bindings/tests/test_build_hooks.py``.
    """

    @POSIX_ONLY_CACHE
    @pytest.mark.agent_authored(model="grok-4.6")
    def test_cache_miss_then_hit(self, monkeypatch, tmp_path, capsys):
        from cuda_python_test_helpers.cython_cache import (
            cython_cache_miss_then_hit,
        )

        cache_root = tmp_path / "cython-cache"
        cache_root.mkdir()
        monkeypatch.setenv("CUDA_PYTHON_CYTHON_CACHE_DIR", str(cache_root))

        cache_path = build_hooks._cython_cache_path(
            "cuda-core",
            compiler_directives={"language_level": 3},
            language_level=3,
            cplus=False,
        )
        assert cache_path is not None
        cython_cache_miss_then_hit(cache_path, tmp_path, capsys)


class TestCythonAlias(CythonAliasMixin):
    """`_stable_cython_alias` tests for cuda.core."""

    build_hooks = build_hooks
