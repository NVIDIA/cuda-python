# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.

import functools
import glob
import importlib.util
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options as _CythonOptions
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist

COMPILE_FOR_COVERAGE = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


# Please keep in sync with the copy in cuda_bindings/build_hooks.py.
def _import_get_cuda_path_or_home():
    """Import get_cuda_path_or_home, working around PEP 517 namespace shadowing.

    See https://github.com/NVIDIA/cuda-python/issues/1824 for why this helper is needed.
    """
    try:
        import cuda.pathfinder
    except ModuleNotFoundError as exc:
        if exc.name not in ("cuda", "cuda.pathfinder"):
            raise
        try:
            import cuda
        except ModuleNotFoundError:
            cuda = None

        for p in sys.path:
            sp_cuda = Path(p) / "cuda"
            if (sp_cuda / "pathfinder").is_dir():
                cuda.__path__ = list(cuda.__path__) + [str(sp_cuda)]
                break
        else:
            raise ModuleNotFoundError(
                "cuda-pathfinder is not installed in the build environment. "
                "Ensure 'cuda-pathfinder>=1.5' is in build-system.requires."
            )
        import cuda.pathfinder

    pathfinder_dir = Path(cuda.pathfinder.__file__).parent
    print(
        f"Using cuda-pathfinder {cuda.pathfinder.__version__} from {pathfinder_dir}",
        file=sys.stderr,
    )
    return cuda.pathfinder.get_cuda_path_or_home


@functools.cache
def _get_cuda_path() -> str:
    get_cuda_path_or_home = _import_get_cuda_path_or_home()
    cuda_path = get_cuda_path_or_home()
    if not cuda_path:
        raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
    print("CUDA path:", cuda_path)
    return cuda_path


_PACKAGE_DIR = Path(__file__).parent / "cuda" / "core"

# Generated at build time by _write_build_info(); read by cuda/core/__init__.py.
_BUILD_INFO_PATH = _PACKAGE_DIR / "_build_info.py"


@functools.cache
def _load_bindings_floor():
    """Load cuda/core/_bindings_floor.py, the floor's single source of truth.

    Loaded by file path: the package this backend builds is not importable
    during its own build, and the module is deliberately import-free.
    """
    path = _PACKAGE_DIR / "_bindings_floor.py"
    spec = importlib.util.spec_from_file_location("_cuda_core_bindings_floor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_cuda_h_version(cuda_path: str) -> int:
    """The CUDA_VERSION macro (e.g. 13040 for 13.4) of the cuda.h under cuda_path."""
    cuda_h = os.path.join(cuda_path, "include", "cuda.h")
    try:
        with open(cuda_h, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    raise RuntimeError(
        f"Cannot read CUDA_VERSION from {cuda_h}. "
        "Ensure CUDA_PATH or CUDA_HOME points to a valid CUDA installation with include/cuda.h."
    )


@functools.cache
def _determine_cuda_major_version() -> str:
    """Determine the CUDA major version for building cuda.core.

    This version is used for two purposes:
    1. Determining which cuda-bindings version to install as a build dependency
    2. Setting CUDA_CORE_BUILD_MAJOR for Cython compile-time conditionals

    The version is derived from (in order of priority):
    1. CUDA_CORE_BUILD_MAJOR environment variable (explicit override, e.g. in CI)
    2. CUDA_VERSION macro in cuda.h from CUDA_PATH or CUDA_HOME

    Since CUDA_PATH or CUDA_HOME is required for the build (to provide include
    directories), the cuda.h header should always be available. The override
    only short-circuits this detection; _check_build_configuration() still
    reads the header and rejects one whose major disagrees.
    """
    # Explicit override, e.g. in CI.
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is not None:
        print("CUDA MAJOR VERSION:", cuda_major)
        return cuda_major

    # Derive from the CUDA headers (the authoritative source for what we compile against).
    try:
        cuda_version = _read_cuda_h_version(_get_cuda_path())
    except RuntimeError as exc:
        # CUDA_PATH or CUDA_HOME is required for the build, so we should not reach
        # here in normal circumstances. Raise an error to make the issue clear.
        raise RuntimeError(
            "Cannot determine CUDA major version. "
            "Set CUDA_CORE_BUILD_MAJOR environment variable, or ensure CUDA_PATH or CUDA_HOME "
            "points to a valid CUDA installation with include/cuda.h."
        ) from exc
    # CUDA_VERSION is e.g. 12020 for 12.2.
    cuda_major = str(cuda_version // 1000)
    print("CUDA MAJOR VERSION:", cuda_major)
    return cuda_major


def _check_build_configuration(cuda_path: str, cuda_major: str) -> None:
    """Reject build configurations cuda.core does not support, then record the build.

    cuda.core supports one configuration per CUDA major series: the installed
    cuda-bindings is at least the series' floor (cuda/core/_bindings_floor.py)
    and the cuda.h it compiles against has the same major.minor as that
    cuda-bindings, which is the header cuda-bindings itself was generated from.
    The pip build requirement (get_requires_for_build_*) states the floor, but
    conda-forge, pixi and --no-build-isolation installs bypass it, so the check
    lives here, where every build path passes.

    A too-old cuda-bindings used to surface late as an ImportError at module
    init or as a feature that was silently compiled out; a mismatched header as
    an unclear Cython error (see https://github.com/NVIDIA/cuda-python/issues/2783).
    """
    floor = _load_bindings_floor()
    major = int(cuda_major)
    if major not in floor.CUDA_BINDINGS_FLOOR:
        raise RuntimeError(
            f"cuda.core does not support CUDA {major}; supported CUDA major versions: "
            f"{', '.join(str(m) for m in floor.SUPPORTED_CUDA_MAJORS)}"
        )
    requirement = floor.pip_requirement(major)

    try:
        bindings_module = importlib.import_module("cuda.bindings")
    except ImportError as exc:
        raise RuntimeError(
            f"cuda.core requires cuda-bindings to build (install '{requirement}'). "
            "Isolated builds install it automatically; other builds must provide it."
        ) from exc
    bindings_version = bindings_module.__version__
    bindings = floor.release_triple(bindings_version)
    if bindings is None:
        raise RuntimeError(
            f"Cannot parse the installed cuda-bindings version {bindings_version!r}. "
            "A shallow git clone of cuda-bindings reports a bogus version; see CONTRIBUTING.md."
        )
    if bindings[0] != major:
        raise RuntimeError(
            f"Building cuda.core for CUDA {major}, but the installed cuda-bindings is "
            f"{bindings_version}. Install '{requirement}'."
        )
    if bindings < floor.CUDA_BINDINGS_FLOOR[major]:
        raise RuntimeError(
            f"cuda.core requires cuda-bindings >= {floor.format_version(floor.CUDA_BINDINGS_FLOOR[major])} "
            f"for CUDA {major}, but {bindings_version} is installed. Install '{requirement}'."
        )

    cuda_version = _read_cuda_h_version(cuda_path)
    header = (cuda_version // 1000, cuda_version // 10 % 100)
    if header != bindings[:2]:
        raise RuntimeError(
            f"cuda.h under {cuda_path} is CUDA {header[0]}.{header[1]}, but the installed cuda-bindings "
            f"is {bindings_version}. cuda.core must be built against a cuda.h of the same "
            "major.minor as its cuda-bindings (the header cuda-bindings was generated from). "
            "Point CUDA_PATH or CUDA_HOME at a matching CUDA Toolkit, or install matching cuda-bindings."
        )
    print(f"Build configuration: CUDA {header[0]}.{header[1]} headers, cuda-bindings {bindings_version}")
    _write_build_info(major, cuda_version, floor.CUDA_BINDINGS_FLOOR[major], bindings_version)


def _write_build_info(cuda_major: int, cuda_version: int, floor: tuple, bindings_version: str) -> None:
    """Record what this build compiled against, for the import-time check.

    cuda/core/__init__.py reads this module before it selects the versioned
    subpackage and refuses an installed cuda-bindings older than the floor or
    older, by minor, than the header (see _bindings_floor.required_minimum).
    Like _version.py, the file is generated, gitignored, and shipped.
    """
    _BUILD_INFO_PATH.write_text(
        "# Generated by build_hooks.py at build time. Do not edit or commit.\n"
        f"CUDA_MAJOR = {cuda_major}\n"
        f"CUDA_VERSION = {cuda_version}  # the cuda.h this build compiled against\n"
        f"CUDA_BINDINGS_FLOOR = {tuple(floor)!r}\n"
        f"CUDA_BINDINGS_BUILD_VERSION = {bindings_version!r}\n",
        encoding="utf-8",
    )


# used later by setup()
_extensions = None

# Where per-configuration build artifacts live. Anchored to this file rather
# than the cwd, since a project can be built from anywhere.
_BUILD_DIR = Path(__file__).parent / "build"

# Records the CUDA major of the last completed build, so setup.py can force
# build_ext when it changes. Written by record_build_major().
_BUILD_MAJOR_STAMP = _BUILD_DIR / ".build-cuda-major"

force_build_ext = False


def _check_build_major() -> str:
    """Return the CUDA major to key build artifacts by, and set force_build_ext.

    Cython's up-to-date check does not hash ``compile_time_env``, so generated
    sources for one CUDA major would otherwise be reused for another. Keying
    the generated-source directory fixes that, but not the compiled extension:
    in an editable install it lands in the source tree under a name keyed by
    the Python ABI tag alone, with nowhere to record the CUDA major. On a
    cu12 -> cu13 -> cu12 round trip build_ext would find the older cu12
    generated source next to the newer cu13 .so and skip the rebuild, so the
    major is also stamped and build_ext forced whenever it changes.
    """
    global force_build_ext

    cuda_major = _determine_cuda_major_version()
    try:
        previous = _BUILD_MAJOR_STAMP.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None

    # A missing stamp means the last build's major is unknown, so force too.
    # On a first build that costs nothing: there are no artifacts to reuse.
    if previous != cuda_major:
        print(f"CUDA major of last build: {previous} (building {cuda_major}); forcing a full rebuild")
        force_build_ext = True

    return cuda_major


def record_build_major() -> None:
    """Stamp the CUDA major of the build that just completed.

    setup.py calls this after build_ext succeeds, so that a build which failed
    partway through does not claim outputs it never produced.
    """
    _BUILD_MAJOR_STAMP.parent.mkdir(parents=True, exist_ok=True)
    _BUILD_MAJOR_STAMP.write_text(_determine_cuda_major_version() + "\n", encoding="utf-8")


def _relativize_extension_sources(extensions) -> None:
    """Keep absolute source paths out of setuptools' temporary build tree."""
    for extension in extensions:
        extension.sources = [
            os.path.relpath(source, start=Path.cwd()) if os.path.isabs(source) else source
            for source in extension.sources
        ]


def _extension_sources(mod_name):
    """The module's .pyx plus its C++, if any: every .cpp under
    cuda/core/_cpp/<stem>/, or the single legacy file cuda/core/_cpp/<stem>.cpp.
    Example: _tensor_map.pyx compiles _cpp/tensor_map.cpp."""
    sources = [f"cuda/core/{mod_name}.pyx"]
    cpp_stem = Path("cuda", "core", "_cpp", mod_name.lstrip("_"))
    if cpp_stem.is_dir():
        cpp_sources = sorted(str(path) for path in cpp_stem.rglob("*.cpp"))
        if not cpp_sources:
            raise RuntimeError(f"{cpp_stem}/ exists but contains no .cpp files")
        sources.extend(cpp_sources)
    elif cpp_stem.with_suffix(".cpp").is_file():
        sources.append(str(cpp_stem.with_suffix(".cpp")))
    return sources


def _extension_depends():
    """Headers whose edits must rebuild an extension: every header under a
    directory-form module's cuda/core/_cpp/<stem>/ (a single-file module has
    none).

    The same list serves every extension. A module that cimports a
    directory-form module compiles against the header its .pxd names, and
    cythonize copies each `depends` entry into its build directory before
    compiling, so the copied header finds its sibling includes beside it
    (quoted includes resolve next to the copy, not in the source tree).
    Listing the whole directory keeps the rule free of include parsing; the
    cost is that every extension rebuilds when any of these headers changes,
    exactly as editing the one monolithic header did before the split."""
    cpp = Path("cuda", "core", "_cpp")
    return sorted(
        str(path)
        for module_dir in cpp.iterdir()
        if module_dir.is_dir()
        for path in module_dir.rglob("*")
        if path.suffix in (".h", ".hpp")
    )


def _build_cuda_core(debug=False):
    # Customizing the build hooks is needed because we must defer cythonization until cuda-bindings,
    # now a required build-time dependency that's dynamically installed via the other hook below,
    # is installed. Otherwise, cimport any cuda.bindings modules would fail!
    #
    # This function populates "_extensions".
    global _extensions

    # Resolve CUDA first so the pathfinder import repairs PEP 517 namespace shadowing before importing bindings.
    cuda_path = _get_cuda_path()

    # Add cuda-bindings to sys.path so Cython can find .pxd files
    # This is needed for editable installs where meta path finders don't work for Cython
    # We need to add the directory containing the 'cuda' package so Cython can resolve
    # "from cuda.bindings cimport cydriver"
    try:
        import cuda.bindings

        bindings_path = Path(cuda.bindings.__file__).parent  # .../cuda/bindings/
        print(f"Using cuda-bindings {cuda.bindings.__version__} from {bindings_path}", file=sys.stderr)
        cuda_package_dir = bindings_path.parent.parent  # .../cuda_bindings/ (contains cuda/)
        if str(cuda_package_dir) not in sys.path:
            sys.path.insert(0, str(cuda_package_dir))
            print(f"Added cuda-bindings parent path for Cython: {cuda_package_dir}", file=sys.stderr)
    except ImportError:
        # cuda-bindings not available in editable mode, will use installed version
        pass

    _posix_only_modules = frozenset(
        {
            "_utils/_wsl_locale",
        }
    )

    # It seems setuptools' wildcard support has problems for namespace packages,
    # so we explicitly spell out all Extension instances.
    def module_names():
        root_path = os.path.sep.join(["cuda", "core", ""])
        for filename in glob.glob(f"{root_path}/**/*.pyx", recursive=True):
            mod = filename[len(root_path) : -4]
            if sys.platform == "win32" and mod.replace(os.path.sep, "/") in _posix_only_modules:
                continue
            yield mod

    all_include_dirs = [os.path.join(cuda_path, "include")]
    extra_compile_args = []
    extra_link_args = []
    extra_cythonize_kwargs = {}
    if sys.platform == "win32":
        extra_compile_args += ["/std:c++17"]
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    else:
        extra_compile_args += ["-std=c++17"]
        if debug:
            extra_cythonize_kwargs["gdb_debug"] = True
            extra_compile_args += ["-g", "-O0"]
            extra_compile_args += ["-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O2"]
            extra_link_args += ["-Wl,--strip-all"]
    if COMPILE_FOR_COVERAGE:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    depends = _extension_depends()
    ext_modules = tuple(
        Extension(
            f"cuda.core.{mod.replace(os.path.sep, '.')}",
            sources=_extension_sources(mod),
            depends=depends,
            include_dirs=[
                "cuda/core/_include",
                "cuda/core/_cpp",
            ]
            + all_include_dirs,
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        for mod in module_names()
    )

    # Deliberately after the cuda.bindings import above: this re-enters
    # _get_cuda_path() and reads cuda.h, which must not run before the
    # pathfinder import has repaired PEP 517 namespace shadowing.
    cuda_major = _check_build_major()
    _check_build_configuration(cuda_path, cuda_major)

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": int(cuda_major)}
    compiler_directives = {"embedsignature": True, "warn.deprecated.IF": False, "freethreading_compatible": True}
    _CythonOptions.warning_errors = True
    if COMPILE_FOR_COVERAGE:
        compiler_directives["linetrace"] = True
    _extensions = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        # CUDA_PYTHON_COVERAGE deliberately generates in-tree so the sources can
        # be packaged; every other build gets its own per-configuration cache,
        # anchored alongside the stamp so both resolve the same from any cwd.
        # Cython also copies each extension's extern headers and `depends` under
        # this directory and compiles against the copies. Copies are refreshed by
        # mtime and never deleted, so remove build/ after renaming or deleting a
        # header under _cpp/.
        build_dir="." if COMPILE_FOR_COVERAGE else str(_BUILD_DIR / "cython" / f"cu{cuda_major}"),
        nthreads=nthreads,
        compiler_directives=compiler_directives,
        compile_time_env=compile_time_env,
        **extra_cythonize_kwargs,
    )
    # Cython returns generated sources under the absolute build_dir above.
    # setuptools mirrors absolute source paths into build/temp, which can push
    # MSVC linker output paths past MAX_PATH in deeper Windows checkouts.
    _relativize_extension_sources(_extensions)

    return


def _add_cython_include_paths_to_pth(wheel_path: str) -> None:
    """
    Modify the .pth file in an editable install wheel to add Cython include paths.

    This is needed because Cython cannot find .pxd files through meta path finders,
    it only looks in sys.path directories. By adding direct paths to the .pth file,
    we enable Cython to find .pxd files from editable-installed cuda-bindings.

    See: https://github.com/scikit-build/scikit-build-core/pull/516
    See: https://github.com/cython/cython/issues/7326
    """
    # Find cuda-bindings location
    # When building with pixi path dependencies, cuda-bindings should be importable
    try:
        import cuda.bindings

        bindings_path = Path(cuda.bindings.__file__).parent  # .../cuda/bindings/
        # We need the directory containing the 'cuda' package for Cython imports
        cuda_package_dir = bindings_path.parent.parent  # .../cuda_bindings/ (contains cuda/)
        print(f"Found cuda-bindings at: {bindings_path}", file=sys.stderr)
        print(f"Will add to .pth for Cython: {cuda_package_dir}", file=sys.stderr)
    except ImportError:
        # If cuda-bindings isn't available yet, we can't add the path
        # This might happen in some build scenarios, but it's okay - the
        # wildcard dependency will work in those cases
        print("cuda-bindings not found in current environment, skipping .pth modification")
        return

    # Create a temporary directory for wheel manipulation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wheel_file = Path(wheel_path)

        # Extract the wheel
        extract_dir = tmpdir_path / "extracted"
        with zipfile.ZipFile(wheel_file, "r") as zf:
            zf.extractall(extract_dir)

        # Find the .pth file (should be named something like __editable___cuda_core-*.pth)
        pth_files = list(extract_dir.glob("**/*.pth"))
        if not pth_files:
            print("Warning: No .pth file found in editable wheel", file=sys.stderr)
            return

        # Modify each .pth file (usually just one)
        for pth_file in pth_files:
            print(f"Modifying {pth_file.name} to add Cython include paths", file=sys.stderr)

            # Read existing content
            content = pth_file.read_text()

            # Add the cuda-bindings source path to sys.path for Cython
            # This allows Cython to find .pxd files via direct path lookup
            # The path must be the directory containing the 'cuda' package
            path_to_add = str(cuda_package_dir.absolute())

            # Ensure content ends with newline before adding path
            if not content.endswith("\n"):
                content += "\n"

            # Append to the .pth file (after the import hook line)
            if path_to_add not in content:
                pth_file.write_text(content + path_to_add + "\n")
                print(f"Added Cython include path: {cuda_package_dir}", file=sys.stderr)

        # Repackage the wheel
        # Remove the old wheel first
        wheel_file.unlink()

        # Create new wheel with same name
        with zipfile.ZipFile(wheel_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zf.write(file_path, arcname)

        print(f"Successfully patched {wheel_file.name}", file=sys.stderr)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    debug_default = sys.platform != "win32"  # Debug builds not supported on Windows
    debug = config_settings.get("debug", debug_default) if config_settings else debug_default
    _build_cuda_core(debug=debug)
    wheel_name = _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)

    # Patch the .pth file to add Cython include paths
    wheel_path = os.path.join(wheel_directory, wheel_name)
    _add_cython_include_paths_to_pth(wheel_path)

    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    debug = config_settings.get("debug", False) if config_settings else False
    _build_cuda_core(debug=debug)
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def _get_cuda_bindings_require():
    """The cuda-bindings build requirement: the floor of the CUDA major being built.

    Honored by isolated builds only; _check_build_configuration() enforces the
    same rule for every other build path.
    """
    floor = _load_bindings_floor()
    cuda_major = int(_determine_cuda_major_version())
    if cuda_major not in floor.CUDA_BINDINGS_FLOOR:
        raise RuntimeError(
            f"cuda.core does not support CUDA {cuda_major}; supported CUDA major versions: "
            f"{', '.join(str(m) for m in floor.SUPPORTED_CUDA_MAJORS)}"
        )
    return [floor.pip_requirement(cuda_major)]


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
