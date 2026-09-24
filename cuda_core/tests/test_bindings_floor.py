# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cuda-bindings version floor in cuda/core/_bindings_floor.py.

The tests cover the floor as read from the pyproject extras and the import-time check
built on it. They also cover the consistency hook that guards ci/versions.yml and the docs.

The tests check source-tree properties and pure functions and need no GPU. The file
also runs with --noconftest, which skips the CUDA setup in conftest.py. The consistency
tests read pyproject.toml and ci/versions.yml from the checkout. They need the source
tree next to the tests. Every CI job that runs tests/ has it.

    pytest tests/test_bindings_floor.py -v --noconftest
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cuda.core import _bindings_floor as floor_mod
from cuda.core._bindings_floor import (
    bindings_requirement,
    check_installed_bindings,
    cuda_version_of,
    floors_from_extras,
    format_version,
    header_minor,
    release_triple,
)

CUDA_CORE = Path(__file__).resolve().parent.parent
REPO = CUDA_CORE.parent
HOOK = REPO / "toolshed" / "check_cuda_core_bindings_floor.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    """toolshed/check_cuda_core_bindings_floor.py lives outside cuda_core/, so an sdist tree lacks it."""
    if not HOOK.is_file():
        pytest.skip(f"{HOOK} is not in this tree. The hook tests need the monorepo checkout")
    return _load("check_cuda_core_bindings_floor", HOOK)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floor_module_is_import_free():
    """build_hooks.py, conf.py and the hook load it by file path, so it must stay standard-library only."""
    source = Path(floor_mod.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+(\w+)", source, re.M)
    assert set(imports) <= {"__future__", "collections", "re"}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_come_from_the_pyproject_extras(hook):
    """The extras are the single source. read_floors() returns one release triple per major."""
    floors = hook.read_floors(REPO)
    assert sorted(floors) == [12, 13]
    for major, floor in floors.items():
        assert floor[0] == major
        assert len(floor) == 3


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_from_extras_accepts_the_declared_form():
    extras = {
        "cu12": ["cuda-bindings[all]>=12.9.8,<13", "cuda-toolkit==12.*"],
        "cu13": ["cuda-bindings[all]<14,>=13.4.1", "cuda-toolkit==13.*"],
        "test": ["pytest"],
    }
    assert floors_from_extras(extras) == {12: (12, 9, 8), 13: (13, 4, 1)}
    assert floors_from_extras({"cu13": ['cuda-bindings>=13.4.1,<14 ; python_version >= "3.10"']}) == {13: (13, 4, 1)}
    # Other spellings of the same upper bound.
    assert floors_from_extras({"cu13": ["cuda-bindings>=13.4.1,<14.0"]}) == {13: (13, 4, 1)}
    assert floors_from_extras({"cu13": ["cuda-bindings>=13.4.1,==13.*"]}) == {13: (13, 4, 1)}


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(
    ("extras", "message"),
    [
        ({"cu13": ["cuda-toolkit==13.*"]}, "exactly one cuda-bindings requirement, found 0"),
        ({"cu13": ["cuda-bindings>=13.4.1", "cuda-bindings<14"]}, "exactly one cuda-bindings requirement, found 2"),
        ({"cu13": ["cuda-bindings>=13.4.1"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=13.4,<14"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=13.4.1,<14,==13.*"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=13.4.1,<13.5"]}, "must pin cuda-bindings as"),  # a minor, not a major
        ({"cu13": ["cuda-bindings>=12.9.8,<14"]}, "does not confine it to CUDA 13"),
        ({"cu13": ["cuda-bindings>=13.4.1,<13"]}, "does not confine it to CUDA 13"),
        ({"cu13": ["cuda-bindings>=13.4.1,<15"]}, "does not confine it to CUDA 13"),
        ({"cu13": ["cuda-bindings>=13.4.1,==12.*"]}, "does not confine it to CUDA 13"),
        ({"test": ["pytest"]}, "declares no cu<major> extra"),
    ],
)
def test_floors_from_extras_rejects_other_forms(extras, message):
    with pytest.raises(ValueError, match=message):
        floors_from_extras(extras)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_from_extras_ignores_lookalike_names():
    extras = {"cu13": ["cuda-bindings-extra>=1.0", "cuda-bindings>=13.4.1,<14"]}
    assert floors_from_extras(extras) == {13: (13, 4, 1)}


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("13.4.1", (13, 4, 1)),
        ("13.4.1a0", (13, 4, 1)),
        ("13.4.2.dev249+g471618971c2", (13, 4, 2)),
        ("12.9.8", (12, 9, 8)),
        (" 12.9.9.dev2 ", (12, 9, 9)),
        ("0.1.dev1+g0d22cb444", None),  # shallow clone
        ("13.4", None),
        ("", None),
    ],
)
def test_release_triple(version, expected):
    assert release_triple(version) == expected


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_formatting_helpers():
    assert format_version((13, 4, 1)) == "13.4.1"
    assert cuda_version_of((13, 4, 1)) == 13040
    assert cuda_version_of((12, 9, 8)) == 12090
    assert bindings_requirement((13, 4, 1)) == "cuda-bindings>=13.4.1,<14"
    assert bindings_requirement((13, 4, 1), below=(13, 5)) == "cuda-bindings>=13.4.1,<13.5"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_header_minor():
    assert header_minor(13040) == (13, 4)
    assert header_minor(12090) == (12, 9)
    assert header_minor(13000) == (13, 0)


class TestCheckInstalledBindings:
    FLOOR = (13, 4, 1)
    HEADER = cuda_version_of(FLOOR)

    def check(self, installed, major=13, header=None, floor=None, installed_header=None):
        """installed_header defaults to the header of the installed version's major.minor,
        the header that a release of that version was generated from."""
        if installed_header is None:
            triple = release_triple(installed) or (major, 0, 0)
            installed_header = cuda_version_of(triple)
        return check_installed_bindings(
            installed, installed_header, major, self.HEADER if header is None else header, floor or self.FLOOR, "1.3.0"
        )

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize(
        "installed",
        [
            "13.4.1",
            "13.4.2",
            "13.4.2.dev249+gabcdef0",  # main-built cuda-bindings in CI
            "13.5.0b1",  # newer cuda-bindings than the build: supported
        ],
    )
    def test_accepts_the_floor_and_newer(self, installed):
        assert self.check(installed) == release_triple(installed)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_older_than_the_floor_with_the_fix(self):
        with pytest.raises(ImportError) as excinfo:
            self.check("13.3.1")
        message = str(excinfo.value)
        assert "requires cuda-bindings >= 13.4.1 for CUDA 13" in message
        assert "but cuda-bindings 13.3.1 is installed" in message
        assert 'pip install -U "cuda-bindings>=13.4.1,<14"' in message  # double quotes work in cmd.exe too

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_bindings_generated_from_an_older_header_than_the_build(self):
        # The build uses 13.5 headers. A cuda-bindings generated from 13.4 lacks table entries.
        with pytest.raises(ImportError) as excinfo:
            self.check("13.4.1", header=self.HEADER + 10)
        message = str(excinfo.value)
        assert "was compiled against CUDA 13.5 headers and needs cuda-bindings 13.5 or newer" in message
        assert "but cuda-bindings 13.4.1 is installed" in message
        assert "This does not require a newer CUDA driver or toolkit" in message
        assert floor_mod.SUPPORT_URL in message
        assert 'pip install -U "cuda-bindings>=13.5.0,<14"' in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_header_rule_compares_headers_not_version_strings(self):
        # A development cuda-bindings carries the previous release's version string,
        # 13.4.2.dev5, but was generated from the new 13.5 header. The check accepts it.
        floor = (13, 4, 2)
        assert self.check("13.4.2.dev5+gabc", header=13050, floor=floor, installed_header=13050) == (13, 4, 2)
        # The check rejects the converse, a 13.5 version string generated from 13.4 headers.
        with pytest.raises(ImportError, match="needs cuda-bindings 13.5 or newer"):
            self.check("13.5.0", header=13050, floor=floor, installed_header=13040)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_floor_comes_from_the_build_record(self):
        # A build recorded with a lower floor and header accepts what the default floor rejects.
        assert self.check("13.3.0", header=cuda_version_of((13, 2, 0)), floor=(13, 2, 0)) == (13, 3, 0)
        # A higher recorded floor rejects what the default floor accepts.
        with pytest.raises(ImportError) as excinfo:
            self.check("13.4.1", floor=(13, 5, 0))
        message = str(excinfo.value)
        assert "requires cuda-bindings >= 13.5.0 for CUDA 13" in message
        assert 'pip install -U "cuda-bindings>=13.5.0,<14"' in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_another_major_than_the_build(self):
        with pytest.raises(ImportError, match="build is for CUDA 12, but the installed cuda-bindings is 13.4.1"):
            self.check("13.4.1", major=12, header=12090, floor=(12, 9, 8))

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("installed", ["11.8.0", "14.0.0"])
    def test_another_major_names_only_the_fix_it_knows(self, installed):
        """A single-build install cannot know which other builds exist, so the message must not
        promise one."""
        with pytest.raises(ImportError) as excinfo:
            self.check(installed, major=12, header=12090, floor=(12, 9, 8))
        message = str(excinfo.value)
        assert 'Install cuda-bindings 12.x with: pip install "cuda-bindings==12.*"' in message
        assert f"If a cuda.core build for CUDA {installed.split('.')[0]} exists, install it instead." in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("installed", ["0.1.dev1+g0d22cb444", "garbage"])
    def test_rejects_unparseable_versions(self, installed):
        with pytest.raises(
            ImportError,
            match=rf"requires a cuda-bindings 13\.x release, "
            rf"but the installed cuda-bindings version is {re.escape(installed)}",
        ):
            self.check(installed)


class TestImportTimeCheck:
    """`import cuda.core` runs check_installed_bindings against the installed build's record
    before it imports any extension module. A fake cuda.bindings in a child interpreter
    exercises the reject paths end to end. Issue #2783 asked for this test."""

    _CHILD = textwrap.dedent("""
        import sys, types
        fake = types.ModuleType("cuda.bindings")
        fake.__version__ = {version!r}
        driver = types.ModuleType("cuda.bindings.driver")
        driver.CUDA_VERSION = {cuda_version}
        fake.driver = driver
        sys.modules["cuda.bindings"] = fake
        sys.modules["cuda.bindings.driver"] = driver
        try:
            import cuda.core
        except ImportError as exc:
            print("IMPORTERROR:", exc)
            raise SystemExit(0)
        raise SystemExit("cuda.core imported with a fake cuda-bindings " + fake.__version__)
    """)

    @staticmethod
    def _build():
        from cuda.core import _build_info

        return _build_info.CUDA_MAJOR, _build_info.CUDA_VERSION, tuple(_build_info.CUDA_BINDINGS_FLOOR)

    def _import_error(self, version, cuda_version, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # the installed build, not a source tree
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", self._CHILD.format(version=version, cuda_version=cuda_version)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.startswith("IMPORTERROR:"), result.stdout
        return result.stdout

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_below_the_floor_fails_at_import_with_the_fix(self, tmp_path):
        major, cuda_version, floor = self._build()
        below = f"{major}.0.1"
        message = self._import_error(below, major * 1000, tmp_path)
        assert f"requires cuda-bindings >= {format_version(floor)} for CUDA {major}" in message
        assert f'pip install -U "cuda-bindings>={format_version(floor)},<{major + 1}"' in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_bindings_from_an_older_header_fail_at_import(self, tmp_path):
        major, cuda_version, floor = self._build()
        # At the floor by version, but generated from a header one minor below the build's.
        message = self._import_error(format_version(floor), cuda_version - 10, tmp_path)
        assert f"was compiled against CUDA {major}.{header_minor(cuda_version)[1]} headers" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_unparseable_version_fails_at_import(self, tmp_path):
        major, cuda_version, floor = self._build()
        # A major but no release triple. The build's major keeps the merged wheel on its cu<major>
        # build. A foreign major, such as a shallow clone's 0.1.dev1, stops earlier there with
        # "no build for CUDA 0".
        no_triple = f"{major}.4"
        message = self._import_error(no_triple, cuda_version, tmp_path)
        assert (
            f"requires a cuda-bindings {major}.x release, but the installed cuda-bindings version is {no_triple}"
            in message
        )
        # No major at all.
        message = self._import_error("garbage", cuda_version, tmp_path)
        assert "requires a cuda-bindings release, but the installed cuda-bindings version is 'garbage'" in message


class TestConsistencyHook:
    """Tests for toolshed/check_cuda_core_bindings_floor.py, which is also the pre-commit hook."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_checkout_is_consistent(self, hook):
        assert hook.check(REPO) == []
        assert hook.main(["--repo-root", str(REPO)]) == 0

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_a_malformed_extra_fails_the_hook(self, hook, tmp_path, capsys):
        # check() returns before it reads ci/versions.yml or the docs, so the tree needs only these two files.
        core = tmp_path / "cuda_core" / "cuda" / "core"
        core.mkdir(parents=True)
        shutil.copy(CUDA_CORE / "cuda" / "core" / "_bindings_floor.py", core)
        (tmp_path / "cuda_core" / "pyproject.toml").write_text(
            '[project.optional-dependencies]\ncu13 = ["cuda-bindings[all]>=13.4,<14"]\n', encoding="utf-8"
        )
        assert hook.main(["--repo-root", str(tmp_path)]) == 1
        assert "pyproject.toml: the 'cu13' extra must pin cuda-bindings" in capsys.readouterr().err

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_ci_toolkit_pins_must_not_sit_below_the_floors_minor(self, hook):
        floors = {12: (12, 9, 8), 13: (13, 4, 1)}
        good = 'cuda:\n  build:\n    version: "13.4.2"\n  prev_build:\n    version: "12.9.1"\n'
        assert hook.ci_pin_problems(floors, good) == []
        assert hook.ci_pin_problems(floors, good.replace("13.4.2", "13.5.0")) == []  # the toolkit-bump window
        stale = good.replace("13.4.2", "13.3.0")
        (problem,) = hook.ci_pin_problems(floors, stale)
        assert "cuda.build.version is 13.3 but the CUDA 13 floor is cuda-bindings 13.4.1" in problem
        (problem,) = hook.ci_pin_problems({13: (13, 4, 1)}, good)
        assert "pins CUDA 12, which has no cu12 extra" in problem
        (problem,) = hook.ci_pin_problems(floors, 'cuda:\n  build:\n    version: "13.4.2"\n')
        assert "no build or prev_build toolkit pin for CUDA 12" in problem

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_docs_must_use_the_substitutions(self, hook, tmp_path):
        docs = tmp_path / "cuda_core" / "docs" / "source"
        (docs / "release").mkdir(parents=True)
        (docs / "support.rst").write_text("``cuda-bindings`` >= |cuda-bindings-floor-cu13|\n", encoding="utf-8")
        (docs / "release" / "1.3.0-notes.rst").write_text("requires ``cuda-bindings`` >= 13.4.1\n", encoding="utf-8")
        assert hook.docs_problems(tmp_path) == []
        (docs / "install.rst").write_text("Install ``cuda-bindings`` >= 13.4.1.\n", encoding="utf-8")
        (docs / "api_nvml.rst").write_text("requires ``cuda-bindings`` 13.4.1 or later\n", encoding="utf-8")
        problems = hook.docs_problems(tmp_path)
        assert [p.split(":")[0] for p in problems] == [
            "cuda_core/docs/source/api_nvml.rst",
            "cuda_core/docs/source/install.rst",
        ]
        assert problems[1].startswith("cuda_core/docs/source/install.rst:1:")
