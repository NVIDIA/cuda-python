# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cuda-bindings version floor (cuda/core/_bindings_floor.py): reading it
from the pyproject extras, the import-time check built on it, and the
consistency hook that guards ci/versions.yml and the docs.

Source-tree properties and pure functions only: no GPU, so this file also runs
with --noconftest (conftest.py initializes CUDA). The consistency tests read
pyproject.toml and ci/versions.yml from the checkout, so they need the source
tree next to the tests, which every CI job that runs tests/ has.

    pytest tests/test_bindings_floor.py -v --noconftest
"""

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

from cuda.core import _bindings_floor as floor_mod
from cuda.core._bindings_floor import (
    bindings_requirement,
    check_installed_bindings,
    cuda_version_of,
    floors_from_extras,
    format_version,
    release_triple,
    required_minimum,
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
        pytest.skip(f"{HOOK} is not in this tree; the hook tests need the monorepo checkout")
    return _load("check_cuda_core_bindings_floor", HOOK)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floor_module_is_import_free():
    """build_hooks.py, conf.py and the hook load it by file path; it must stay standard-library only."""
    source = Path(floor_mod.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+(\w+)", source, re.M)
    assert set(imports) <= {"__future__", "collections", "re"}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_come_from_the_pyproject_extras(hook):
    """The extras are the single source; reading them back gives one release triple per major."""
    floors = hook.read_floors(REPO)
    assert sorted(floors) == [12, 13]
    for major, floor in floors.items():
        assert floor[0] == major
        assert len(floor) == 3


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_from_extras_accepts_the_declared_form():
    extras = {
        "cu12": ["cuda-bindings[all]>=12.9.8,==12.*", "cuda-toolkit==12.*"],
        "cu13": ["cuda-bindings[all]==13.*,>=13.4.1", "cuda-toolkit==13.*"],
        "test": ["pytest"],
    }
    assert floors_from_extras(extras) == {12: (12, 9, 8), 13: (13, 4, 1)}
    assert floors_from_extras({"cu13": ['cuda-bindings>=13.4.1,==13.* ; python_version >= "3.10"']}) == {13: (13, 4, 1)}


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(
    ("extras", "message"),
    [
        ({"cu13": ["cuda-toolkit==13.*"]}, "exactly one cuda-bindings requirement, found 0"),
        ({"cu13": ["cuda-bindings>=13.4.1", "cuda-bindings==13.*"]}, "exactly one cuda-bindings requirement, found 2"),
        ({"cu13": ["cuda-bindings>=13.4.1"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=13.4,==13.*"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=13.4.1,==13.*,<14"]}, "must pin cuda-bindings as"),
        ({"cu13": ["cuda-bindings>=12.9.8,==13.*"]}, "majors do not match CUDA 13"),
        ({"cu13": ["cuda-bindings>=13.4.1,==12.*"]}, "majors do not match CUDA 13"),
        ({"test": ["pytest"]}, "declares no cu<major> extra"),
    ],
)
def test_floors_from_extras_rejects_other_forms(extras, message):
    with pytest.raises(ValueError, match=message):
        floors_from_extras(extras)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_from_extras_ignores_lookalike_names():
    extras = {"cu13": ["cuda-bindings-extra>=1.0", "cuda-bindings>=13.4.1,==13.*"]}
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
    assert bindings_requirement((13, 4, 1)) == "cuda-bindings>=13.4.1,==13.*"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_required_minimum_is_the_floor_or_the_header_minor():
    floor = (13, 4, 1)
    header_at_floor = cuda_version_of(floor)
    assert required_minimum(floor, header_at_floor) == floor
    # A build against a newer header than the floor's minor demands that minor:
    # the driver-pointer keys are derived from the header's macros.
    assert required_minimum(floor, header_at_floor + 10) == (13, 5, 0)
    # An older header cannot win over the floor.
    assert required_minimum(floor, 13000) == floor


class TestCheckInstalledBindings:
    FLOOR = (13, 4, 1)
    HEADER = cuda_version_of(FLOOR)

    def check(self, installed, major=13, header=None, floor=None):
        return check_installed_bindings(
            installed, major, self.HEADER if header is None else header, floor or self.FLOOR, "1.3.0"
        )

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize(
        "installed",
        [
            "13.4.1",
            "13.4.2",
            "13.4.2.dev249+gabcdef0",  # main-built bindings in CI
            "13.5.0b1",  # newer bindings than the build: supported
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
        assert "(found 13.3.1)" in message
        assert "pip install -U 'cuda-bindings>=13.4.1,==13.*'" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_a_minor_older_than_the_header(self):
        # Built against a header one minor above the floor; the floor itself no longer suffices.
        with pytest.raises(ImportError, match=r"requires cuda-bindings >= 13\.5\.0"):
            self.check("13.4.1", header=self.HEADER + 10)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_floor_comes_from_the_build_record(self):
        # A build recorded with a lower floor (and header) accepts what the default floor rejects.
        assert self.check("13.3.0", header=cuda_version_of((13, 2, 0)), floor=(13, 2, 0)) == (13, 3, 0)
        # A higher recorded floor rejects what the default floor accepts, even with the header at 13.4.
        with pytest.raises(ImportError) as excinfo:
            self.check("13.4.1", floor=(13, 5, 0))
        message = str(excinfo.value)
        assert "requires cuda-bindings >= 13.5.0 for CUDA 13" in message
        assert "pip install -U 'cuda-bindings>=13.5.0,==13.*'" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_another_major_than_the_build(self):
        with pytest.raises(ImportError, match="build is for CUDA 12, but the installed cuda-bindings is 13.4.1"):
            self.check("13.4.1", major=12, header=12090, floor=(12, 9, 8))

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("installed", ["11.8.0", "14.0.0"])
    def test_another_major_names_only_the_fix_it_knows(self, installed):
        """A plain (single-build) install cannot know which other builds exist, so the message
        must not promise one."""
        with pytest.raises(ImportError) as excinfo:
            self.check(installed, major=12, header=12090, floor=(12, 9, 8))
        message = str(excinfo.value)
        assert "Install cuda-bindings 12.x (pip install 'cuda-bindings==12.*')" in message
        assert f"build for CUDA {installed.split('.')[0]} if one exists" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("installed", ["0.1.dev1+g0d22cb444", "garbage"])
    def test_rejects_unparseable_versions(self, installed):
        with pytest.raises(
            ImportError, match=rf"a cuda-bindings 13\.x release is required \(found {re.escape(installed)}\)"
        ):
            self.check(installed)


class TestConsistencyHook:
    """toolshed/check_cuda_core_bindings_floor.py, also the pre-commit hook."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_the_checkout_is_consistent(self, hook):
        assert hook.check(REPO) == []
        assert hook.main(["--repo-root", str(REPO)]) == 0

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_a_malformed_extra_fails_the_hook(self, hook, tmp_path, capsys):
        # check() returns before reading ci/versions.yml or the docs, so the tree needs only these two files.
        core = tmp_path / "cuda_core" / "cuda" / "core"
        core.mkdir(parents=True)
        shutil.copy(CUDA_CORE / "cuda" / "core" / "_bindings_floor.py", core)
        (tmp_path / "cuda_core" / "pyproject.toml").write_text(
            '[project.optional-dependencies]\ncu13 = ["cuda-bindings[all]>=13.4,==13.*"]\n', encoding="utf-8"
        )
        assert hook.main(["--repo-root", str(tmp_path)]) == 1
        assert "pyproject.toml: the 'cu13' extra must pin cuda-bindings" in capsys.readouterr().err

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_ci_toolkit_pins_must_sit_in_the_floors_minor(self, hook):
        floors = {12: (12, 9, 8), 13: (13, 4, 1)}
        good = 'cuda:\n  build:\n    version: "13.4.2"\n  prev_build:\n    version: "12.9.1"\n'
        assert hook.ci_pin_problems(floors, good) == []
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
