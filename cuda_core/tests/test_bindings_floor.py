# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cuda-bindings version floor (cuda/core/_bindings_floor.py) and the
import-time check built on it.

Source-tree properties and pure functions only: no GPU, so this file also runs
with --noconftest (conftest.py initializes CUDA). The consistency tests read
pyproject.toml and ci/versions.yml from the checkout, so they need the source
tree next to the tests, which every CI job that runs tests/ has.

    pytest tests/test_bindings_floor.py -v --noconftest
"""

import re
from pathlib import Path

import pytest

from cuda.core import _bindings_floor as floor_mod
from cuda.core._bindings_floor import (
    CUDA_BINDINGS_FLOOR,
    SUPPORTED_CUDA_MAJORS,
    check_installed_bindings,
    cuda_version_of,
    format_version,
    pip_requirement,
    release_triple,
    required_minimum,
)

CUDA_CORE = Path(__file__).resolve().parent.parent
REPO = CUDA_CORE.parent


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floor_module_is_import_free():
    """build_hooks.py loads it by file path during the build; it must stay standard-library only."""
    source = Path(floor_mod.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+(\w+)", source, re.M)
    assert set(imports) <= {"__future__", "re"}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_floors_are_release_triples_of_their_major():
    assert SUPPORTED_CUDA_MAJORS == (12, 13)
    for major, floor in CUDA_BINDINGS_FLOOR.items():
        assert len(floor) == 3
        assert floor[0] == major


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
    assert pip_requirement(13) == f"cuda-bindings>={format_version(CUDA_BINDINGS_FLOOR[13])},==13.*"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_required_minimum_is_the_floor_or_the_header_minor():
    floor = CUDA_BINDINGS_FLOOR[13]
    header_at_floor = cuda_version_of(floor)
    assert required_minimum(13, header_at_floor) == floor
    # A build against a newer header than the floor's minor demands that minor:
    # the driver-pointer keys are derived from the header's macros.
    assert required_minimum(13, header_at_floor + 10) == (13, floor[1] + 1, 0)
    # An older header cannot win over the floor.
    assert required_minimum(13, 13000) == floor


class TestCheckInstalledBindings:
    FLOOR = CUDA_BINDINGS_FLOOR[13]
    HEADER = cuda_version_of(FLOOR)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize(
        "installed",
        [
            format_version(FLOOR),
            f"{FLOOR[0]}.{FLOOR[1]}.{FLOOR[2] + 1}",
            f"{FLOOR[0]}.{FLOOR[1]}.{FLOOR[2] + 1}.dev249+gabcdef0",  # main-built bindings in CI
            f"{FLOOR[0]}.{FLOOR[1] + 1}.0b1",  # newer bindings than the build: supported
        ],
    )
    def test_accepts_the_floor_and_newer(self, installed):
        assert check_installed_bindings(installed, 13, self.HEADER, "1.3.0") == release_triple(installed)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_older_than_the_floor_with_the_fix(self):
        older = f"{self.FLOOR[0]}.{self.FLOOR[1] - 1}.1"
        with pytest.raises(ImportError) as excinfo:
            check_installed_bindings(older, 13, self.HEADER, "1.3.0")
        message = str(excinfo.value)
        assert f"requires cuda-bindings >= {format_version(self.FLOOR)} for CUDA 13" in message
        assert f"(found {older})" in message
        assert f"pip install -U 'cuda-bindings>={format_version(self.FLOOR)},==13.*'" in message

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_a_minor_older_than_the_header(self):
        # Built against a header one minor above the floor; the floor itself no longer suffices.
        header = self.HEADER + 10
        with pytest.raises(ImportError, match=rf"requires cuda-bindings >= 13\.{self.FLOOR[1] + 1}\.0"):
            check_installed_bindings(format_version(self.FLOOR), 13, header, "1.3.0")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejects_another_major_than_the_build(self):
        with pytest.raises(ImportError, match="build is for CUDA 12, but the installed cuda-bindings is 13.4.1"):
            check_installed_bindings("13.4.1", 12, 12090, "1.3.0")

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.parametrize("installed", ["0.1.dev1+g0d22cb444", "11.8.0", "14.0.0", "garbage"])
    def test_rejects_unsupported_or_unparseable_versions(self, installed):
        with pytest.raises(
            ImportError, match=rf"cuda-bindings 12\.x or 13\.x must be installed \(found {re.escape(installed)}\)"
        ):
            check_installed_bindings(installed, 13, self.HEADER, "1.3.0")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_pyproject_extras_pin_the_floor():
    """The static `cu12`/`cu13` extras cannot read the module; keep them in step by test."""
    pyproject = (CUDA_CORE / "pyproject.toml").read_text(encoding="utf-8")
    for major in SUPPORTED_CUDA_MAJORS:
        m = re.search(rf'^cu{major} = \["cuda-bindings\[all\]([^"]+)"', pyproject, re.M)
        assert m, f"no cu{major} extra in pyproject.toml"
        assert m.group(1) == pip_requirement(major).removeprefix("cuda-bindings")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_ci_toolkit_pins_match_the_floors_minor():
    """CI builds each major against the toolkit pinned in ci/versions.yml; the
    build requires that header's major.minor to equal the bindings', so the
    floor of each major must sit in the same minor as its toolkit pin."""
    versions = (REPO / "ci" / "versions.yml").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^\s+(build|prev_build):\s*\n\s+version:\s*\"(\d+\.\d+)", versions, re.M))
    assert set(pins) == {"build", "prev_build"}, pins
    by_major = {int(v.split(".")[0]): v for v in pins.values()}
    for major, floor in CUDA_BINDINGS_FLOOR.items():
        assert by_major[major] == f"{floor[0]}.{floor[1]}", (
            f"ci/versions.yml builds CUDA {major} against {by_major[major]} but the floor is {format_version(floor)}"
        )
