# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import sys
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest
from packaging.utils import canonicalize_name

from . import run_samples


class _Distribution:
    def __init__(self, version: str, extras: tuple[str, ...] = ()) -> None:
        self.version = version
        self.metadata = Message()
        for extra in extras:
            self.metadata["Provides-Extra"] = extra


@pytest.fixture
def installed_distributions(monkeypatch: pytest.MonkeyPatch):
    installed: dict[str, _Distribution] = {}
    requested: list[str] = []

    def distribution(name: str) -> _Distribution:
        canonical_name = canonicalize_name(name)
        requested.append(canonical_name)
        try:
            return installed[canonical_name]
        except KeyError:
            raise importlib.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(run_samples.importlib.metadata, "distribution", distribution)

    def install(name: str, version: str, extras: tuple[str, ...] = ()) -> None:
        installed[canonicalize_name(name)] = _Distribution(version, extras)

    return install, requested


def _sample(tmp_path: Path, dependencies: str) -> Path:
    sample = tmp_path / "sample.py"
    sample.write_text(
        f"# /// script\n# dependencies = {dependencies}\n# ///\nprint('not executed')\n",
        encoding="utf-8",
    )
    return sample


def _sample_entry(root: Path, category: str, name: str) -> Path:
    sample_dir = root / category / name
    sample_dir.mkdir(parents=True)
    entry = sample_dir / f"{name}.py"
    entry.write_text("print('not executed')\n", encoding="utf-8")
    return entry


@pytest.mark.agent_authored(model="gpt-5")
def test_collection_preserves_duplicate_leaf_names_with_path_aware_ids(tmp_path: Path) -> None:
    _sample_entry(tmp_path, "first", "duplicate")
    _sample_entry(tmp_path, "second", "duplicate")

    entries = run_samples.collect_sample_entries(tmp_path, "cuda_bindings")

    assert list(entries) == [
        "cuda_bindings/first/duplicate",
        "cuda_bindings/second/duplicate",
    ]


@pytest.mark.agent_authored(model="gpt-5")
def test_namespaced_config_distinguishes_duplicate_leaf_names(tmp_path: Path) -> None:
    first = _sample_entry(tmp_path, "first", "duplicate")
    second = _sample_entry(tmp_path, "second", "duplicate")
    config = {
        "cuda_bindings/first/duplicate": {"python": {"args": ["first"]}},
        "cuda_bindings/second/duplicate": {"python": {"args": ["second"]}},
        "duplicate": {"python": {"args": ["legacy"]}},
    }

    first_plan = run_samples.build_run_plan(
        first,
        config,
        gpu_count=1,
        sample_key="cuda_bindings/first/duplicate",
    )
    second_plan = run_samples.build_run_plan(
        second,
        config,
        gpu_count=1,
        sample_key="cuda_bindings/second/duplicate",
    )

    assert first_plan.args == ["first"]
    assert second_plan.args == ["second"]


@pytest.mark.agent_authored(model="gpt-5")
def test_namespaced_config_lookup_falls_back_to_legacy_leaf_key(tmp_path: Path) -> None:
    entry = _sample_entry(tmp_path, "category", "legacy")

    plan = run_samples.build_run_plan(
        entry,
        {"legacy": {"python": {"args": ["compatible"]}}},
        gpu_count=1,
        sample_key="cuda_core/category/legacy",
    )

    assert plan.args == ["compatible"]


@pytest.mark.parametrize(
    ("version", "requirement", "is_met"),
    [
        ("13.2.0", "demo>=13,<13.3", True),
        ("13.3.1", "demo>=13,<13.3", False),
        ("12.9.0", "demo>=13,<13.3", False),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_dependency_version_bounds(
    tmp_path: Path, installed_distributions, version: str, requirement: str, is_met: bool
) -> None:
    install, _ = installed_distributions
    install("demo", version)

    missing = run_samples.missing_dependencies(_sample(tmp_path, f"[{requirement!r}]"))

    assert (missing == []) is is_met
    if not is_met:
        assert f"version {version} does not satisfy <13.3,>=13" in missing[0]


@pytest.mark.agent_authored(model="gpt-5")
def test_cuda_python_uses_cuda_bindings_as_version_provider(tmp_path: Path, installed_distributions) -> None:
    install, requested = installed_distributions
    install("cuda-bindings", "13.2.0")

    missing = run_samples.missing_dependencies(_sample(tmp_path, "['cuda-python>=13,<13.3']"))

    assert missing == []
    assert requested == ["cuda-bindings"]


@pytest.mark.agent_authored(model="gpt-5")
def test_cupy_cuda_requirement_accepts_conda_cupy_distribution(tmp_path: Path, installed_distributions) -> None:
    install, requested = installed_distributions
    install("cupy", "14.0.1")

    missing = run_samples.missing_dependencies(_sample(tmp_path, "['cupy-cuda13x>=14']"))

    assert missing == []
    assert requested == ["cupy-cuda13x", "cupy"]


@pytest.mark.agent_authored(model="gpt-5")
def test_environment_markers_control_dependency_checks(tmp_path: Path, installed_distributions) -> None:
    _, requested = installed_distributions
    dependencies = "[\"ignored; python_version < '1'\", \"required; python_version >= '3'\"]"

    missing = run_samples.missing_dependencies(_sample(tmp_path, dependencies))

    assert len(missing) == 1
    assert missing[0].startswith('required; python_version >= "3"')
    assert requested == ["required"]


@pytest.mark.agent_authored(model="gpt-5")
def test_requested_extras_must_be_provided(tmp_path: Path, installed_distributions) -> None:
    install, _ = installed_distributions
    install("demo", "1.0", extras=("other",))

    missing = run_samples.missing_dependencies(_sample(tmp_path, "['demo[cu13]>=1']"))

    assert len(missing) == 1
    assert "does not provide extra(s): cu13" in missing[0]


@pytest.mark.agent_authored(model="gpt-5")
def test_provided_extra_satisfies_requirement(tmp_path: Path, installed_distributions) -> None:
    install, _ = installed_distributions
    install("demo", "1.0", extras=("CU13",))

    missing = run_samples.missing_dependencies(_sample(tmp_path, "['demo[cu13]>=1']"))

    assert missing == []


@pytest.mark.parametrize(
    "dependencies",
    [
        "['unterminated'",
        "['not a valid requirement ???']",
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_malformed_dependency_metadata_is_an_error(tmp_path: Path, dependencies: str) -> None:
    sample = _sample(tmp_path, dependencies)

    result = run_samples.run_sample(run_samples.RunPlan(sample, [], [], timeout=1))

    assert result.status == "ERROR"
    assert result.return_code == -1
    assert "invalid PEP 723 metadata" in result.detail


@pytest.mark.agent_authored(model="gpt-5")
def test_importable_module_does_not_satisfy_distribution_requirement(
    tmp_path: Path, installed_distributions, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = installed_distributions
    monkeypatch.setitem(sys.modules, "cupy", ModuleType("cupy"))

    missing = run_samples.missing_dependencies(_sample(tmp_path, "['cupy-cuda13x>=14']"))

    assert len(missing) == 1
    assert "no provider distribution installed" in missing[0]
