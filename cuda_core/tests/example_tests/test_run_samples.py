# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import subprocess
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


def _runtime_module(result: tuple[object, int | None] | BaseException) -> ModuleType:
    runtime = ModuleType("cuda.bindings.runtime")

    def get_device_count() -> tuple[object, int | None]:
        if isinstance(result, BaseException):
            raise result
        return result

    runtime.cudaGetDeviceCount = get_device_count
    return runtime


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ((0, 3), 3),
        ((100, None), 0),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_runtime_gpu_count_uses_runtime_result(
    monkeypatch: pytest.MonkeyPatch,
    result: tuple[object, int | None],
    expected: int,
) -> None:
    runtime = _runtime_module(result)
    monkeypatch.setattr(run_samples._runner.importlib, "import_module", lambda _name: runtime)

    assert run_samples._runner._runtime_gpu_count() == expected


@pytest.mark.agent_authored(model="gpt-5")
def test_gpu_count_prefers_cuda_runtime_over_host_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-visible")
    monkeypatch.setattr(run_samples._runner, "_runtime_gpu_count", lambda: 1)

    def unexpected_nvidia_smi(*args, **kwargs):
        raise AssertionError("nvidia-smi must not run when the CUDA Runtime is available")

    monkeypatch.setattr(run_samples._runner.subprocess, "run", unexpected_nvidia_smi)

    assert run_samples.get_gpu_count() == 1


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("", 0),
        ("   ", 0),
        ("-1", 0),
        ("none", 0),
        ("0", 1),
        ("0, 2", 2),
        ("GPU-01234567-89ab-cdef-0123-456789abcdef", 1),
        ("MIG-GPU-01234567-89ab-cdef-0123-456789abcdef/1/2", 1),
        ("MIG-01234567-89ab-cdef-0123-456789abcdef", 1),
        ("NoDevFiles", 0),
        ("0,-1,2", 1),
        ("0,invalid,2", 1),
        (",0", 0),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_gpu_count_honors_explicit_cuda_visible_devices_without_runtime(
    monkeypatch: pytest.MonkeyPatch, visible: str, expected: int
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(run_samples._runner, "_runtime_gpu_count", lambda: None)

    def unexpected_nvidia_smi(*args, **kwargs):
        raise AssertionError("nvidia-smi must not override CUDA_VISIBLE_DEVICES")

    monkeypatch.setattr(run_samples._runner.subprocess, "run", unexpected_nvidia_smi)

    assert run_samples.get_gpu_count() == expected


@pytest.mark.parametrize("exception", [ImportError("missing extension"), OSError("loader failure")])
@pytest.mark.agent_authored(model="gpt-5")
def test_runtime_gpu_count_handles_expected_import_failures(
    monkeypatch: pytest.MonkeyPatch, exception: BaseException
) -> None:
    def fail_import(name: str) -> ModuleType:
        raise exception

    monkeypatch.setattr(run_samples._runner.importlib, "import_module", fail_import)

    assert run_samples._runner._runtime_gpu_count() is None


@pytest.mark.parametrize("exception", [OSError("loader failure"), RuntimeError("runtime unavailable")])
@pytest.mark.agent_authored(model="gpt-5")
def test_runtime_gpu_count_handles_expected_call_failures(
    monkeypatch: pytest.MonkeyPatch, exception: BaseException
) -> None:
    runtime = _runtime_module(exception)
    monkeypatch.setattr(run_samples._runner.importlib, "import_module", lambda _name: runtime)

    assert run_samples._runner._runtime_gpu_count() is None


@pytest.mark.agent_authored(model="gpt-5")
def test_gpu_count_uses_nvidia_smi_only_when_visibility_is_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_samples._runner, "_runtime_gpu_count", lambda: None)
    smi = subprocess.CompletedProcess(
        ["nvidia-smi", "-L"],
        0,
        stdout="GPU 0: first\nGPU 1: second\n  MIG 1g.10gb Device 0: child\n",
    )
    monkeypatch.setattr(run_samples._runner.subprocess, "run", lambda *_args, **_kwargs: smi)

    assert run_samples.get_gpu_count() == 2


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
