# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from check_pixi_samples_source_pruning import LOCAL_PACKAGES, PLATFORMS, is_known_samples_source_pruning

SCRIPT = Path(__file__).parent.parent / "check_pixi_samples_source_pruning.py"
MANIFEST = """[feature.samples.dependencies]
cuda-core = { path = "." }
[feature.local-deps.dependencies]
cuda-bindings = { path = "../cuda_bindings" }
cuda-pathfinder = { path = "../cuda_pathfinder" }
[feature.samples.pypi-options.dependency-overrides]
cuda-bindings = { version = "*", env-markers = "python_version < '0'" }
cuda-core = { version = "*", env-markers = "python_version < '0'" }
cuda-pathfinder = { version = "*", env-markers = "python_version < '0'" }
[environments]
samples = { features = ["cu13", "samples", "local-deps"], solve-group = "samples" }
"""


@pytest.fixture
def lockfiles():
    original = ["version: 7\nplatforms:\n"]
    for alias, platform in PLATFORMS.items():
        original.append(f"- name: {alias}\n  subdir: {platform}\n")
    original.append("environments:\n  cu13:\n    packages: {}\n  samples:\n    packages:\n")
    definitions = []
    references = []
    for platform_index, alias in enumerate(PLATFORMS):
        original.append(f"      {alias}:\n      - conda: python-{alias}\n")
        for package_index, (name, path) in enumerate(LOCAL_PACKAGES.items()):
            identifier = f"{platform_index * 3 + package_index:08x}"
            reference = f"      - conda_source: {name}[{identifier}] @ {path}\n"
            original.append(reference)
            references.append(reference)
            definitions.append(f"- conda_source: {name}[{identifier}] @ {path}\n  depends:\n  - python\n")
    original.append("packages:\n")
    original.extend(definitions)
    text = "".join(original)
    repaired = text
    for reference in references:
        repaired = repaired.replace(reference, "")
    return text, repaired, references


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_exact_nine_samples_removals_are_recognized(lockfiles):
    original, repaired, _ = lockfiles
    assert is_known_samples_source_pruning(original, repaired)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_reverse_cycle_and_unchanged_lock_are_rejected(lockfiles):
    original, repaired, _ = lockfiles
    assert not is_known_samples_source_pruning(repaired, original)
    assert not is_known_samples_source_pruning(original, original)


@pytest.mark.parametrize("field", ["version: 7", "depends:", "python-p1", "subdir: linux-64"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_any_other_repaired_byte_change_is_rejected(lockfiles, field):
    original, repaired, _ = lockfiles
    assert not is_known_samples_source_pruning(original, repaired.replace(field, field + "-changed", 1))


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_incomplete_removal_and_missing_original_reference_are_rejected(lockfiles):
    original, repaired, references = lockfiles
    assert not is_known_samples_source_pruning(original, repaired.replace("      p1:\n", "      p1:\n" + references[0]))
    assert not is_known_samples_source_pruning(original.replace(references[0], ""), repaired)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_duplicate_reference_is_rejected(lockfiles):
    original, repaired, references = lockfiles
    duplicate = original.replace(references[0], references[0] * 2)
    assert not is_known_samples_source_pruning(duplicate, repaired)


@pytest.mark.parametrize("replacement", ["../published", "../cuda_bindings-extra"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_wrong_source_path_is_rejected(lockfiles, replacement):
    original, repaired, _ = lockfiles
    assert not is_known_samples_source_pruning(
        original.replace("../cuda_bindings", replacement), repaired.replace("../cuda_bindings", replacement)
    )


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_source_record_definition_is_rejected(lockfiles):
    original, repaired, _ = lockfiles
    definition = "- conda_source: cuda-bindings[00000000] @ ../cuda_bindings\n"
    assert not is_known_samples_source_pruning(original.replace(definition, ""), repaired.replace(definition, ""))


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_changed_reference_identifier_is_rejected(lockfiles):
    original, repaired, references = lockfiles
    original = original.replace(references[0], references[0].replace("00000000", "ffffffff"))
    assert not is_known_samples_source_pruning(original, repaired)


@pytest.mark.parametrize("replacement", ["      p1:\n      p1:\n", "      unexpected:\n"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_duplicate_or_unexpected_platform_section_is_rejected(lockfiles, replacement):
    original, repaired, _ = lockfiles
    assert not is_known_samples_source_pruning(
        original.replace("      p1:\n", replacement), repaired.replace("      p1:\n", replacement)
    )


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_other_environment_removal_and_crlf_are_rejected(lockfiles):
    original, repaired, references = lockfiles
    original = original.replace("    packages: {}\n", "    packages:\n      p1:\n" + references[0])
    assert not is_known_samples_source_pruning(original, repaired)
    assert not is_known_samples_source_pruning(original.replace("\n", "\r\n"), repaired.replace("\n", "\r\n"))


@pytest.fixture
def cli_inputs(tmp_path, lockfiles):
    original, repaired, _ = lockfiles
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)  # noqa: S603, S607
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],  # noqa: S607
        input=original,
        text=True,
        capture_output=True,
        check=True,
        cwd=tmp_path,
    ).stdout.strip()
    (tmp_path / "pixi.lock").write_text(repaired, encoding="utf-8")
    (tmp_path / "pixi.toml").write_text(MANIFEST, encoding="utf-8")
    return tmp_path, blob


def _run_cli(cli_inputs, **overrides):
    cwd, blob = cli_inputs
    arguments = {
        "original-blob": blob,
        "lockfile": "pixi.lock",
        "manifest-path": "pixi.toml",
        "pixi-version": "v0.73.0",
        "check-status": "1",
    }
    arguments.update(overrides)
    return subprocess.run(  # noqa: S603 - controlled fixture arguments, without a shell.
        [sys.executable, str(SCRIPT), *(item for key, value in arguments.items() for item in (f"--{key}", value))],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_recognizes_defect(cli_inputs):
    assert _run_cli(cli_inputs).returncode == 0


@pytest.mark.parametrize("overrides", [{"pixi-version": "0.81.0"}, {"check-status": "0"}, {"check-status": "124"}])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_rejects_other_versions_and_statuses(cli_inputs, overrides):
    assert _run_cli(cli_inputs, **overrides).returncode == 1


@pytest.mark.parametrize(
    "replacement",
    [
        ("python_version < '0'", "python_version >= '0'"),
        ('path = "../cuda_bindings"', 'path = "../published"'),
        ('"cu13", "samples", "local-deps"', '"cu13", "samples"'),
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_rejects_changed_manifest_configuration(cli_inputs, replacement):
    cwd, _ = cli_inputs
    (cwd / "pixi.toml").write_text(MANIFEST.replace(*replacement), encoding="utf-8")
    assert _run_cli(cli_inputs).returncode == 1


@pytest.mark.parametrize(
    "overrides", [{"original-blob": "invalid"}, {"original-blob": "0" * 40}, {"lockfile": "missing"}]
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_reports_operational_errors(cli_inputs, overrides):
    result = _run_cli(cli_inputs, **overrides)
    assert result.returncode == 2
    assert result.stderr


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_reports_invalid_toml(cli_inputs):
    cwd, _ = cli_inputs
    (cwd / "pixi.toml").write_text("[invalid", encoding="utf-8")
    result = _run_cli(cli_inputs)
    assert result.returncode == 2
    assert "could not read its inputs" in result.stderr


@pytest.mark.parametrize("manifest", ['feature = "invalid"', '[feature]\nsamples = "invalid"', "environments = []"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_rejects_unexpected_manifest_table_types(cli_inputs, manifest):
    cwd, _ = cli_inputs
    (cwd / "pixi.toml").write_text(manifest, encoding="utf-8")
    result = _run_cli(cli_inputs)
    assert result.returncode == 1
    assert not result.stderr


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cli_reports_invalid_utf8(cli_inputs):
    cwd, _ = cli_inputs
    (cwd / "pixi.lock").write_bytes(b"\xff")
    result = _run_cli(cli_inputs)
    assert result.returncode == 2
    assert "could not read its inputs" in result.stderr
