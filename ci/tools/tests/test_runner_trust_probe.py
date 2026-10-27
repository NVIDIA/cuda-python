# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the runner trust probe workflow and helper scripts."""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "runner-trust-probe.yml"
LINUX_SCRIPT = REPO_ROOT / "ci" / "tools" / "runner_trust_probe.sh"
WINDOWS_SCRIPT = REPO_ROOT / "ci" / "tools" / "runner_trust_probe.ps1"
EMSIGN_G3_SHA256 = "86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B"


def _split_pem_bundle(bundle_path: Path) -> list[tuple[str, str]]:
    """Return (pem_text, colon_sha256_fingerprint) for each cert in a PEM bundle."""
    certs: list[tuple[str, str]] = []
    current: list[str] = []
    b64_lines: list[str] = []
    in_cert = False
    for line in bundle_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if "-----BEGIN CERTIFICATE-----" in line:
            in_cert = True
            current = [line]
            b64_lines = []
        elif "-----END CERTIFICATE-----" in line and in_cert:
            current.append(line)
            der = base64.b64decode("".join(b64_lines))
            digest = hashlib.sha256(der).hexdigest().upper()
            fp = ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))
            certs.append(("".join(current), fp))
            in_cert = False
        elif in_cert:
            current.append(line)
            b64_lines.append(line.strip())
    return certs


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_runner_trust_probe_workflow_structure() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if True in workflow:
        workflow["on"] = workflow.pop(True)

    assert {"schedule", "workflow_dispatch"} <= set(workflow["on"])
    assert set(workflow["on"]) <= {"schedule", "workflow_dispatch", "pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {
        "compute-matrix",
        "probe-linux-host",
        "probe-linux-gpu",
        "probe-windows",
    }


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_linux_probe_passes_on_clean_bundle_and_fails_with_emsign_root(tmp_path: Path) -> None:
    system_bundle = Path("/etc/ssl/certs/ca-certificates.crt")
    if not system_bundle.is_file():
        pytest.skip("/etc/ssl/certs/ca-certificates.crt not present")

    certs = _split_pem_bundle(system_bundle)
    clean_pems = [pem for pem, fp in certs if fp != EMSIGN_G3_SHA256][:5]
    assert clean_pems, "Expected at least one non-emSign certificate in system bundle"

    clean_bundle = tmp_path / "clean.crt"
    clean_bundle.write_text("".join(clean_pems), encoding="utf-8")
    summary = tmp_path / "summary.md"

    env = dict(os.environ)
    env["TRUST_PROBE_CA_BUNDLE"] = str(clean_bundle)
    env["TRUST_PROBE_TEST_ENDPOINT"] = ""
    env["GITHUB_STEP_SUMMARY"] = str(summary)

    res_clean = subprocess.run(  # noqa: S603
        [str(LINUX_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res_clean.returncode == 0, res_clean.stdout + res_clean.stderr
    assert "[PASS] Root anchor absent" in res_clean.stdout
    assert "| Root anchor absent | **PASS** |" in summary.read_text(encoding="utf-8")

    # Test that presence of the fingerprint causes a gating failure.
    # Reuse any certificate from the bundle and override TRUST_PROBE_TARGET_SHA256
    # (or use the actual emSign cert if present in the host bundle).
    emsign_pems = [pem for pem, fp in certs if fp == EMSIGN_G3_SHA256]
    if emsign_pems:
        dirty_bundle = tmp_path / "dirty.crt"
        dirty_bundle.write_text("".join(clean_pems) + emsign_pems[0], encoding="utf-8")
        env_dirty = dict(env)
        env_dirty["TRUST_PROBE_CA_BUNDLE"] = str(dirty_bundle)
    else:
        env_dirty = dict(env)
        env_dirty["TRUST_PROBE_TARGET_SHA256"] = next(fp for _pem, fp in certs)

    res_dirty = subprocess.run(  # noqa: S603
        [str(LINUX_SCRIPT)],
        env=env_dirty,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res_dirty.returncode == 1, res_dirty.stdout + res_dirty.stderr
    assert "[FAIL] Root anchor absent" in res_dirty.stdout


@pytest.mark.agent_authored(model="gemini-4-argon")
def test_windows_probe_script_is_ascii_and_contains_fingerprint() -> None:
    text = WINDOWS_SCRIPT.read_text(encoding="utf-8")
    assert text.isascii()
    assert EMSIGN_G3_SHA256 in text
    assert "Cert:\\LocalMachine\\Root" in text
    assert "Cert:\\LocalMachine\\Disallowed" in text
