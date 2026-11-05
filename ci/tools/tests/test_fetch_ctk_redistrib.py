# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fetch_ctk_redistrib import (
    compute_file_sha256,
    get_component_relative_path,
    get_component_sha256,
    main,
    verify_metadata_sha256,
)

# Minimal redistrib-shaped metadata. "cccl" exercises the cuda_cccl -> cccl
# alias; "no_sha" lacks the sha256 field.
FAKE_METADATA = {
    "cuda_nvcc": {
        "linux-x86_64": {
            "relative_path": "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.0.0-archive.tar.xz",
            "sha256": "a" * 64,
        },
    },
    "cccl": {
        "linux-x86_64": {
            "relative_path": "cccl/linux-x86_64/cccl-linux-x86_64-13.0.0-archive.tar.xz",
            "sha256": "b" * 64,
        },
    },
    "no_sha": {
        "linux-x86_64": {
            "relative_path": "no_sha/linux-x86_64/no_sha.tar.xz",
        },
    },
}


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_sha256_returns_hash():
    assert get_component_sha256(FAKE_METADATA, host_platform="linux-64", component="cuda_nvcc") == "a" * 64


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_sha256_resolves_alias():
    # cuda_cccl is absent; it must resolve to the renamed cccl key.
    assert get_component_sha256(FAKE_METADATA, host_platform="linux-64", component="cuda_cccl") == "b" * 64


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_sha256_unknown_component_raises():
    with pytest.raises(KeyError):
        get_component_sha256(FAKE_METADATA, host_platform="linux-64", component="does_not_exist")


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_sha256_unavailable_subdir_raises():
    with pytest.raises(KeyError):
        get_component_sha256(FAKE_METADATA, host_platform="win-64", component="cuda_nvcc")


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_sha256_missing_field_raises():
    with pytest.raises(KeyError):
        get_component_sha256(FAKE_METADATA, host_platform="linux-64", component="no_sha")


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_get_component_relative_path_still_resolves():
    assert get_component_relative_path(FAKE_METADATA, host_platform="linux-64", component="cuda_nvcc").endswith(
        "cuda_nvcc-linux-x86_64-13.0.0-archive.tar.xz"
    )


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_verify_metadata_match(tmp_path, monkeypatch):
    f = tmp_path / "redistrib.json"
    f.write_bytes(b"pinned-metadata")
    monkeypatch.setitem(
        sys.modules["fetch_ctk_redistrib"].KNOWN_REDISTRIB_SHA256,
        "1.2.3",
        compute_file_sha256(str(f)),
    )
    assert verify_metadata_sha256(cuda_version="1.2.3", metadata_path=str(f)) == 0


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_verify_metadata_mismatch_fails_closed(tmp_path, monkeypatch):
    f = tmp_path / "redistrib.json"
    f.write_bytes(b"tampered-metadata")
    monkeypatch.setitem(
        sys.modules["fetch_ctk_redistrib"].KNOWN_REDISTRIB_SHA256,
        "1.2.3",
        "f" * 64,
    )
    assert verify_metadata_sha256(cuda_version="1.2.3", metadata_path=str(f)) == 1


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_verify_metadata_unknown_version_warns(tmp_path, capsys):
    f = tmp_path / "redistrib.json"
    f.write_bytes(b"whatever")
    # An unpinned version must not fail closed (it would break CI on every new
    # CTK), but it must emit a warning.
    assert verify_metadata_sha256(cuda_version="99.9.9", metadata_path=str(f)) == 0
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_main_component_sha256(tmp_path, capsys):
    f = tmp_path / "redistrib.json"
    f.write_text(json.dumps(FAKE_METADATA), encoding="utf-8")
    rc = main(
        [
            "component-sha256",
            "--host-platform",
            "linux-64",
            "--component",
            "cuda_nvcc",
            "--metadata-path",
            str(f),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "a" * 64


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_main_verify_metadata_mismatch_returns_1(tmp_path, monkeypatch):
    f = tmp_path / "redistrib.json"
    f.write_bytes(b"tampered")
    monkeypatch.setitem(
        sys.modules["fetch_ctk_redistrib"].KNOWN_REDISTRIB_SHA256,
        "1.2.3",
        "f" * 64,
    )
    rc = main(["verify-metadata", "--cuda-version", "1.2.3", "--metadata-path", str(f)])
    assert rc == 1
