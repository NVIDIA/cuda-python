# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
from urllib.error import HTTPError

import pytest

from ci.tools import verify_release

TAG = "v12.9.11"
VERSION = "12.9.11"
BINDINGS = "cuda_bindings-12.9.11-cp311-cp311-win_amd64.whl"
METAPACKAGE = "cuda_python-12.9.11-py3-none-any.whl"
DIGEST = hashlib.sha256(b"staged wheel").hexdigest()
ARCHIVE = f"cuda-python-{TAG}.tar.gz"
RELEASE_API = f"https://api.github.com/repos/NVIDIA/cuda-python/releases/tags/{TAG}"
DOWNLOAD = f"https://github.com/NVIDIA/cuda-python/releases/download/{TAG}"


def _pypi_url(component):
    return f"https://pypi.org/pypi/{component}/{VERSION}/json"


def _pypi_response(name, digest=DIGEST):
    return {
        "urls": [
            {"filename": name, "digests": {"sha256": digest}, "packagetype": "bdist_wheel", "yanked": False},
        ],
    }


@pytest.fixture
def public_server(monkeypatch):
    responses = {
        _pypi_url("cuda-bindings"): _pypi_response(BINDINGS),
        _pypi_url("cuda-python"): _pypi_response(METAPACKAGE),
        RELEASE_API: {
            "draft": False,
            "tag_name": TAG,
            "assets": [
                {"name": BINDINGS, "digest": f"sha256:{DIGEST}"},
                {"name": METAPACKAGE, "digest": f"sha256:{DIGEST}"},
                {"name": ARCHIVE},
                {"name": f"{ARCHIVE}.sha256sum"},
            ],
        },
        f"{DOWNLOAD}/{ARCHIVE}": b"archive content",
        f"{DOWNLOAD}/{ARCHIVE}.sha256sum": hashlib.sha256(b"archive content").hexdigest().encode(),
    }
    requested = []

    def urlopen(request, *, timeout):
        url = request.full_url
        requested.append(url)
        if url.startswith("https://nvidia.github.io/"):
            return io.BytesIO()
        payload = responses.get(url)
        if payload is None:
            raise HTTPError(url, 404, "Not found", {}, None)
        if isinstance(payload, list):
            payload = payload.pop(0)
        return io.BytesIO(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

    monkeypatch.setattr(verify_release, "urlopen", urlopen)
    monkeypatch.setattr(verify_release.time, "sleep", lambda _: None)
    return responses, requested


@pytest.mark.agent_authored(model="GPT-6")
def test_dry_run_does_not_probe_public_surfaces(public_server, tmp_path):
    (tmp_path / BINDINGS).write_bytes(b"staged wheel")
    manifest = verify_release.make_manifest(TAG, "cuda-bindings", "dry-run", tmp_path)

    assert manifest["wheels"] == {BINDINGS: DIGEST}
    _, errors = verify_release.verify_manifest(manifest)
    assert errors == []
    assert public_server[1] == []


@pytest.mark.agent_authored(model="GPT-6")
@pytest.mark.parametrize("digest", [DIGEST, "a" * 64])
def test_automatic_report_compares_selected_release_to_staged_wheels(public_server, digest):
    manifest = {"tag": TAG, "component": "cuda-bindings", "mode": "full-release", "wheels": {BINDINGS: DIGEST}}
    responses, requested = public_server
    responses[_pypi_url("cuda-bindings")] = _pypi_response(BINDINGS, digest)

    _, errors = verify_release.verify_manifest(manifest)
    if digest == DIGEST:
        assert errors == []
    else:
        assert errors == [f"PyPI cuda-bindings {VERSION}: SHA-256 mismatch for {BINDINGS}"]
    assert requested == [
        _pypi_url("cuda-bindings"),
        f"https://nvidia.github.io/cuda-python/cuda-bindings/{VERSION}/",
        f"https://nvidia.github.io/cuda-python/cuda-bindings/{VERSION}/release/{VERSION}-notes.html",
    ]


@pytest.mark.agent_authored(model="GPT-6")
def test_automatic_report_retries_missing_wheels_and_fails_if_still_missing(public_server):
    manifest = {"tag": TAG, "component": "cuda-bindings", "mode": "full-release", "wheels": {BINDINGS: DIGEST}}
    responses, requested = public_server
    url = _pypi_url("cuda-bindings")
    responses[url] = [{"urls": []}, _pypi_response(BINDINGS)]

    _, errors = verify_release.verify_manifest(manifest)
    assert errors == []
    assert requested.count(url) == 2

    responses[url] = {"urls": []}
    _, errors = verify_release.verify_manifest(manifest)
    assert len(errors) == 1
    assert f"missing non-yanked wheels: {BINDINGS}" in errors[0]


@pytest.mark.agent_authored(model="GPT-6")
def test_manual_report_checks_both_components_and_downloaded_archive(public_server):
    _, requested = public_server
    passed, errors = verify_release.verify_tag(TAG)
    assert errors == []
    assert _pypi_url("cuda-bindings") in requested
    assert _pypi_url("cuda-python") in requested
    assert f"{DOWNLOAD}/{ARCHIVE}" in requested
    assert f"{DOWNLOAD}/{ARCHIVE}.sha256sum" in requested
    assert f"GitHub release {TAG}: 2 wheels match PyPI" in passed


@pytest.mark.agent_authored(model="GPT-6")
def test_manual_report_fails_for_partial_publication_and_corrupt_archive(public_server):
    responses, _ = public_server
    del responses[_pypi_url("cuda-python")]
    responses[f"{DOWNLOAD}/{ARCHIVE}"] = b"wrong archive content"

    _, errors = verify_release.verify_tag(TAG)
    assert len(errors) == 2
    assert any("PyPI cuda-python" in error and "HTTP 404" in error for error in errors)
    assert any("does not match its published SHA-256 checksum" in error for error in errors)


@pytest.mark.agent_authored(model="GPT-6")
def test_pathfinder_post_release_uses_its_versioned_landing_page(public_server):
    filename = "cuda_pathfinder-1.9.1.post1-py3-none-any.whl"
    pypi_url = "https://pypi.org/pypi/cuda-pathfinder/1.9.1.post1/json"
    manifest = {
        "tag": "cuda-pathfinder-v1.9.1.post1",
        "component": "cuda-pathfinder",
        "mode": "full-release",
        "wheels": {filename: DIGEST},
    }
    responses, requested = public_server
    responses[pypi_url] = _pypi_response(filename)

    _, errors = verify_release.verify_manifest(manifest)
    assert errors == []
    assert requested == [pypi_url, "https://nvidia.github.io/cuda-python/cuda-pathfinder/1.9.1/"]
