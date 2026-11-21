# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record staged wheel hashes and verify the public results of a CUDA Python release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import TypeVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from .bindings_config import parse_prefixed_version
from .check_tagged_release_notes import COMPONENTS

_PYPI = "https://pypi.org/pypi"
_DOCS = "https://nvidia.github.io/cuda-python"
_REPO = "NVIDIA/cuda-python"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases/tags"
_ASSET_DOWNLOAD = f"https://github.com/{_REPO}/releases/download"
_RETRY_DELAYS = (0, 10, 20, 30)
_MODES = ("dry-run", "full-release")
_T = TypeVar("_T")


class VerificationError(Exception):
    """The public release or verification metadata failed a required check."""


class PublicationPendingError(VerificationError):
    """A registry response may still be waiting for publication to propagate."""


def _tag_components(tag: str) -> tuple[Version, tuple[str, ...]]:
    for _, prefix in COMPONENTS.values():
        version = parse_prefixed_version(tag, prefix)
        if version is not None:
            return version, tuple(name for name, (_, candidate) in COMPONENTS.items() if candidate == prefix)
    raise ValueError(f"Invalid CUDA Python release tag: {tag!r}")


def _wheel_hashes(wheels: dict[str, str], component: str, version: Version) -> dict[str, str]:
    if not isinstance(wheels, dict) or not wheels:
        raise ValueError(f"No wheels for {component} {version}")
    for name, digest in wheels.items():
        distribution, wheel_version, _, _ = parse_wheel_filename(name)
        if distribution != component or wheel_version != version:
            raise ValueError(f"{name!r} is not a {component} {version} wheel")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name!r} has no valid SHA-256 digest")
    return wheels


def make_manifest(tag: str, component: str, mode: str, wheel_dir: Path) -> dict:
    version, components = _tag_components(tag)
    if component not in components or mode not in _MODES:
        raise ValueError(f"Invalid release component or mode for {tag!r}: {component!r}, {mode!r}")
    wheels = {}
    for wheel in sorted(wheel_dir.glob("*.whl")):
        with wheel.open("rb") as stream:
            wheels[wheel.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"tag": tag, "component": component, "mode": mode, "wheels": _wheel_hashes(wheels, component, version)}


def _read(url: str, read: Callable[[HTTPResponse], _T], *, github_token: str = "") -> _T:
    # All requests use fixed HTTPS endpoints and normalized version or wheel identifiers.
    request = Request(url, headers={"User-Agent": "cuda-python-release-verification"})  # noqa: S310
    if github_token:
        request.add_unredirected_header("Authorization", f"Bearer {github_token}")
    detail = ""
    for delay in _RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            with urlopen(request, timeout=15) as response:  # noqa: S310
                return read(response)
        except HTTPError as exc:
            detail = f"HTTP {exc.code}"
            if exc.code not in (404, 429) and exc.code < 500:
                break
        except (OSError, HTTPException, PublicationPendingError) as exc:
            detail = str(exc)
    raise VerificationError(f"{url}: {detail}")


def _compare_wheels(actual: dict[str, str], expected: dict[str, str]) -> list[str]:
    return [
        *(f"missing wheel {name}" for name in sorted(expected.keys() - actual.keys())),
        *(f"unexpected wheel {name}" for name in sorted(actual.keys() - expected.keys())),
        *(
            f"SHA-256 mismatch for {name}"
            for name in sorted(expected.keys() & actual.keys())
            if actual[name] != expected[name]
        ),
    ]


def _read_pypi(component: str, version: Version, expected: dict[str, str] | None) -> dict[str, str]:
    def parse(response):
        files = json.load(response)["urls"]
        wheels = {
            entry["filename"]: entry["digests"]["sha256"]
            for entry in files
            if entry["packagetype"] == "bdist_wheel" and not entry["yanked"]
        }
        missing = sorted(expected.keys() - wheels.keys()) if expected is not None else []
        if missing:
            raise PublicationPendingError("missing non-yanked wheels: " + ", ".join(missing))
        return _wheel_hashes(wheels, component, version)

    return _read(f"{_PYPI}/{component}/{version}/json", parse)


def _docs_urls(component: str, version: Version) -> list[str]:
    # Core uses the exact version directory. The other builds reuse the
    # corresponding base directory for post releases.
    docs_version = str(version)
    if version.is_postrelease and component != "cuda-core":
        docs_version = docs_version.split(".post", 1)[0]
    root = _DOCS if component == "cuda-python" else f"{_DOCS}/{component}"
    landing = f"{root}/{docs_version}/"
    # The Pathfinder notes check allows empty placeholders.
    if component == "cuda-pathfinder" or version.is_postrelease:
        return [landing]
    return [landing, f"{landing}release/{version}-notes.html"]


def _github_assets(tag: str) -> dict:
    release = _read(f"{_RELEASES_API}/{tag}", json.load, github_token=os.environ.get("GH_TOKEN", ""))
    if release["draft"] or release["tag_name"] != tag:
        raise VerificationError(f"GitHub does not show a public release for {tag}")
    return {asset["name"]: asset for asset in release["assets"]}


def _check_archive(tag: str, assets: dict) -> str:
    archive = f"cuda-python-{tag}.tar.gz"
    checksum = f"{archive}.sha256sum"
    missing = {archive, checksum} - assets.keys()
    if missing:
        raise VerificationError("GitHub release is missing: " + ", ".join(sorted(missing)))

    download = f"{_ASSET_DOWNLOAD}/{tag}"
    expected = _read(f"{download}/{checksum}", lambda response: response.read(1024).decode("ascii").strip())
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise VerificationError(f"GitHub release asset {checksum} does not contain a SHA-256 digest")
    actual = _read(f"{download}/{archive}", lambda response: hashlib.file_digest(response, "sha256").hexdigest())
    if actual != expected:
        raise VerificationError(f"GitHub release asset {archive} does not match its published SHA-256 checksum")
    return archive


def _audit(
    tag: str, version: Version, components: tuple[str, ...], staged_wheels: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    passed = []
    errors = []
    public_wheels = {}
    for component in components:
        try:
            wheels = _read_pypi(component, version, staged_wheels)
            differences = _compare_wheels(wheels, staged_wheels) if staged_wheels is not None else []
            if differences:
                errors.extend(f"PyPI {component} {version}: {difference}" for difference in differences)
            else:
                public_wheels[component] = wheels
                passed.append(f"PyPI {component} {version}: {len(wheels)} non-yanked wheels")
        except (VerificationError, ValueError, KeyError) as exc:
            errors.append(f"PyPI {component} {version}: {exc}")

        for url in _docs_urls(component, version):
            try:
                _read(url, lambda _: None)
                passed.append(f"Documentation: {url}")
            except VerificationError as exc:
                errors.append(f"Documentation: {exc}")

    if staged_wheels is not None:
        passed.append("Use a manual audit after the GitHub release is public to check its source archive and assets.")
        return passed, errors

    try:
        assets = _github_assets(tag)
    except (VerificationError, ValueError, KeyError) as exc:
        errors.append(f"GitHub release {tag}: {exc}")
        return passed, errors

    if len(public_wheels) == len(components):
        pypi = {name: digest for wheels in public_wheels.values() for name, digest in wheels.items()}
        github = {
            name: (asset.get("digest") or "").removeprefix("sha256:")
            for name, asset in assets.items()
            if name.endswith(".whl")
        }
        differences = _compare_wheels(github, pypi)
        if differences:
            errors.extend(f"GitHub release {tag}: {difference}" for difference in differences)
        else:
            passed.append(f"GitHub release {tag}: {len(github)} wheels match PyPI")

    try:
        archive = _check_archive(tag, assets)
        passed.append(f"GitHub release {tag}: {archive} matches its published SHA-256 checksum")
    except (VerificationError, ValueError) as exc:
        errors.append(f"GitHub release {tag}: {exc}")
    return passed, errors


def verify_manifest(manifest: dict) -> tuple[list[str], list[str]]:
    if not isinstance(manifest, dict):
        raise ValueError("Expected a JSON object for release verification metadata")
    tag, component, mode, wheels = (manifest[key] for key in ("tag", "component", "mode", "wheels"))
    if not isinstance(tag, str):
        raise ValueError("Release tag must be a string")
    version, components = _tag_components(tag)
    if component not in components or mode not in _MODES:
        raise ValueError(f"Invalid release component or mode for {tag!r}: {component!r}, {mode!r}")
    wheels = _wheel_hashes(wheels, component, version)
    if mode == "dry-run":
        return [f"Dry run for {component} {tag}. Public verification skipped."], []
    return _audit(tag, version, (component,), staged_wheels=wheels)


def verify_tag(tag: str) -> tuple[list[str], list[str]]:
    version, components = _tag_components(tag)
    return _audit(tag, version, components)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    manifest = subcommands.add_parser("make-manifest", help="record the staged release metadata as JSON")
    manifest.add_argument("--tag", required=True)
    manifest.add_argument("--component", required=True, choices=COMPONENTS)
    manifest.add_argument("--mode", required=True, choices=_MODES)
    manifest.add_argument("--wheel-dir", required=True, type=Path)
    verify = subcommands.add_parser("verify", help="check the public results of a release")
    target = verify.add_mutually_exclusive_group(required=True)
    target.add_argument("--manifest", type=Path)
    target.add_argument("--tag")
    args = parser.parse_args(argv)

    try:
        if args.command == "make-manifest":
            print(json.dumps(make_manifest(args.tag, args.component, args.mode, args.wheel_dir)))
            return 0
        if args.manifest is not None:
            passed, errors = verify_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        else:
            passed, errors = verify_tag(args.tag)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Invalid release verification input: {exc}", file=sys.stderr)
        return 1

    for line in passed:
        print(line)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write("## Public release verification\n\n")
            stream.writelines(f"- {line}\n" for line in passed)
            stream.writelines(f"- **Error** {error}\n" for error in errors)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
