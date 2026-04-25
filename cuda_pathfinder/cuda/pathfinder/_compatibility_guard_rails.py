# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib.metadata
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from cuda.pathfinder._binaries.find_nvidia_binary_utility import (
    find_nvidia_binary_utility as _find_nvidia_binary_utility,
)
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES_ALL
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)
from cuda.pathfinder._headers.find_nvidia_headers import (
    LocatedHeaderDir,
)
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as _locate_nvidia_header_directory,
)
from cuda.pathfinder._headers.header_descriptor import HEADER_DESCRIPTORS
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    LocatedBitcodeLib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    locate_bitcode_lib as _locate_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    LocatedStaticLib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    locate_static_lib as _locate_static_lib,
)
from cuda.pathfinder._utils.driver_info import (
    DriverCudaVersion,
    QueryDriverCudaVersionError,
    query_driver_cuda_version,
)
from cuda.pathfinder._utils.toolkit_info import ReadCudaHeaderVersionError, read_cuda_header_version

ItemKind: TypeAlias = str
PackagedWith: TypeAlias = str
ConstraintOperator: TypeAlias = str
ConstraintArg: TypeAlias = int | str | tuple[str, int] | None

_CTK_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")
_REQUIRES_DIST_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.-]+)\s*(?P<specifier_text>[^;]*)(?:\s*;|$)")
_VERSION_SPECIFIER_RE = re.compile(r"^\s*(?P<operator>==|<=|>=|<|>)\s*(?P<version>[0-9][A-Za-z0-9.+-]*?(?:\.\*)?)\s*$")

_STATIC_LIBS_PACKAGED_WITH: dict[str, PackagedWith] = {
    "cudadevrt": "ctk",
}
_BITCODE_LIBS_PACKAGED_WITH: dict[str, PackagedWith] = {
    "device": "ctk",
    "nvshmem_device": "other",
}
_BINARY_PACKAGED_WITH: dict[str, PackagedWith] = dict.fromkeys(SUPPORTED_BINARIES_ALL, "ctk")


class CompatibilityCheckError(RuntimeError):
    """Raised when compatibility checks reject a resolved item."""


class CompatibilityInsufficientMetadataError(CompatibilityCheckError):
    """Raised when v1 compatibility checks cannot reach a definitive answer."""


@dataclass(frozen=True, slots=True)
class CtkMetadata:
    ctk_version: CtkVersion
    ctk_root: str | None
    source: str


@dataclass(frozen=True, slots=True)
class CtkVersion:
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True, slots=True)
class ComparisonConstraint:
    operator: ConstraintOperator
    value: int

    def matches(self, candidate: int) -> bool:
        if self.operator == "==":
            return candidate == self.value
        if self.operator == "<":
            return candidate < self.value
        if self.operator == "<=":
            return candidate <= self.value
        if self.operator == ">":
            return candidate > self.value
        if self.operator == ">=":
            return candidate >= self.value
        raise AssertionError(f"Unsupported operator: {self.operator!r}")

    def __str__(self) -> str:
        return f"{self.operator}{self.value}"


@dataclass(frozen=True, slots=True)
class VersionSpecifier:
    operator: ConstraintOperator
    version: str


@dataclass(frozen=True, slots=True)
class ResolvedItem:
    name: str
    kind: ItemKind
    packaged_with: PackagedWith
    abs_path: str
    found_via: str | None
    ctk_root: str | None
    ctk_version: CtkVersion | None
    ctk_version_source: str | None

    def describe(self) -> str:
        found_via = "" if self.found_via is None else f" via {self.found_via}"
        return f"{self.kind} {self.name!r}{found_via} at {self.abs_path!r}"


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: str
    message: str

    def require_compatible(self) -> None:
        if self.status == "compatible":
            return
        if self.status == "insufficient_metadata":
            raise CompatibilityInsufficientMetadataError(self.message)
        raise CompatibilityCheckError(self.message)


def _coerce_constraint(name: str, raw_value: ConstraintArg) -> ComparisonConstraint | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, int):
        return ComparisonConstraint("==", raw_value)
    if isinstance(raw_value, tuple):
        if len(raw_value) != 2:
            raise ValueError(f"{name} tuple constraints must have exactly two elements.")
        operator, value = raw_value
        if operator not in ("==", "<", "<=", ">", ">="):
            raise ValueError(f"{name} has unsupported operator {operator!r}.")
        if not isinstance(value, int):
            raise ValueError(f"{name} constraint value must be an integer.")
        return ComparisonConstraint(operator, value)
    if isinstance(raw_value, str):
        match = re.fullmatch(r"\s*(==|<|<=|>|>=)?\s*(\d+)\s*", raw_value)
        if match is None:
            raise ValueError(f"{name} must be an int, a (operator, value) tuple, or a string like '>=12'.")
        operator = match.group(1) or "=="
        value = int(match.group(2))
        return ComparisonConstraint(operator, value)
    raise ValueError(f"{name} must be an int, a (operator, value) tuple, or a string like '>=12'.")


def _parse_ctk_version(cuda_version: str) -> CtkVersion | None:
    match = _CTK_VERSION_RE.match(cuda_version)
    if match is None:
        return None
    return CtkVersion(major=int(match.group("major")), minor=int(match.group("minor")))


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_name(dist: importlib.metadata.Distribution) -> str | None:
    # Work around mypy's typing of Distribution.metadata as PackageMetadata:
    # the runtime object behaves like a string mapping, but mypy does not
    # expose Mapping.get() on PackageMetadata.
    metadata = cast(Mapping[str, str], dist.metadata)
    return metadata.get("Name")


def _release_version_parts(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^\d+(?:\.\d+)*", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def _compare_release_versions(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> int:
    max_len = max(len(lhs), len(rhs))
    lhs_padded = lhs + (0,) * (max_len - len(lhs))
    rhs_padded = rhs + (0,) * (max_len - len(rhs))
    if lhs_padded < rhs_padded:
        return -1
    if lhs_padded > rhs_padded:
        return 1
    return 0


def _parse_version_specifiers(specifier_text: str) -> tuple[VersionSpecifier, ...]:
    stripped = specifier_text.strip()
    if not stripped:
        return ()
    parsed: list[VersionSpecifier] = []
    for raw_clause in stripped.split(","):
        match = _VERSION_SPECIFIER_RE.match(raw_clause)
        if match is None:
            return ()
        parsed.append(VersionSpecifier(operator=match.group("operator"), version=match.group("version")))
    return tuple(parsed)


def _version_satisfies_specifiers(version: str, specifiers: tuple[VersionSpecifier, ...]) -> bool:
    if not specifiers:
        return False
    for specifier in specifiers:
        if specifier.operator == "==":
            prefix = specifier.version.removesuffix(".*")
            if version == prefix or version.startswith(prefix + "."):
                continue
            return False
        candidate_parts = _release_version_parts(version)
        required_parts = _release_version_parts(specifier.version)
        if candidate_parts is None or required_parts is None:
            return False
        comparison = _compare_release_versions(candidate_parts, required_parts)
        if specifier.operator == "<" and comparison < 0:
            continue
        if specifier.operator == "<=" and comparison <= 0:
            continue
        if specifier.operator == ">" and comparison > 0:
            continue
        if specifier.operator == ">=" and comparison >= 0:
            continue
        return False
    return True


@functools.cache
def _owned_distribution_candidates(abs_path: str) -> tuple[tuple[str, str], ...]:
    normalized_abs_path = os.path.normpath(os.path.abspath(abs_path))
    matches: set[tuple[str, str]] = set()
    for dist in importlib.metadata.distributions():
        dist_name = _distribution_name(dist)
        if not dist_name:
            continue
        for file in dist.files or ():
            candidate_abs_path = os.path.normpath(os.path.abspath(str(dist.locate_file(file))))
            if candidate_abs_path == normalized_abs_path:
                matches.add((dist_name, dist.version))
    return tuple(sorted(matches))


@functools.cache
def _cuda_toolkit_requirement_maps() -> tuple[
    tuple[str, CtkVersion, dict[str, tuple[tuple[VersionSpecifier, ...], ...]]], ...
]:
    results: list[tuple[str, CtkVersion, dict[str, tuple[tuple[VersionSpecifier, ...], ...]]]] = []
    for dist in importlib.metadata.distributions():
        dist_name = _distribution_name(dist)
        if _normalize_distribution_name(dist_name or "") != "cuda-toolkit":
            continue
        ctk_version = _parse_ctk_version(dist.version)
        if ctk_version is None:
            continue
        requirement_map: dict[str, set[tuple[VersionSpecifier, ...]]] = {}
        for requirement in dist.requires or ():
            match = _REQUIRES_DIST_RE.match(requirement)
            if match is None:
                continue
            req_name = _normalize_distribution_name(match.group("name"))
            parsed_specifiers = _parse_version_specifiers(match.group("specifier_text"))
            if not parsed_specifiers:
                continue
            requirement_map.setdefault(req_name, set()).add(parsed_specifiers)
        results.append(
            (
                dist.version,
                ctk_version,
                {
                    name: tuple(
                        sorted(
                            specifier_sets,
                            key=lambda specifiers: tuple(
                                (specifier.operator, specifier.version) for specifier in specifiers
                            ),
                        )
                    )
                    for name, specifier_sets in requirement_map.items()
                },
            )
        )
    return tuple(results)


def _wheel_metadata_for_abs_path(abs_path: str) -> CtkMetadata | None:
    matched_versions: dict[CtkVersion, str] = {}
    for owner_name, owner_version in _owned_distribution_candidates(abs_path):
        normalized_owner_name = _normalize_distribution_name(owner_name)
        for toolkit_dist_version, ctk_version, requirement_map in _cuda_toolkit_requirement_maps():
            requirement_specifier_sets = requirement_map.get(normalized_owner_name, ())
            if not any(
                _version_satisfies_specifiers(owner_version, specifiers) for specifiers in requirement_specifier_sets
            ):
                continue
            matched_versions[ctk_version] = (
                f"wheel metadata via {owner_name}=={owner_version} pinned by cuda-toolkit=={toolkit_dist_version}"
            )
    if len(matched_versions) != 1:
        return None
    [(ctk_version, source)] = matched_versions.items()
    return CtkMetadata(ctk_version=ctk_version, ctk_root=None, source=source)


def _normalized_ctk_root_for_cuda_header(cuda_header_path: Path) -> Path:
    ctk_root = cuda_header_path.parent.parent
    if ctk_root.parent.name == "targets":
        return ctk_root.parent.parent
    return ctk_root


@functools.cache
def _cuda_header_metadata_for_ctk_root_candidate(ctk_root_candidate: str) -> CtkMetadata | None:
    candidate_path = Path(ctk_root_candidate)
    header_paths: list[Path] = []

    direct_header = candidate_path / "include" / "cuda.h"
    if direct_header.is_file():
        header_paths.append(direct_header)

    targets_dir = candidate_path / "targets"
    if targets_dir.is_dir():
        header_paths.extend(sorted(path for path in targets_dir.glob("*/include/cuda.h") if path.is_file()))

    matches: list[tuple[CtkVersion, Path, Path]] = []
    for cuda_header_path in header_paths:
        try:
            version = read_cuda_header_version(str(cuda_header_path))
        except ReadCudaHeaderVersionError:
            continue
        matches.append(
            (
                CtkVersion(major=version.major, minor=version.minor),
                _normalized_ctk_root_for_cuda_header(cuda_header_path),
                cuda_header_path,
            )
        )

    if not matches:
        return None

    ctk_version, ctk_root, source_path = matches[0]
    if any(other_version != ctk_version for other_version, _other_root, _other_source in matches[1:]):
        return None

    return CtkMetadata(
        ctk_version=ctk_version,
        ctk_root=str(ctk_root),
        source=f"cuda.h at {source_path}",
    )


def _ctk_metadata_for_abs_path(abs_path: str) -> CtkMetadata | None:
    current = Path(abs_path)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        ctk_metadata = _cuda_header_metadata_for_ctk_root_candidate(str(candidate))
        if ctk_metadata is not None:
            return ctk_metadata
    return _wheel_metadata_for_abs_path(abs_path)


def _resolve_item(
    *,
    name: str,
    kind: ItemKind,
    packaged_with: PackagedWith,
    abs_path: str,
    found_via: str | None,
) -> ResolvedItem:
    ctk_metadata = _ctk_metadata_for_abs_path(abs_path)
    return ResolvedItem(
        name=name,
        kind=kind,
        packaged_with=packaged_with,
        abs_path=abs_path,
        found_via=found_via,
        ctk_root=None if ctk_metadata is None else ctk_metadata.ctk_root,
        ctk_version=None if ctk_metadata is None else ctk_metadata.ctk_version,
        ctk_version_source=None if ctk_metadata is None else ctk_metadata.source,
    )


def _resolve_dynamic_lib_item(libname: str, loaded: LoadedDL) -> ResolvedItem:
    if loaded.abs_path is None:
        raise CompatibilityInsufficientMetadataError(
            f"Could not determine an absolute path for dynamic library {libname!r}."
        )
    desc = LIB_DESCRIPTORS[libname]
    return _resolve_item(
        name=libname,
        kind="dynamic-lib",
        packaged_with=desc.packaged_with,
        abs_path=loaded.abs_path,
        found_via=loaded.found_via,
    )


def _resolve_header_item(libname: str, located: LocatedHeaderDir) -> ResolvedItem:
    if located.abs_path is None:
        raise CompatibilityInsufficientMetadataError(
            f"Could not determine an absolute path for header directory {libname!r}."
        )
    desc = HEADER_DESCRIPTORS[libname]
    metadata_abs_path = os.path.join(located.abs_path, desc.header_basename)
    return _resolve_item(
        name=libname,
        kind="header-dir",
        packaged_with=desc.packaged_with,
        abs_path=metadata_abs_path,
        found_via=located.found_via,
    )


def _resolve_static_lib_item(located: LocatedStaticLib) -> ResolvedItem:
    packaged_with = _STATIC_LIBS_PACKAGED_WITH[located.name]
    return _resolve_item(
        name=located.name,
        kind="static-lib",
        packaged_with=packaged_with,
        abs_path=located.abs_path,
        found_via=located.found_via,
    )


def _resolve_bitcode_lib_item(located: LocatedBitcodeLib) -> ResolvedItem:
    packaged_with = _BITCODE_LIBS_PACKAGED_WITH[located.name]
    return _resolve_item(
        name=located.name,
        kind="bitcode-lib",
        packaged_with=packaged_with,
        abs_path=located.abs_path,
        found_via=located.found_via,
    )


def _resolve_binary_item(utility_name: str, abs_path: str) -> ResolvedItem:
    packaged_with = _BINARY_PACKAGED_WITH[utility_name]
    return _resolve_item(
        name=utility_name,
        kind="binary",
        packaged_with=packaged_with,
        abs_path=abs_path,
        found_via=None,
    )


def compatibility_check(
    driver_cuda_version: DriverCudaVersion, item1: ResolvedItem, item2: ResolvedItem
) -> CompatibilityResult:
    for item in (item1, item2):
        if item.packaged_with != "ctk":
            return CompatibilityResult(
                status="insufficient_metadata",
                message=(
                    "v1 compatibility checks only give definitive answers for "
                    f"packaged_with='ctk' items. {item.describe()} is packaged_with={item.packaged_with!r}."
                ),
            )
        if item.ctk_version is None or item.ctk_version_source is None:
            return CompatibilityResult(
                status="insufficient_metadata",
                message=(
                    "v1 compatibility checks require either an enclosing CUDA Toolkit root "
                    "with cuda.h or wheel metadata that can be traced to an installed "
                    f"cuda-toolkit distribution. Could not determine the CTK version for {item.describe()}."
                ),
            )

    assert item1.ctk_version is not None
    assert item2.ctk_version is not None

    if item1.ctk_version != item2.ctk_version:
        return CompatibilityResult(
            status="incompatible",
            message=(
                f"{item1.describe()} resolves to CTK {item1.ctk_version}, while "
                f"{item2.describe()} resolves to CTK {item2.ctk_version}. "
                "v1 requires an exact CTK major.minor match."
            ),
        )

    if driver_cuda_version.major < item1.ctk_version.major:
        return CompatibilityResult(
            status="incompatible",
            message=(
                f"Driver version {driver_cuda_version.encoded} only supports CUDA major version {driver_cuda_version.major}, "
                f"but {item1.describe()} requires CTK {item1.ctk_version}. "
                "v1 requires driver_major >= ctk_major."
            ),
        )

    return CompatibilityResult(
        status="compatible",
        message=(
            f"{item1.describe()} and {item2.describe()} both resolve to CTK {item1.ctk_version}, "
            f"and driver version {driver_cuda_version.encoded} satisfies the v1 driver guard rail."
        ),
    )


class CompatibilityGuardRails:
    """Resolve CUDA artifacts while enforcing minimal v1 compatibility guard rails."""

    def __init__(
        self,
        *,
        ctk_major: ConstraintArg = None,
        ctk_minor: ConstraintArg = None,
        driver_cuda_version: DriverCudaVersion | None = None,
    ) -> None:
        self._ctk_major_constraint = _coerce_constraint("ctk_major", ctk_major)
        self._ctk_minor_constraint = _coerce_constraint("ctk_minor", ctk_minor)
        self._configured_driver_cuda_version = driver_cuda_version
        self._driver_cuda_version = driver_cuda_version
        self._resolved_items: list[ResolvedItem] = []

    def _get_driver_cuda_version(self) -> DriverCudaVersion:
        if self._driver_cuda_version is None:
            try:
                self._driver_cuda_version = query_driver_cuda_version()
            except QueryDriverCudaVersionError as exc:
                raise CompatibilityCheckError(
                    "Failed to query the CUDA driver version needed for compatibility checks."
                ) from exc
        return self._driver_cuda_version

    def _enforce_supported_packaging(self, item: ResolvedItem) -> None:
        if item.packaged_with == "ctk":
            return
        raise CompatibilityInsufficientMetadataError(
            "v1 compatibility checks only give definitive answers for "
            f"packaged_with='ctk' items, plus compatibility-neutral driver libraries. "
            f"{item.describe()} is packaged_with={item.packaged_with!r}."
        )

    def _enforce_ctk_metadata(self, item: ResolvedItem) -> None:
        if item.ctk_version is not None and item.ctk_version_source is not None:
            return
        raise CompatibilityInsufficientMetadataError(
            "v1 compatibility checks require either an enclosing CUDA Toolkit root "
            "with cuda.h or wheel metadata that can be traced to an installed "
            f"cuda-toolkit distribution. Could not determine the CTK version for {item.describe()}."
        )

    def _enforce_constraints(self, item: ResolvedItem) -> None:
        assert item.ctk_version is not None
        if self._ctk_major_constraint is not None and not self._ctk_major_constraint.matches(item.ctk_version.major):
            raise CompatibilityCheckError(
                f"{item.describe()} resolves to CTK {item.ctk_version}, which does not satisfy "
                f"ctk_major{self._ctk_major_constraint}."
            )
        if self._ctk_minor_constraint is not None and not self._ctk_minor_constraint.matches(item.ctk_version.minor):
            raise CompatibilityCheckError(
                f"{item.describe()} resolves to CTK {item.ctk_version}, which does not satisfy "
                f"ctk_minor{self._ctk_minor_constraint}."
            )

    def _anchor_item(self) -> ResolvedItem | None:
        for item in self._resolved_items:
            if item.packaged_with == "ctk":
                return item
        return None

    def _remember(self, item: ResolvedItem) -> None:
        if item not in self._resolved_items:
            self._resolved_items.append(item)

    def _reset_for_testing(self) -> None:
        self._driver_cuda_version = self._configured_driver_cuda_version
        self._resolved_items.clear()

    def _register_and_check(self, item: ResolvedItem) -> None:
        # Driver libraries come from the installed display driver rather than a
        # CUDA Toolkit line, so they do not need CTK metadata and must not lock
        # the process-wide CTK anchor.
        if item.packaged_with == "driver":
            self._remember(item)
            return
        self._enforce_supported_packaging(item)
        self._enforce_ctk_metadata(item)
        self._enforce_constraints(item)
        anchor = self._anchor_item()
        if anchor is None:
            anchor = item
        compatibility_check(self._get_driver_cuda_version(), anchor, item).require_compatible()
        self._remember(item)

    def load_nvidia_dynamic_lib(self, libname: str) -> LoadedDL:
        """Load a CUDA dynamic library and reject v1-incompatible resolutions."""
        loaded = _load_nvidia_dynamic_lib(libname)
        self._register_and_check(_resolve_dynamic_lib_item(libname, loaded))
        return loaded

    def locate_nvidia_header_directory(self, libname: str) -> LocatedHeaderDir | None:
        """Locate a CUDA header directory and reject v1-incompatible resolutions."""
        located = _locate_nvidia_header_directory(libname)
        if located is None:
            return None
        self._register_and_check(_resolve_header_item(libname, located))
        return located

    def find_nvidia_header_directory(self, libname: str) -> str | None:
        """Locate a CUDA header directory and return only the path string."""
        located = self.locate_nvidia_header_directory(libname)
        return None if located is None else located.abs_path

    def locate_static_lib(self, name: str) -> LocatedStaticLib:
        """Locate a CUDA static library and reject v1-incompatible resolutions."""
        located = _locate_static_lib(name)
        self._register_and_check(_resolve_static_lib_item(located))
        return located

    def find_static_lib(self, name: str) -> str:
        """Locate a CUDA static library and return only the path string."""
        abs_path = self.locate_static_lib(name).abs_path
        assert isinstance(abs_path, str)
        return abs_path

    def locate_bitcode_lib(self, name: str) -> LocatedBitcodeLib:
        """Locate a CUDA bitcode library and reject v1-incompatible resolutions."""
        located = _locate_bitcode_lib(name)
        self._register_and_check(_resolve_bitcode_lib_item(located))
        return located

    def find_bitcode_lib(self, name: str) -> str:
        """Locate a CUDA bitcode library and return only the path string."""
        abs_path = self.locate_bitcode_lib(name).abs_path
        assert isinstance(abs_path, str)
        return abs_path

    def find_nvidia_binary_utility(self, utility_name: str) -> str | None:
        """Locate a CUDA binary utility and reject v1-incompatible resolutions."""
        abs_path = _find_nvidia_binary_utility(utility_name)
        if abs_path is None:
            return None
        self._register_and_check(_resolve_binary_item(utility_name, abs_path))
        assert isinstance(abs_path, str)
        return abs_path
