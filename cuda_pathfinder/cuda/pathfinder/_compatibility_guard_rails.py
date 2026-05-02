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

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from cuda.pathfinder._binaries.find_nvidia_binary_utility import (
    find_nvidia_binary_utility as _find_nvidia_binary_utility,
)
from cuda.pathfinder._binaries.supported_nvidia_binaries import (
    SUPPORTED_BINARIES_ALL,
    SUPPORTED_BINARIES_CTK_COMPANION_TAGS,
)
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
    SUPPORTED_BITCODE_LIBS_CTK_COMPANION_TAGS,
    LocatedBitcodeLib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    locate_bitcode_lib as _locate_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    SUPPORTED_STATIC_LIBS_CTK_COMPANION_TAGS,
    LocatedStaticLib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    locate_static_lib as _locate_static_lib,
)
from cuda.pathfinder._utils.driver_info import (
    DriverCudaVersion,
    DriverReleaseVersion,
    QueryDriverCudaVersionError,
    QueryDriverReleaseVersionError,
    query_driver_cuda_version,
    query_driver_release_version,
)
from cuda.pathfinder._utils.toolkit_info import ReadCudaHeaderVersionError, read_cuda_header_version

ItemKind: TypeAlias = str
PackagedWith: TypeAlias = str
CtkVersionConstraintArg: TypeAlias = str | SpecifierSet | None
PairwiseItemRelationKind: TypeAlias = str
DriverCompatibilityKind: TypeAlias = str
PipelineArtifactKind: TypeAlias = str

_CTK_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")
_CTK_VERSION_CONSTRAINT_ERROR = (
    "ctk_version must be None, a non-empty PEP 440 specifier string like '>=13.2,<14', "
    "or a packaging.specifiers.SpecifierSet."
)
_PAIRWISE_ITEM_RELATION_NONE = "none"
_PAIRWISE_ITEM_RELATION_EXACT_CTK_MATCH_REQUIRED = "exact-ctk-match-required"
_DRIVER_COMPATIBILITY_BACKWARD = "backward-compatibility"
_DRIVER_COMPATIBILITY_MINOR_VERSION = "minor-version-compatibility"
_PIPELINE_ARTIFACT_KIND_LTOIR = "ltoir"
_PIPELINE_ARTIFACT_KIND_PTX = "ptx"
_PIPELINE_ARTIFACT_KIND_ELF = "elf"
_PIPELINE_ARTIFACT_KIND_CUBIN = "cubin"
_PIPELINE_ARTIFACT_KINDS = (
    _PIPELINE_ARTIFACT_KIND_LTOIR,
    _PIPELINE_ARTIFACT_KIND_PTX,
    _PIPELINE_ARTIFACT_KIND_ELF,
    _PIPELINE_ARTIFACT_KIND_CUBIN,
)
_MIN_DRIVER_BRANCH_FOR_MINOR_VERSION_COMPATIBILITY_BY_CTK_MAJOR = {
    11: 450,
    12: 525,
    13: 580,
}


@dataclass(frozen=True, slots=True)
class PairwiseItemRelation:
    kind: PairwiseItemRelationKind
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DriverCompatibilityDecision:
    kind: DriverCompatibilityKind
    detail: str


@dataclass(frozen=True, slots=True)
class DeclaredDynamicLibPipeline:
    producer_libname: str
    consumer_libname: str
    artifact_kind: PipelineArtifactKind


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


class DriverCtkCompatibilityError(CompatibilityCheckError):
    """Raised when driver-vs-CTK policy rejects a resolved item."""


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

    def as_pep440_version(self) -> Version:
        return Version(str(self))


@dataclass(frozen=True, slots=True)
class CtkVersionConstraint:
    specifier: SpecifierSet
    text: str

    def matches(self, candidate: CtkVersion) -> bool:
        return bool(self.specifier.contains(candidate.as_pep440_version(), prereleases=True))

    def __str__(self) -> str:
        return self.text


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
    dynamic_link_component: str | None
    ctk_companion_tags: tuple[str, ...]

    def describe(self) -> str:
        found_via = "" if self.found_via is None else f" via {self.found_via}"
        return f"{self.kind} {self.name!r}{found_via} at {self.abs_path!r}"


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: str
    message: str
    error_type: type[CompatibilityCheckError] = CompatibilityCheckError

    def require_compatible(self) -> None:
        if self.status == "compatible":
            return
        if self.status == "insufficient_metadata":
            raise CompatibilityInsufficientMetadataError(self.message)
        raise self.error_type(self.message)


def _parse_ctk_version(cuda_version: str) -> CtkVersion | None:
    match = _CTK_VERSION_RE.match(cuda_version)
    if match is None:
        return None
    return CtkVersion(major=int(match.group("major")), minor=int(match.group("minor")))


def _coerce_ctk_version_constraint(raw_value: CtkVersionConstraintArg) -> CtkVersionConstraint | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, SpecifierSet):
        return CtkVersionConstraint(specifier=raw_value, text=str(raw_value))
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise ValueError(_CTK_VERSION_CONSTRAINT_ERROR)
        try:
            specifier = SpecifierSet(stripped)
        except InvalidSpecifier as exc:
            raise ValueError(_CTK_VERSION_CONSTRAINT_ERROR) from exc
        return CtkVersionConstraint(specifier=specifier, text=stripped)
    raise ValueError(_CTK_VERSION_CONSTRAINT_ERROR)


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_name(dist: importlib.metadata.Distribution) -> str | None:
    # Work around mypy's typing of Distribution.metadata as PackageMetadata:
    # the runtime object behaves like a string mapping, but mypy does not
    # expose Mapping.get() on PackageMetadata.
    metadata = cast(Mapping[str, str], dist.metadata)
    return metadata.get("Name")


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
def _cuda_toolkit_requirement_maps() -> tuple[tuple[str, CtkVersion, dict[str, tuple[SpecifierSet, ...]]], ...]:
    results: list[tuple[str, CtkVersion, dict[str, tuple[SpecifierSet, ...]]]] = []
    for dist in importlib.metadata.distributions():
        dist_name = _distribution_name(dist)
        if _normalize_distribution_name(dist_name or "") != "cuda-toolkit":
            continue
        ctk_version = _parse_ctk_version(dist.version)
        if ctk_version is None:
            continue
        requirement_map: dict[str, set[str]] = {}
        for requirement_text in dist.requires or ():
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement:
                continue
            specifier_text = str(requirement.specifier)
            if not specifier_text:
                continue
            req_name = _normalize_distribution_name(requirement.name)
            requirement_map.setdefault(req_name, set()).add(specifier_text)
        results.append(
            (
                dist.version,
                ctk_version,
                {
                    name: tuple(SpecifierSet(specifier_text) for specifier_text in sorted(specifier_set_texts))
                    for name, specifier_set_texts in requirement_map.items()
                },
            )
        )
    return tuple(results)


def _wheel_metadata_for_abs_path(abs_path: str) -> CtkMetadata | None:
    matched_versions: dict[CtkVersion, str] = {}
    for owner_name, owner_version in _owned_distribution_candidates(abs_path):
        try:
            owner_parsed_version = Version(owner_version)
        except InvalidVersion:
            continue
        normalized_owner_name = _normalize_distribution_name(owner_name)
        for toolkit_dist_version, ctk_version, requirement_map in _cuda_toolkit_requirement_maps():
            requirement_specifier_sets = requirement_map.get(normalized_owner_name, ())
            if not any(
                specifier_set.contains(owner_parsed_version, prereleases=True)
                for specifier_set in requirement_specifier_sets
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
    dynamic_link_component: str | None = None,
    ctk_companion_tags: tuple[str, ...] = (),
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
        dynamic_link_component=dynamic_link_component,
        ctk_companion_tags=ctk_companion_tags,
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
        dynamic_link_component=desc.dynamic_link_component,
        ctk_companion_tags=desc.ctk_companion_tags,
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
        ctk_companion_tags=desc.ctk_companion_tags,
    )


def _resolve_static_lib_item(located: LocatedStaticLib) -> ResolvedItem:
    packaged_with = _STATIC_LIBS_PACKAGED_WITH[located.name]
    return _resolve_item(
        name=located.name,
        kind="static-lib",
        packaged_with=packaged_with,
        abs_path=located.abs_path,
        found_via=located.found_via,
        ctk_companion_tags=SUPPORTED_STATIC_LIBS_CTK_COMPANION_TAGS.get(located.name, ()),
    )


def _resolve_bitcode_lib_item(located: LocatedBitcodeLib) -> ResolvedItem:
    packaged_with = _BITCODE_LIBS_PACKAGED_WITH[located.name]
    return _resolve_item(
        name=located.name,
        kind="bitcode-lib",
        packaged_with=packaged_with,
        abs_path=located.abs_path,
        found_via=located.found_via,
        ctk_companion_tags=SUPPORTED_BITCODE_LIBS_CTK_COMPANION_TAGS.get(located.name, ()),
    )


def _resolve_binary_item(utility_name: str, abs_path: str) -> ResolvedItem:
    packaged_with = _BINARY_PACKAGED_WITH[utility_name]
    return _resolve_item(
        name=utility_name,
        kind="binary",
        packaged_with=packaged_with,
        abs_path=abs_path,
        found_via=None,
        ctk_companion_tags=SUPPORTED_BINARIES_CTK_COMPANION_TAGS.get(utility_name, ()),
    )


def _unsupported_packaging_message(
    item: ResolvedItem, *, allow_compatibility_neutral_driver_libraries: bool = False
) -> str:
    message = "v1 compatibility checks only give definitive answers for packaged_with='ctk' items"
    if allow_compatibility_neutral_driver_libraries:
        message += ", plus compatibility-neutral driver libraries"
    return f"{message}. {item.describe()} is packaged_with={item.packaged_with!r}."


def _missing_ctk_metadata_message(item: ResolvedItem) -> str:
    return (
        "v1 compatibility checks require either an enclosing CUDA Toolkit root "
        "with cuda.h or wheel metadata that can be traced to an installed "
        f"cuda-toolkit distribution. Could not determine the CTK version for {item.describe()}."
    )


def _ctk_constraint_failure_message(item: ResolvedItem, constraint: CtkVersionConstraint) -> str:
    assert item.ctk_version is not None
    return f"{item.describe()} resolves to CTK {item.ctk_version}, which does not satisfy ctk_version{constraint}."


def _driver_backward_compatibility_detail(driver_cuda_version: DriverCudaVersion, item: ResolvedItem) -> str:
    assert item.ctk_version is not None
    return (
        f"the driver satisfies backward compatibility because cuDriverGetVersion() reports "
        f"CUDA {driver_cuda_version.major}.{driver_cuda_version.minor}, which is not older than CTK {item.ctk_version}"
    )


def _driver_minor_version_compatibility_detail(
    driver_cuda_version: DriverCudaVersion,
    driver_release_version: DriverReleaseVersion,
    item: ResolvedItem,
    *,
    required_branch: int,
) -> str:
    assert item.ctk_version is not None
    return (
        "the driver satisfies NVIDIA's same-major minor-version compatibility because "
        f"cuDriverGetVersion() reports older CUDA {driver_cuda_version.major}.{driver_cuda_version.minor}, "
        f"but display-driver release {driver_release_version.text} meets the published CUDA "
        f"{item.ctk_version.major}.x minimum branch >= {required_branch}"
    )


def _ctk_pair_mismatch_message(
    item1: ResolvedItem,
    item2: ResolvedItem,
    relation: PairwiseItemRelation,
) -> str:
    assert item1.ctk_version is not None
    assert item2.ctk_version is not None
    assert relation.reason is not None
    requirement_reason = relation.reason[:1].upper() + relation.reason[1:]
    return (
        f"{item1.describe()} resolves to CTK {item1.ctk_version}, while "
        f"{item2.describe()} resolves to CTK {item2.ctk_version}. "
        f"{requirement_reason}, so v1 requires an exact CTK major.minor match."
    )


def _driver_major_mismatch_message(driver_cuda_version: DriverCudaVersion, item: ResolvedItem) -> str:
    assert item.ctk_version is not None
    return (
        f"Driver version {driver_cuda_version.encoded} only supports CUDA major version {driver_cuda_version.major}, "
        f"but {item.describe()} requires CTK {item.ctk_version}. "
        "v1 requires driver_major >= ctk_major."
    )


def _driver_cuda_version_too_old_message(driver_cuda_version: DriverCudaVersion, item: ResolvedItem) -> str:
    assert item.ctk_version is not None
    return (
        f"cuDriverGetVersion() reports CUDA {driver_cuda_version.major}.{driver_cuda_version.minor}, "
        f"but {item.describe()} requires CTK {item.ctk_version}. "
        "NVIDIA's published minor-version compatibility starts with CUDA 11, so v1 requires "
        "the driver CUDA version to be at least the CTK version for older CTK majors."
    )


def _missing_driver_release_version_message(driver_cuda_version: DriverCudaVersion, item: ResolvedItem) -> str:
    assert item.ctk_version is not None
    return (
        f"cuDriverGetVersion() reports older CUDA {driver_cuda_version.major}.{driver_cuda_version.minor} for "
        f"{item.describe()}, which requires CTK {item.ctk_version}. Determining whether NVIDIA's same-major "
        "minor-version compatibility applies requires the display-driver release version (for example "
        "'535.54.03' or branch '535')."
    )


def _driver_release_branch_too_old_message(
    driver_cuda_version: DriverCudaVersion,
    driver_release_version: DriverReleaseVersion,
    item: ResolvedItem,
    *,
    required_branch: int,
) -> str:
    assert item.ctk_version is not None
    return (
        f"cuDriverGetVersion() reports older CUDA {driver_cuda_version.major}.{driver_cuda_version.minor}, "
        f"and display-driver release {driver_release_version.text} (branch {driver_release_version.branch}) "
        f"is below NVIDIA's published CUDA {item.ctk_version.major}.x minimum branch >= {required_branch} "
        f"for {item.describe()}."
    )


def _declared_dynamic_lib_pipeline_description(pipeline: DeclaredDynamicLibPipeline) -> str:
    return (
        f"declared dynamic-lib pipeline {pipeline.producer_libname!r} -> {pipeline.consumer_libname!r} "
        f"for artifact kind {pipeline.artifact_kind!r}"
    )


def _nvjitlink_ltoir_major_mismatch_message(
    producer_item: ResolvedItem,
    consumer_item: ResolvedItem,
    pipeline: DeclaredDynamicLibPipeline,
) -> str:
    assert producer_item.ctk_version is not None
    assert consumer_item.ctk_version is not None
    return (
        f"{_declared_dynamic_lib_pipeline_description(pipeline)} uses producer {producer_item.describe()} at "
        f"CTK {producer_item.ctk_version} and consumer {consumer_item.describe()} at CTK {consumer_item.ctk_version}. "
        "For LTOIR inputs, NVIDIA documents nvJitLink compatibility only within a major release family, "
        "so v1 requires the producer and consumer CTK majors to match."
    )


def _nvjitlink_ltoir_consumer_too_old_message(
    producer_item: ResolvedItem,
    consumer_item: ResolvedItem,
    pipeline: DeclaredDynamicLibPipeline,
) -> str:
    assert producer_item.ctk_version is not None
    assert consumer_item.ctk_version is not None
    return (
        f"{_declared_dynamic_lib_pipeline_description(pipeline)} uses producer {producer_item.describe()} at "
        f"CTK {producer_item.ctk_version} and consumer {consumer_item.describe()} at CTK {consumer_item.ctk_version}. "
        "For LTOIR inputs, NVIDIA documents that nvJitLink must be >= the producer version, "
        "so v1 rejects an older nvJitLink consumer."
    )


def _nvvm_pipeline_conservative_message(
    producer_item: ResolvedItem,
    consumer_item: ResolvedItem,
    pipeline: DeclaredDynamicLibPipeline,
) -> str:
    assert producer_item.ctk_version is not None
    assert consumer_item.ctk_version is not None
    return (
        f"{_declared_dynamic_lib_pipeline_description(pipeline)} involves {producer_item.describe()} at "
        f"CTK {producer_item.ctk_version} and {consumer_item.describe()} at CTK {consumer_item.ctk_version}. "
        "v1 remains conservative for explicit nvvm pipeline contexts until NVVM IR version and dialect are modeled, "
        "so it requires an exact CTK major.minor match."
    )


def _compatible_pair_message(
    driver_decision: DriverCompatibilityDecision,
    item1: ResolvedItem,
    item2: ResolvedItem,
    relation: PairwiseItemRelation,
) -> str:
    assert item1.ctk_version is not None
    assert item2.ctk_version is not None
    if relation.kind == _PAIRWISE_ITEM_RELATION_NONE:
        return (
            f"{item1.describe()} resolves to CTK {item1.ctk_version}, "
            f"{item2.describe()} resolves to CTK {item2.ctk_version}, "
            "and v1 does not require exact CTK lockstep for this pair. "
            f"Separately, {driver_decision.detail}."
        )
    assert relation.reason is not None
    return (
        f"{item1.describe()} and {item2.describe()} both resolve to CTK {item1.ctk_version}. "
        f"{relation.reason[:1].upper() + relation.reason[1:]}. Separately, {driver_decision.detail}."
    )


def _supported_packaging_result(item: ResolvedItem) -> CompatibilityResult | None:
    if item.packaged_with == "ctk":
        return None
    return CompatibilityResult(status="insufficient_metadata", message=_unsupported_packaging_message(item))


def _ctk_metadata_result(item: ResolvedItem) -> CompatibilityResult | None:
    if item.ctk_version is not None and item.ctk_version_source is not None:
        return None
    return CompatibilityResult(status="insufficient_metadata", message=_missing_ctk_metadata_message(item))


def _shared_ctk_companion_tags(item1: ResolvedItem, item2: ResolvedItem) -> tuple[str, ...]:
    return tuple(sorted(set(item1.ctk_companion_tags).intersection(item2.ctk_companion_tags)))


def _classify_pairwise_item_relation(item1: ResolvedItem, item2: ResolvedItem) -> PairwiseItemRelation:
    if item1.packaged_with == "driver" or item2.packaged_with == "driver":
        return PairwiseItemRelation(_PAIRWISE_ITEM_RELATION_NONE)
    if item1.dynamic_link_component is not None and item1.dynamic_link_component == item2.dynamic_link_component:
        return PairwiseItemRelation(
            _PAIRWISE_ITEM_RELATION_EXACT_CTK_MATCH_REQUIRED,
            reason=f"they are in the same authored dynamic-link component {item1.dynamic_link_component!r}",
        )
    shared_companion_tags = _shared_ctk_companion_tags(item1, item2)
    if shared_companion_tags:
        if len(shared_companion_tags) == 1:
            tag_description = repr(shared_companion_tags[0])
            reason = f"they share the authored companion tag {tag_description}"
        else:
            tags_description = ", ".join(repr(tag) for tag in shared_companion_tags)
            reason = f"they share the authored companion tags {tags_description}"
        return PairwiseItemRelation(_PAIRWISE_ITEM_RELATION_EXACT_CTK_MATCH_REQUIRED, reason=reason)
    return PairwiseItemRelation(_PAIRWISE_ITEM_RELATION_NONE)


def _ctk_coherence_result(
    item1: ResolvedItem,
    item2: ResolvedItem,
    relation: PairwiseItemRelation,
) -> CompatibilityResult | None:
    assert item1.ctk_version is not None
    assert item2.ctk_version is not None
    if item1.ctk_version == item2.ctk_version:
        return None
    return CompatibilityResult(status="incompatible", message=_ctk_pair_mismatch_message(item1, item2, relation))


def _pipeline_compatibility_result(_item1: ResolvedItem, _item2: ResolvedItem) -> CompatibilityResult | None:
    # Generic pairwise policy stays artifact-coherence-only. Milestone 6 adds
    # explicit pipeline-aware rules via declared dynamic-lib pipeline contexts,
    # because producer/consumer direction and artifact kind are not inferable
    # from bare item pairs alone.
    return None


def _pairwise_policy_result(
    item1: ResolvedItem,
    item2: ResolvedItem,
    relation: PairwiseItemRelation | None = None,
) -> CompatibilityResult | None:
    if relation is None:
        relation = _classify_pairwise_item_relation(item1, item2)
    if relation.kind == _PAIRWISE_ITEM_RELATION_NONE:
        return None
    if relation.kind == _PAIRWISE_ITEM_RELATION_EXACT_CTK_MATCH_REQUIRED:
        result = _ctk_coherence_result(item1, item2, relation)
        if result is not None:
            return result
        return _pipeline_compatibility_result(item1, item2)
    raise AssertionError(f"Unhandled pairwise item relation: {relation.kind!r}")


def _dynamic_lib_pipeline_items(
    item1: ResolvedItem,
    item2: ResolvedItem,
    pipeline: DeclaredDynamicLibPipeline,
) -> tuple[ResolvedItem, ResolvedItem] | None:
    if item1.kind != "dynamic-lib" or item2.kind != "dynamic-lib":
        return None
    if item1.name == pipeline.producer_libname and item2.name == pipeline.consumer_libname:
        return item1, item2
    if item2.name == pipeline.producer_libname and item1.name == pipeline.consumer_libname:
        return item2, item1
    return None


def _declared_dynamic_lib_pipeline_result(
    producer_item: ResolvedItem,
    consumer_item: ResolvedItem,
    pipeline: DeclaredDynamicLibPipeline,
) -> CompatibilityResult | None:
    assert producer_item.ctk_version is not None
    assert consumer_item.ctk_version is not None
    if "nvvm" in (producer_item.name, consumer_item.name):
        if producer_item.ctk_version == consumer_item.ctk_version:
            return None
        return CompatibilityResult(
            status="incompatible",
            message=_nvvm_pipeline_conservative_message(producer_item, consumer_item, pipeline),
        )
    if producer_item.name == "nvrtc" and consumer_item.name == "nvJitLink":
        if pipeline.artifact_kind in (
            _PIPELINE_ARTIFACT_KIND_PTX,
            _PIPELINE_ARTIFACT_KIND_ELF,
            _PIPELINE_ARTIFACT_KIND_CUBIN,
        ):
            # NVIDIA documents broader compatibility for PTX/ELF/CUBIN inputs than for LTOIR.
            return None
        assert pipeline.artifact_kind == _PIPELINE_ARTIFACT_KIND_LTOIR
        if producer_item.ctk_version.major != consumer_item.ctk_version.major:
            return CompatibilityResult(
                status="incompatible",
                message=_nvjitlink_ltoir_major_mismatch_message(producer_item, consumer_item, pipeline),
            )
        if (
            consumer_item.ctk_version.major,
            consumer_item.ctk_version.minor,
        ) < (
            producer_item.ctk_version.major,
            producer_item.ctk_version.minor,
        ):
            return CompatibilityResult(
                status="incompatible",
                message=_nvjitlink_ltoir_consumer_too_old_message(producer_item, consumer_item, pipeline),
            )
    return None


def _driver_cuda_version_supports_ctk_by_backward_compatibility(
    driver_cuda_version: DriverCudaVersion,
    ctk_version: CtkVersion,
) -> bool:
    return (driver_cuda_version.major, driver_cuda_version.minor) >= (ctk_version.major, ctk_version.minor)


def _driver_compatibility_outcome(
    driver_cuda_version: DriverCudaVersion,
    item: ResolvedItem,
    *,
    driver_release_version: DriverReleaseVersion | None = None,
) -> DriverCompatibilityDecision | CompatibilityResult:
    assert item.ctk_version is not None
    if _driver_cuda_version_supports_ctk_by_backward_compatibility(driver_cuda_version, item.ctk_version):
        return DriverCompatibilityDecision(
            kind=_DRIVER_COMPATIBILITY_BACKWARD,
            detail=_driver_backward_compatibility_detail(driver_cuda_version, item),
        )
    if driver_cuda_version.major != item.ctk_version.major:
        return CompatibilityResult(
            status="incompatible",
            message=_driver_major_mismatch_message(driver_cuda_version, item),
            error_type=DriverCtkCompatibilityError,
        )
    required_branch = _MIN_DRIVER_BRANCH_FOR_MINOR_VERSION_COMPATIBILITY_BY_CTK_MAJOR.get(item.ctk_version.major)
    if required_branch is None:
        return CompatibilityResult(
            status="incompatible",
            message=_driver_cuda_version_too_old_message(driver_cuda_version, item),
            error_type=DriverCtkCompatibilityError,
        )
    if driver_release_version is None:
        return CompatibilityResult(
            status="insufficient_metadata",
            message=_missing_driver_release_version_message(driver_cuda_version, item),
        )
    if driver_release_version.branch >= required_branch:
        return DriverCompatibilityDecision(
            kind=_DRIVER_COMPATIBILITY_MINOR_VERSION,
            detail=_driver_minor_version_compatibility_detail(
                driver_cuda_version,
                driver_release_version,
                item,
                required_branch=required_branch,
            ),
        )
    return CompatibilityResult(
        status="incompatible",
        message=_driver_release_branch_too_old_message(
            driver_cuda_version,
            driver_release_version,
            item,
            required_branch=required_branch,
        ),
        error_type=DriverCtkCompatibilityError,
    )


def compatibility_check(
    driver_cuda_version: DriverCudaVersion,
    item1: ResolvedItem,
    item2: ResolvedItem,
    *,
    driver_release_version: DriverReleaseVersion | None = None,
) -> CompatibilityResult:
    for item in (item1, item2):
        result = _supported_packaging_result(item)
        if result is not None:
            return result
        result = _ctk_metadata_result(item)
        if result is not None:
            return result

    relation = _classify_pairwise_item_relation(item1, item2)
    result = _pairwise_policy_result(item1, item2, relation)
    if result is not None:
        return result

    driver_outcome = _driver_compatibility_outcome(
        driver_cuda_version,
        item1,
        driver_release_version=driver_release_version,
    )
    if isinstance(driver_outcome, CompatibilityResult):
        return driver_outcome

    return CompatibilityResult(
        status="compatible",
        message=_compatible_pair_message(driver_outcome, item1, item2, relation),
    )


class CompatibilityGuardRails:
    """Resolve CUDA artifacts while enforcing minimal v1 compatibility guard rails."""

    def __init__(
        self,
        *,
        ctk_version: CtkVersionConstraintArg = None,
        driver_cuda_version: DriverCudaVersion | None = None,
        driver_release_version: DriverReleaseVersion | None = None,
    ) -> None:
        self._ctk_version_constraint = _coerce_ctk_version_constraint(ctk_version)
        self._configured_driver_cuda_version = driver_cuda_version
        self._driver_cuda_version = driver_cuda_version
        self._configured_driver_release_version = driver_release_version
        self._driver_release_version = driver_release_version
        self._resolved_items: list[ResolvedItem] = []
        self._declared_dynamic_lib_pipelines: set[DeclaredDynamicLibPipeline] = set()
        self._checked_dynamic_lib_pipelines: set[DeclaredDynamicLibPipeline] = set()

    def _get_driver_cuda_version(self) -> DriverCudaVersion:
        if self._driver_cuda_version is None:
            try:
                self._driver_cuda_version = query_driver_cuda_version()
            except QueryDriverCudaVersionError as exc:
                raise CompatibilityCheckError(
                    "Failed to query the CUDA driver version needed for compatibility checks."
                ) from exc
        return self._driver_cuda_version

    def _get_driver_release_version(self) -> DriverReleaseVersion:
        if self._driver_release_version is None:
            try:
                self._driver_release_version = query_driver_release_version()
            except QueryDriverReleaseVersionError as exc:
                raise CompatibilityInsufficientMetadataError(
                    "Failed to query the display-driver release version needed for compatibility checks."
                ) from exc
        return self._driver_release_version

    def _enforce_supported_packaging(self, item: ResolvedItem) -> None:
        if item.packaged_with == "ctk":
            return
        raise CompatibilityInsufficientMetadataError(
            _unsupported_packaging_message(item, allow_compatibility_neutral_driver_libraries=True)
        )

    def _enforce_ctk_metadata(self, item: ResolvedItem) -> None:
        result = _ctk_metadata_result(item)
        if result is None:
            return
        result.require_compatible()

    def _enforce_constraints(self, item: ResolvedItem) -> None:
        assert item.ctk_version is not None
        if self._ctk_version_constraint is not None and not self._ctk_version_constraint.matches(item.ctk_version):
            raise CompatibilityCheckError(_ctk_constraint_failure_message(item, self._ctk_version_constraint))

    def _enforce_driver_compatibility(self, item: ResolvedItem) -> None:
        driver_cuda_version = self._get_driver_cuda_version()
        assert item.ctk_version is not None
        driver_release_version = (
            None
            if _driver_cuda_version_supports_ctk_by_backward_compatibility(driver_cuda_version, item.ctk_version)
            else self._get_driver_release_version()
        )
        outcome = _driver_compatibility_outcome(
            driver_cuda_version,
            item,
            driver_release_version=driver_release_version,
        )
        if isinstance(outcome, CompatibilityResult):
            outcome.require_compatible()

    def _enforce_pairwise_compatibility(self, prior_item: ResolvedItem, item: ResolvedItem) -> None:
        result = _pairwise_policy_result(prior_item, item)
        if result is not None:
            result.require_compatible()
        self._enforce_declared_dynamic_lib_pipelines_for_pair(prior_item, item)

    def _remembered_item(self, *, kind: ItemKind, name: str) -> ResolvedItem | None:
        for item in reversed(self._resolved_items):
            if item.kind == kind and item.name == name:
                return item
        return None

    def _enforce_declared_dynamic_lib_pipeline_if_ready(self, pipeline: DeclaredDynamicLibPipeline) -> None:
        if pipeline in self._checked_dynamic_lib_pipelines:
            return
        producer_item = self._remembered_item(kind="dynamic-lib", name=pipeline.producer_libname)
        if producer_item is None:
            return
        consumer_item = self._remembered_item(kind="dynamic-lib", name=pipeline.consumer_libname)
        if consumer_item is None:
            return
        result = _declared_dynamic_lib_pipeline_result(producer_item, consumer_item, pipeline)
        if result is not None:
            result.require_compatible()
        self._checked_dynamic_lib_pipelines.add(pipeline)

    def _enforce_declared_dynamic_lib_pipelines_for_pair(self, item1: ResolvedItem, item2: ResolvedItem) -> None:
        for pipeline in self._declared_dynamic_lib_pipelines:
            if _dynamic_lib_pipeline_items(item1, item2, pipeline) is None:
                continue
            self._enforce_declared_dynamic_lib_pipeline_if_ready(pipeline)

    def _enforce_declared_dynamic_lib_pipelines_for_item(self, item: ResolvedItem) -> None:
        if item.kind != "dynamic-lib":
            return
        for pipeline in self._declared_dynamic_lib_pipelines:
            if item.name not in (pipeline.producer_libname, pipeline.consumer_libname):
                continue
            self._enforce_declared_dynamic_lib_pipeline_if_ready(pipeline)

    def _remember(self, item: ResolvedItem) -> None:
        if item not in self._resolved_items:
            self._resolved_items.append(item)
        self._enforce_declared_dynamic_lib_pipelines_for_item(item)

    def _declare_dynamic_lib_pipeline(
        self,
        *,
        producer_libname: str,
        consumer_libname: str,
        artifact_kind: PipelineArtifactKind,
    ) -> None:
        if producer_libname not in LIB_DESCRIPTORS:
            raise ValueError(f"Unknown dynamic library producer: {producer_libname!r}")
        if consumer_libname not in LIB_DESCRIPTORS:
            raise ValueError(f"Unknown dynamic library consumer: {consumer_libname!r}")
        if artifact_kind not in _PIPELINE_ARTIFACT_KINDS:
            allowed_values = ", ".join(repr(kind) for kind in _PIPELINE_ARTIFACT_KINDS)
            raise ValueError(f"Invalid pipeline artifact kind {artifact_kind!r}. Allowed values: {allowed_values}.")
        pipeline = DeclaredDynamicLibPipeline(
            producer_libname=producer_libname,
            consumer_libname=consumer_libname,
            artifact_kind=artifact_kind,
        )
        self._declared_dynamic_lib_pipelines.add(pipeline)
        self._enforce_declared_dynamic_lib_pipeline_if_ready(pipeline)

    def _reset_for_testing(self) -> None:
        self._driver_cuda_version = self._configured_driver_cuda_version
        self._driver_release_version = self._configured_driver_release_version
        self._resolved_items.clear()
        self._declared_dynamic_lib_pipelines.clear()
        self._checked_dynamic_lib_pipelines.clear()

    def _register_and_check(self, item: ResolvedItem) -> None:
        # Driver libraries come from the installed display driver rather than a
        # CUDA Toolkit line, so they do not need CTK metadata and must not
        # create CTK coherence relations by themselves.
        if item.packaged_with == "driver":
            self._remember(item)
            return
        self._enforce_supported_packaging(item)
        self._enforce_ctk_metadata(item)
        self._enforce_constraints(item)
        for prior_item in self._resolved_items:
            self._enforce_pairwise_compatibility(prior_item, item)
        self._enforce_driver_compatibility(item)
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
