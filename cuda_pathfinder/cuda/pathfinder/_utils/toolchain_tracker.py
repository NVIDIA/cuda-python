# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Toolchain consistency tracking for CUDA artifact searches."""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Sequence


class ToolchainSource(Enum):
    """Source location of a CUDA artifact."""

    SITE_PACKAGES = auto()
    CONDA = auto()
    CUDA_HOME = auto()


@dataclass(frozen=True)
class SearchLocation:
    """Defines where and how to search for artifacts.

    Attributes:
        source: Which toolchain source this represents.
        base_dir_func: Function that returns the base directory to search, or None if unavailable.
        subdirs: Subdirectories to check under the base (e.g., ["bin"], ["Library/bin", "bin"]).
        filename_variants: Function that takes artifact name and returns possible filenames.
    """

    source: ToolchainSource
    base_dir_func: Callable[[], Optional[str]]
    subdirs: Sequence[str]
    filename_variants: Callable[[str], Sequence[str]]


@dataclass(frozen=True)
class ArtifactRecord:
    """Record of a found CUDA artifact."""

    name: str
    path: str
    source: ToolchainSource


class ToolchainMismatchError(RuntimeError):
    """Raised when artifacts from different sources are mixed."""

    def __init__(
        self,
        artifact_name: str,
        attempted_source: ToolchainSource,
        preferred_source: ToolchainSource,
        preferred_artifacts: list[ArtifactRecord],
    ):
        self.artifact_name = artifact_name
        self.attempted_source = attempted_source
        self.preferred_source = preferred_source
        self.preferred_artifacts = preferred_artifacts

        artifact_list = ", ".join(f"'{a.name}'" for a in preferred_artifacts)
        message = (
            f"Toolchain mismatch: '{artifact_name}' found in {attempted_source.name}, "
            f"but using {preferred_source.name} (previous: {artifact_list})"
        )
        super().__init__(message)


def search_location(location: SearchLocation, artifact_name: str) -> Optional[str]:
    """Search for an artifact in a specific location.

    Args:
        location: The search location configuration.
        artifact_name: Name of the artifact to find.

    Returns:
        Path to artifact if found, None otherwise.
    """
    base_dir = location.base_dir_func()
    if not base_dir:
        return None

    filenames = location.filename_variants(artifact_name)

    for subdir in location.subdirs:
        dir_path = os.path.join(base_dir, subdir)
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                return file_path

    return None


class SearchContext:
    """Tracks toolchain consistency across artifact searches.

    This context ensures all artifacts come from the same source to prevent
    version mismatches. The first artifact found establishes the preferred
    source for subsequent searches.
    """

    def __init__(self):
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._preferred_source: Optional[ToolchainSource] = None

    @property
    def preferred_source(self) -> Optional[ToolchainSource]:
        """The preferred toolchain source, or None if not yet determined."""
        return self._preferred_source

    def record(self, name: str, path: str, source: ToolchainSource) -> None:
        """Record an artifact and enforce consistency.

        Args:
            name: Artifact name.
            path: Absolute path where found.
            source: Source where found.

        Raises:
            ToolchainMismatchError: If source conflicts with preferred source.
        """
        if self._preferred_source is None:
            self._preferred_source = source
        elif source != self._preferred_source:
            raise ToolchainMismatchError(
                artifact_name=name,
                attempted_source=source,
                preferred_source=self._preferred_source,
                preferred_artifacts=list(self._artifacts.values()),
            )

        self._artifacts[name] = ArtifactRecord(name=name, path=path, source=source)

    def find(self, artifact_name: str, locations: Sequence[SearchLocation]) -> Optional[str]:
        """Search for artifact respecting toolchain consistency.

        Args:
            artifact_name: Name of artifact to find.
            locations: Search locations to try.

        Returns:
            Path to artifact, or None if not found.

        Raises:
            ToolchainMismatchError: If found in different source than preferred.
        """
        # Reorder to search preferred source first
        if self._preferred_source:
            ordered_locations = [loc for loc in locations if loc.source == self._preferred_source]
            ordered_locations += [loc for loc in locations if loc.source != self._preferred_source]
        else:
            ordered_locations = list(locations)

        # Try each location
        for location in ordered_locations:
            if path := search_location(location, artifact_name):
                self.record(artifact_name, path, location.source)
                return path

        return None


# Module-level default context
_default_context = SearchContext()


def get_default_context() -> SearchContext:
    """Get the default module-level search context."""
    return _default_context


def reset_default_context() -> None:
    """Reset the default context to a fresh state."""
    global _default_context
    _default_context = SearchContext()
