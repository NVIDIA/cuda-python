# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sysconfig


class UnsupportedWindowsArchError(RuntimeError):
    """Raised when Python reports an unsupported Windows architecture."""

    def __init__(self, platform_tag: str) -> None:
        self.platform_tag = platform_tag
        super().__init__(
            f"Unsupported Windows Python platform tag: {platform_tag!r}; expected 'win-amd64' or 'win-arm64'"
        )


def windows_python_arch() -> str:
    """Return the current Windows Python interpreter architecture."""
    raw_platform_tag = sysconfig.get_platform()
    platform_tag = raw_platform_tag.lower().replace("_", "-")

    if platform_tag == "win-arm64":
        return "arm64"

    if platform_tag == "win-amd64":
        return "x64"

    raise UnsupportedWindowsArchError(raw_platform_tag)
