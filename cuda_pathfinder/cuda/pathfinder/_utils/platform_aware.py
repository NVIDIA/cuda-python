# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

IS_WINDOWS = sys.platform == "win32"
_WINDOWS_PYTHON_ARCH: str | None

if IS_WINDOWS:
    from cuda.pathfinder._utils.windows_arch import windows_python_arch

    _WINDOWS_PYTHON_ARCH = windows_python_arch()
else:
    _WINDOWS_PYTHON_ARCH = None

# These describe the Python process ABI, not the Windows host architecture.
IS_WINDOWS_X64 = _WINDOWS_PYTHON_ARCH == "x64"
IS_WINDOWS_ARM64 = _WINDOWS_PYTHON_ARCH == "arm64"


def quote_for_shell(s: str) -> str:
    if IS_WINDOWS:
        # This is a relatively heavy import; keep pathfinder lean if possible.
        from subprocess import list2cmdline

        return list2cmdline([s])
    else:
        import shlex

        return shlex.quote(s)
