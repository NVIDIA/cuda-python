# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

IS_WINDOWS = sys.platform == "win32"


def quote_for_shell(s: str) -> str:
    if IS_WINDOWS:
        # This is a relatively heavy import; keep pathfinder lean if possible.
        from subprocess import list2cmdline

        return list2cmdline([s])
    else:
        import shlex

        return shlex.quote(s)
