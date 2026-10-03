# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys


def check_cmd_line_flag(string_ref):
    """Return whether ``string_ref`` was passed on the command line.

    ``sys.argv[0]`` is the program name and is never considered a flag.
    """
    return string_ref in sys.argv[1:]


def get_cmd_line_argument_int(string_ref):
    """Return the integer that follows ``string_ref`` on the command line.

    Returns 0 if ``string_ref`` was not passed, or if nothing follows it.
    """
    args = sys.argv[1:]
    for idx, arg in enumerate(args):
        if arg == string_ref and idx + 1 < len(args):
            return int(args[idx + 1])
    return 0
