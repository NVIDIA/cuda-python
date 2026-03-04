# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys


def check_cmd_line_flag(string_ref):
    return any(string_ref == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def get_cmd_line_argument_int(string_ref):
    for i, k in enumerate(sys.argv):
        if string_ref == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0
