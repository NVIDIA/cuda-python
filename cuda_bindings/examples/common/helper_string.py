# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys


def checkCmdLineFlag(stringRef):
    return any(stringRef == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def getCmdLineArgumentInt(stringRef):
    for i, k in enumerate(sys.argv):
        if stringRef == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0
