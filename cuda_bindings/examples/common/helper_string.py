# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys


def checkCmdLineFlag(stringRef):
    return any(stringRef == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def getCmdLineArgumentInt(stringRef):
    for i, k in enumerate(sys.argv):
        if stringRef == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0
