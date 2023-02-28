# Copyright 2021-2023 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import sys

def checkCmdLineFlag(stringRef):
    k = 0
    for i in sys.argv:
        if stringRef == i and k < len(sys.argv) - 1:
           return True
        k += 1
    return False

def getCmdLineArgumentInt(stringRef):
    k = 0
    for i in sys.argv:
        if stringRef == i and k < len(sys.argv) - 1:
           return sys.argv[k+1]
        k += 1
    return 0
