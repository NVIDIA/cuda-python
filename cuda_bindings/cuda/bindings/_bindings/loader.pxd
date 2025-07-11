# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cdef extern from "loader.h":
    int getCUDALibraryPath(char *libPath, bint isBit64)
