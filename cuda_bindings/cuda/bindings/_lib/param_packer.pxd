# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cdef extern from "param_packer.h":
    int feed(void* ptr, object o, object ct)
