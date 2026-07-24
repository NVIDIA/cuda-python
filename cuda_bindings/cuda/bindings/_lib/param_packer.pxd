# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Include "param_packer.h" so its contents get compiled into every
# Cython extension module that depends on param_packer.pxd.
cdef extern from "param_packer.h":
    int feed(void* ptr, object o, object ct) except? -1
