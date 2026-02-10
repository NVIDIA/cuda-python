# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class Linker:
    cdef:
        object _mnff  # _LinkerMembersNeededForFinalize
        object _options  # LinkerOptions
        object __weakref__
