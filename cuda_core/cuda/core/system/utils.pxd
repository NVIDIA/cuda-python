# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cpython cimport array


cpdef list[int] unpack_bitmask(x: list[int] | array.array)
