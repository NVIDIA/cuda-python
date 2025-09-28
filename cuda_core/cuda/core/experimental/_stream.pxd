# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# TODO: how about cuda.bindings < 12.6.2?
from cuda.bindings cimport cydriver


cdef cydriver.CUstream _try_to_get_stream_ptr(obj: IsStreamT) except*
