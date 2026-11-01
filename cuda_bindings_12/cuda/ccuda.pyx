# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cydriver cimport *
from cuda.bindings import cydriver
__pyx_capi__ = cydriver.__pyx_capi__
del cydriver
