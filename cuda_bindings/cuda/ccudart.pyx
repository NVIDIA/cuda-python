# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings.cyruntime cimport *
from cuda.bindings import cyruntime
__pyx_capi__ = cyruntime.__pyx_capi__
del cyruntime
