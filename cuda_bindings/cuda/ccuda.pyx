# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings.cydriver cimport *
from cuda.bindings import cydriver
__pyx_capi__ = cydriver.__pyx_capi__
del cydriver
