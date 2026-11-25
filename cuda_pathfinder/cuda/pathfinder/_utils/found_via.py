# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``found_via`` labels: where a search found what it was looking for.

These strings are public. They appear in ``LoadedDL.found_via``,
``LocatedHeaderDir.found_via``, ``LocatedStaticLib.found_via`` and
``LocatedBitcodeLib.found_via``, and the Linux loader branches on two of them.
Spell them here and nowhere else.

A product root found through a product-specific environment variable is
labeled with that variable's name (for example ``"CUDNN_PATH"``), so it has no
constant here.
"""

from typing import Final

FOUND_VIA_SITE_PACKAGES: Final = "site-packages"
FOUND_VIA_CONDA: Final = "conda"
FOUND_VIA_SYSTEM_SEARCH: Final = "system-search"
#: Covers both the ``CUDA_PATH`` and the ``CUDA_HOME`` environment variable.
FOUND_VIA_CUDA_PATH: Final = "CUDA_PATH"
FOUND_VIA_PROGRAM_FILES: Final = "ProgramFiles"
FOUND_VIA_SYSTEM_CTK_ROOT: Final = "system-ctk-root"
FOUND_VIA_SUPPORTED_INSTALL_DIR: Final = "supported_install_dir"
FOUND_VIA_WAS_ALREADY_LOADED_FROM_ELSEWHERE: Final = "was-already-loaded-from-elsewhere"
