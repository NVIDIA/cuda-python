# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys
from types import ModuleType


# Make sure 'cuda' is importable as a namespace package
import cuda


class LazyCudaModule(ModuleType):

    def __getattr__(self, name):
        if name == '__version__':
            import warnings
            warnings.warn(
                "accessing cuda.__version__ is deprecated, " "please switch to use cuda.bindings.__version__ instead",
                FutureWarning,
                stacklevel=2,
            )
            from cuda.bindings import __version__

            return __version__

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Patch in LazyCudaModule for `cuda`
sys.modules['cuda'].__class__ = LazyCudaModule
