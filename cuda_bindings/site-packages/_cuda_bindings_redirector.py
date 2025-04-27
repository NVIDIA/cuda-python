# Copyright 2025 NVIDIA Corporation. All rights reserved.

import sys
from types import ModuleType


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


# Important: We need to populate the cuda namespace module first, otherwise
# we'd lose access to any of its submodules. This is a cheap op because there
# is nothing under cuda.bindings.
import cuda.bindings
sys.modules['cuda'].__class__ = LazyCudaModule
