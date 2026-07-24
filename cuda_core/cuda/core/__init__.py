# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._version import __version__


def _import_versioned_module() -> None:
    import importlib

    from cuda import bindings

    cuda_major = bindings.__version__.split(".")[0]
    if cuda_major not in ("12", "13"):
        raise ImportError("cuda.bindings 12.x or 13.x must be installed")

    subdir = f"cu{cuda_major}"
    try:
        versioned_mod = importlib.import_module(f".{subdir}", __package__)
        # Import all symbols from the module
        globals().update(versioned_mod.__dict__)
    except ImportError:
        # This is not a wheel build, but a conda or local build, do nothing
        pass


_import_versioned_module()
del _import_versioned_module


def _patch_rlcompleter_for_cython_properties() -> None:
    # TODO: This can be removed when Python 3.13 is our minimum-supported version:
    #   https://github.com/python/cpython/pull/149577

    # Cython @property on cdef class compiles to a C-level getset_descriptor,
    # which rlcompleter's narrow isinstance(..., property) check misses; the
    # fallback getattr() then invokes the descriptor and any non-AttributeError
    # it raises kills tab completion. Extend that isinstance check to also
    # match getset_descriptor / member_descriptor. Only installed in
    # interactive mode so library users running scripts see no global
    # rlcompleter side effect.
    import os

    if int(os.environ.get("CUDA_CORE_DONT_FIX_TAB_COMPLETION", "0")):
        # Explicit opt-out for users who don't want the global rlcompleter
        # side effect, even in an interactive session.
        return

    import rlcompleter
    from types import GetSetDescriptorType, MemberDescriptorType

    # This works by overriding the `property` built-in with a custom subclass of
    # property, but only in the rlcompleter module.  This subclass overrides the
    # `__instancecheck__` method to also return True for getset_descriptor and
    # member_descriptor types, which are what Cython uses for properties on cdef
    # classes.
    class _PatchedPropMeta(type):
        def __instancecheck__(cls, inst: object) -> bool:
            return isinstance(inst, (property, GetSetDescriptorType, MemberDescriptorType))

    class _PatchedProperty(metaclass=_PatchedPropMeta):
        pass

    rlcompleter.property = _PatchedProperty  # type: ignore[attr-defined]


_patch_rlcompleter_for_cython_properties()
del _patch_rlcompleter_for_cython_properties


from cuda.core import checkpoint, system, utils
from cuda.core._context import *
from cuda.core._context import __all__ as _context_all
from cuda.core._device import *
from cuda.core._device import __all__ as _device_all
from cuda.core._device_resources import *
from cuda.core._device_resources import __all__ as _device_resources_all
from cuda.core._event import *
from cuda.core._event import __all__ as _event_all
from cuda.core._graphics import *
from cuda.core._graphics import __all__ as _graphics_all
from cuda.core._host import *
from cuda.core._host import __all__ as _host_all
from cuda.core._launch_config import *
from cuda.core._launch_config import __all__ as _launch_config_all
from cuda.core._launcher import *
from cuda.core._launcher import __all__ as _launcher_all
from cuda.core._linker import *
from cuda.core._linker import __all__ as _linker_all
from cuda.core._memory import *
from cuda.core._memory import __all__ as _memory_all
from cuda.core._module import *
from cuda.core._module import __all__ as _module_all
from cuda.core._program import *
from cuda.core._program import __all__ as _program_all
from cuda.core._stream import *
from cuda.core._stream import __all__ as _stream_all
from cuda.core._tensor_map import *
from cuda.core._tensor_map import __all__ as _tensor_map_all

__all__ = [
    *_context_all,
    *_device_all,
    *_device_resources_all,
    *_event_all,
    *_graphics_all,
    *_host_all,
    *_launch_config_all,
    *_launcher_all,
    *_linker_all,
    *_memory_all,
    *_module_all,
    *_program_all,
    *_stream_all,
    *_tensor_map_all,
]

# isort: split
# Texture/surface types live under the cuda.core.texture namespace (not the
# flat cuda.core namespace); import the subpackage so it is available as
# `cuda.core.texture` after `import cuda.core`.
# Must come after the cuda.core._* extension imports above: loading graph
# earlier interacts badly with the merged-wheel __path__ rewrite and leaves
# Graph/GraphBuilder/GraphCompleteOptions/GraphDebugPrintOptions missing from
# cuda.core.graph.
import cuda.core.graph
import cuda.core.texture
