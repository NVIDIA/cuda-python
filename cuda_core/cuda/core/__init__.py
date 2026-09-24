# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._version import __version__


def _import_versioned_module() -> None:
    """Check that the installed cuda-bindings is supported, then select the build for it.

    The published wheel carries one build per CUDA major series, as the
    subpackages ``cuda.core.cu12`` and ``cuda.core.cu13``. A conda or local
    build carries one build at the top level. Each build records the CUDA
    header that it compiled against and its cuda-bindings floor in
    ``_build_info``, which build_hooks.py generates. The installed cuda-bindings
    must be of the build's major, at least as new as the floor, and generated
    from a ``cuda.h`` at least as new as the build's. See
    ``_bindings_floor.check_installed_bindings``. If it is not, import fails
    here with an actionable message instead of later with a missing C function
    or a silently disabled feature.
    """
    import importlib

    try:
        from cuda import bindings
    except ModuleNotFoundError as exc:
        if exc.name in ("cuda", "cuda.bindings"):
            raise ImportError("cuda.core requires cuda-bindings. Install cuda-core[cu12] or cuda-core[cu13]") from None
        raise

    def load_build_module(name: str, cuda_major: int):
        # Prefer this major's build in the merged wheel, then fall back to a plain build.
        try:
            return importlib.import_module(f".cu{cuda_major}.{name}", __package__)
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.cu{cuda_major}":
                raise
        return importlib.import_module(f".{name}", __package__)

    version_str = bindings.__version__
    # The major decides which build to consult. _bindings_floor validates everything else.
    try:
        cuda_major = int(version_str.split(".")[0])
    except ValueError:
        raise ImportError(
            f"cuda.core requires a cuda-bindings release, but the installed cuda-bindings version is {version_str!r}"
        ) from None
    try:
        floor = load_build_module("_bindings_floor", cuda_major)
        info = load_build_module("_build_info", cuda_major)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"This cuda.core installation has no build for CUDA {cuda_major}. "
            f"The installed cuda-bindings is {version_str}."
        ) from exc
    # By module object: the `cuda` namespace package need not carry a `bindings` attribute.
    bindings_driver = importlib.import_module("cuda.bindings.driver")
    floor.check_installed_bindings(
        version_str,
        int(bindings_driver.CUDA_VERSION),
        info.CUDA_MAJOR,
        info.CUDA_VERSION,
        info.CUDA_BINDINGS_FLOOR,
        __version__,
    )

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
    # match getset_descriptor / member_descriptor. Installed unconditionally
    # (the patch is scoped to the rlcompleter module, so non-interactive users
    # only pay for the import).
    import os

    raw_opt_out = os.environ.get("CUDA_CORE_DONT_FIX_TAB_COMPLETION", "").strip()
    try:
        opt_out = int(raw_opt_out) != 0
    except ValueError:
        opt_out = raw_opt_out != ""
    if opt_out:
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
from cuda.core._utils.cuda_utils import CUDAError, CUDAWarning, NVRTCError

__all__ = [
    "CUDAError",
    "CUDAWarning",
    "NVRTCError",
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
