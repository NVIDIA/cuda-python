# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._version import __version__


def _import_versioned_module():
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


def _patch_rlcompleter_for_cython_properties():
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
        def __instancecheck__(cls, inst):
            return isinstance(inst, (property, GetSetDescriptorType, MemberDescriptorType))

    class _PatchedProperty(metaclass=_PatchedPropMeta):
        pass

    rlcompleter.property = _PatchedProperty


_patch_rlcompleter_for_cython_properties()
del _patch_rlcompleter_for_cython_properties


from cuda.core import checkpoint, system, utils
from cuda.core._context import Context, ContextOptions
from cuda.core._device import Device
from cuda.core._device_resources import (
    DeviceResources,
    SMResource,
    SMResourceOptions,
    WorkqueueResource,
    WorkqueueResourceOptions,
)
from cuda.core._event import Event, EventOptions
from cuda.core._graphics import GraphicsResource
from cuda.core._launch_config import LaunchConfig
from cuda.core._launcher import launch
from cuda.core._linker import Linker, LinkerOptions
from cuda.core._memory import (
    Buffer,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphMemoryResource,
    LegacyPinnedMemoryResource,
    ManagedMemoryResource,
    ManagedMemoryResourceOptions,
    MemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core._module import Kernel, ObjectCode
from cuda.core._program import Program, ProgramOptions
from cuda.core._stream import (
    LEGACY_DEFAULT_STREAM,
    PER_THREAD_DEFAULT_STREAM,
    Stream,
    StreamOptions,
)
from cuda.core._tensor_map import TensorMapDescriptor, TensorMapDescriptorOptions

# isort: split
# Must come after the cuda.core._* extension imports above: loading graph
# earlier interacts badly with the merged-wheel __path__ rewrite and leaves
# Graph/GraphBuilder/GraphCompleteOptions/GraphDebugPrintOptions missing from
# cuda.core.graph.
import cuda.core.graph
