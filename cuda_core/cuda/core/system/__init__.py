# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# NOTE: We must maintain that it is always possible to import this module
# without CUDA being installed, and without CUDA being initialized or any
# contexts created, so that a user can use NVML to explore things about their
# system without loading CUDA.

from typing import TYPE_CHECKING

__all__ = [
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
    "get_driver_branch",
    "get_kernel_mode_driver_version",
    "get_num_devices",
    "get_nvml_version",
    "get_process_name",
    "get_user_mode_driver_version",
]


from cuda.core.system import typing

from ._system import *

# The TYPE_CHECKING branch is split out from the runtime branch so that
# stubgen-pyx, which only recognizes the literal `if TYPE_CHECKING:` form,
# preserves these imports in the generated .pyi.  When
# CUDA_BINDINGS_NVML_IS_COMPATIBLE is no longer necessary, this complexity can
# be removed.
if TYPE_CHECKING:
    from ._device import *
    from ._system_events import *
    from .exceptions import *
elif CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from ._device import *
    from ._device import __all__ as _device_all
    from ._system_events import *
    from ._system_events import __all__ as _system_events_all
    from .exceptions import *
    from .exceptions import __all__ as _exceptions_all

    __all__.extend(_device_all)
    __all__.extend(_system_events_all)
    __all__.extend(_exceptions_all)
