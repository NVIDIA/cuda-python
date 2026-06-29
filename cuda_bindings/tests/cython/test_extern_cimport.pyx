# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# Regression test for https://github.com/NVIDIA/cuda-python/issues/1663
#
# Prior to the fix, cynvfatbin and cynvml (and several other cy*.pxd files)
# declared enums and struct typedefs as bare inline Cython definitions rather
# than as `cdef extern from` blocks.  That caused C redefinition errors
# whenever a downstream .pyx file also brought in the same C header — either
# directly or transitively through another library.
#
# After the fix, every cy*.pxd wraps its type declarations in `cdef extern from
# 'header.h':`, so the compiler sees an `#include` directive instead of an
# inline definition, and C header guards prevent any duplication.
#
# This file reproduces the exact failure pattern from the issue using two
# representative modules:
#   - cynvfatbin: covers enum typedef redefinition
#   - cynvml:     covers struct typedef redefinition
# If the fix is absent the C compiler raises redeclaration errors; if the fix
# is present the file compiles and the tests pass.


# Simulate a library whose header transitively includes nvFatbin.h.
cdef extern from "nvFatbin.h":
    pass

cimport cuda.bindings.cynvfatbin as cnvfatbin


def test_cynvfatbin_no_redefinition():
    """Enum constants are accessible and the file compiled without redefinition errors."""
    cdef cnvfatbin.nvFatbinResult result = cnvfatbin.NVFATBIN_SUCCESS
    assert result == cnvfatbin.NVFATBIN_SUCCESS


# Simulate a library whose header transitively includes nvml.h.
cdef extern from "nvml.h":
    pass

cimport cuda.bindings.cynvml as cnvml


def test_cynvml_struct_no_redefinition():
    """Struct fields are accessible and the file compiled without redefinition errors."""
    cdef cnvml.nvmlPSUInfo_t psu
    psu.current = 0
    psu.voltage = 0
    psu.power = 0
    assert psu.current == 0
