# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

if sys.platform != "linux":
    pytest.skip("Linux dynamic-loader tests", allow_module_level=True)

from cuda.pathfinder._dynamic_libs import load_dl_linux
from cuda.pathfinder._dynamic_libs.descriptor_catalog import DescriptorSpec


def _descriptor() -> DescriptorSpec:
    return DescriptorSpec(
        name="probe",
        packaged_with="other",
        linux_sonames=("libprobe.so.13", "libprobe.so.12"),
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_already_loaded_library_checks_declared_sonames_in_preference_order(mocker):
    queried_sonames: list[tuple[str, int]] = []

    def cdll(soname, mode):
        queried_sonames.append((soname, mode))
        raise OSError

    mocker.patch.object(load_dl_linux.ctypes, "CDLL", side_effect=cdll)

    loaded = load_dl_linux.check_if_already_loaded_from_elsewhere(_descriptor())

    assert loaded is None
    assert queried_sonames == [
        ("libprobe.so.13", os.RTLD_NOLOAD),
        ("libprobe.so.12", os.RTLD_NOLOAD),
    ]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_system_search_checks_declared_sonames_in_preference_order(mocker):
    queried_sonames: list[str] = []

    def load_lib(_desc, soname):
        queried_sonames.append(soname)
        raise OSError

    mocker.patch.object(load_dl_linux, "_load_lib", side_effect=load_lib)

    loaded = load_dl_linux.load_with_system_search(_descriptor())

    assert loaded is None
    assert queried_sonames == ["libprobe.so.13", "libprobe.so.12"]
