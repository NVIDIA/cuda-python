# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import numpy as np
import pytest
from cuda.bindings import nvml

from . import util
from .conftest import unsupported_before


def test_gpu_get_module_id(nvml_init):
    # Unique module IDs cannot exceed the number of GPUs on the system
    device_count = nvml.device_get_count_v2()

    for i in range(device_count):
        device = nvml.device_get_handle_by_index_v2(i)
        uuid = nvml.device_get_uuid(device)

        if util.is_vgpu(device):
            continue

        module_id = nvml.device_get_module_id(device)
        assert isinstance(module_id, int)


def test_gpu_get_platform_info(all_devices):
    for device in all_devices:
        if util.is_vgpu(device):
            pytest.skip(f"Not supported on vGPU device {device}")

        # Documentation says Blackwell or newer only, but this does seem to pass
        # on some newer GPUs.

        with unsupported_before(device, None):
            platform_info = nvml.device_get_platform_info(device)

        assert isinstance(platform_info, (nvml.PlatformInfo_v1, nvml.PlatformInfo_v2))


# TODO: Test APIs related to GPU instances, which require specific hardware and root

# def test_gpu_instance(all_devices):
#     for device in all_devices:
#         # Requires root
#         gpu_instance = nvml.device_create_gpu_instance(device, nvml.GpuInstanceProfile.PROFILE_1_SLICE)


def test_conf_compute_attestation_report_t(all_devices):
    report = nvml.ConfComputeGpuAttestationReport()
    assert not hasattr(report, "attestation_report_size")
    assert len(report.attestation_report) == 0
    assert not hasattr(report, "cec_attestation_report_size")
    assert len(report.cec_attestation_report) == 0
    assert len(report.nonce) == 32
    assert report.nonce.dtype == np.uint8


def test_gpu_conf_compute_attestation_report(all_devices):
    for device in all_devices:
        # Documentation says AMPERE or newer
        with unsupported_before(device, None), pytest.raises(nvml.UnknownError):
            # The nonce string is nonsensical, so if this "works", we expect an UnknownError
            nvml.device_get_conf_compute_gpu_attestation_report(device, nonce=b"12345678")


def test_conf_compute_gpu_certificate_t():
    cert = nvml.ConfComputeGpuCertificate()
    assert not hasattr(cert, "cert_chain_size")
    assert len(cert.cert_chain) == 0
    assert not hasattr(cert, "attestation_cert_chain_size")
    assert len(cert.attestation_cert_chain) == 0


def test_conf_compute_gpu_certificate(all_devices):
    for device in all_devices:
        # Documentation says AMPERE or newer
        with unsupported_before(device, None), pytest.raises(nvml.UnknownError):
            # This is expected to fail if the device doesn't have a proper certificate
            nvml.device_get_conf_compute_gpu_certificate(device)
