# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import _nvml as nvml

from . import util

PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES = list(range(nvml.PageRetirementCause.COUNT))


def supports_page_retirement(device):
    try:
        for source in range(nvml.PageRetirementCause.COUNT):
            nvml.device_get_retired_pages(device, source)
        return True
    except nvml.NotSupportedError as e:
        return False
    except nvml.FunctionNotFoundError as e:
        return False


def test_page_retirement_notsupported(for_all_devices):
    """
    Verifies that on platforms that don't supports page retirement, APIs will return Not Supported
    """
    device = for_all_devices

    if supports_page_retirement(device):
        pytest.skip("page_retirement not supported")

    if not util.supports_ecc(device):
        pytest.skip("device doesn't support ECC")

    with pytest.raises(nvml.NotSupportedError):
        for source in PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES:
            nvml.device_get_retired_pages(device, source)

    with pytest.raises(nvml.NotSupportedError):
        nvml.device_get_retired_pages_pending_status(device)


def test_page_retirement_supported(for_all_devices):
    """
    Verifies that on platforms that support page_retirement, APIs will return success
    """
    device = for_all_devices

    if not supports_page_retirement(device):
        pytest.skip("page_retirement not supported")

    if not util.supports_ecc(device):
        pytest.skip("device doesn't support ECC")

    try:
        for source in PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES:
            nvml.device_get_retired_pages(device, source)
    except nvml.NotSupportedError:
        pytest.skip("Exception case: Page retirment is not supported in this GPU")

    nvml.device_get_retired_pages_pending_status(device)
