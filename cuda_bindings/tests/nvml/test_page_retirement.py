# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import nvml

from . import util

PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES = list(range(nvml.PageRetirementCause.COUNT))


def supports_page_retirement(device):
    try:
        for source in range(nvml.PageRetirementCause.COUNT):
            nvml.device_get_retired_pages(device, source)
        return True
    except nvml.NotSupportedError as e:
        return False
    except nvml.NotFoundError as e:
        return False
    except nvml.FunctionNotFoundError as e:
        return False


def test_page_retirement_notsupported(all_devices):
    """
    Verifies that on platforms that don't supports page retirement, APIs will return Not Supported
    """
    skip_reasons = set()

    for device in all_devices:
        if supports_page_retirement(device):
            skip_reasons.add(f"page_retirement is supported for {device}")
            continue

        if not util.supports_ecc(device):
            skip_reasons.add(f"device doesn't support ECC for {device}")
            continue

        with pytest.raises(nvml.NotSupportedError):
            for source in PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES:
                nvml.device_get_retired_pages(device, source)

        with pytest.raises(nvml.NotSupportedError):
            nvml.device_get_retired_pages_pending_status(device)

    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))


def test_page_retirement_supported(all_devices):
    """
    Verifies that on platforms that support page_retirement, APIs will return success
    """
    skip_reasons = set()

    for device in all_devices:
        if not supports_page_retirement(device):
            skip_reasons.add(f"page_retirement not supported for {device}")
            continue

        if not util.supports_ecc(device):
            skip_reasons.add(f"device doesn't support ECC for {device}")
            continue

        try:
            for source in PAGE_RETIREMENT_PUBLIC_CAUSE_TYPES:
                nvml.device_get_retired_pages(device, source)
        except nvml.NotSupportedError:
            skip_reasons.add(f"Exception case: Page retirement is not supported in this GPU {device}")
            continue

        nvml.device_get_retired_pages_pending_status(device)

    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))
