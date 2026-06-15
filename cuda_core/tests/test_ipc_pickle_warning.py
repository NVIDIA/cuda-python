# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test warnings when unpickling IPC-enabled Buffer objects."""

import warnings

from cuda.core import Buffer
from cuda.core._utils.cuda_utils import reset_ipc_pickle_warning, warn_ipc_buffer_unpickle

NBYTES = 64


def test_warn_on_buffer_unpickle(ipc_device, ipc_memory_resource):
    """Unpickling an IPC buffer warns about the trust boundary."""
    mr = ipc_memory_resource
    buf = mr.allocate(NBYTES, stream=ipc_device.default_stream)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reset_ipc_pickle_warning()
        Buffer._reduce_helper(mr, buf.ipc_descriptor)

        assert len(w) == 1, f"Expected 1 warning, got {len(w)}: {[str(x.message) for x in w]}"
        warning = w[0]
        assert warning.category is UserWarning
        assert "unpickl" in str(warning.message).lower()
        assert "ipc" in str(warning.message).lower()
        assert "trusted" in str(warning.message).lower()


def test_ipc_pickle_warning_emitted_only_once(ipc_device, ipc_memory_resource):
    """The unpickle trust-boundary warning is emitted at most once per process."""
    mr = ipc_memory_resource
    buf1 = mr.allocate(NBYTES, stream=ipc_device.default_stream)
    buf2 = mr.allocate(NBYTES, stream=ipc_device.default_stream)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reset_ipc_pickle_warning()
        Buffer._reduce_helper(mr, buf1.ipc_descriptor)
        Buffer._reduce_helper(mr, buf2.ipc_descriptor)
        warn_ipc_buffer_unpickle()

        ipc_warnings = [x for x in w if "unpickl" in str(x.message).lower()]
        assert len(ipc_warnings) == 1
