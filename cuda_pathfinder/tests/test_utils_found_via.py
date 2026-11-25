# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.pathfinder._utils import found_via


@pytest.mark.agent_authored(model="claude-fable-5-2")
def test_found_via_labels_keep_their_public_values():
    # Callers compare ``found_via`` against these exact strings.
    assert found_via.FOUND_VIA_SITE_PACKAGES == "site-packages"
    assert found_via.FOUND_VIA_CONDA == "conda"
    assert found_via.FOUND_VIA_SYSTEM_SEARCH == "system-search"
    assert found_via.FOUND_VIA_CUDA_PATH == "CUDA_PATH"
    assert found_via.FOUND_VIA_PROGRAM_FILES == "ProgramFiles"
    assert found_via.FOUND_VIA_SYSTEM_CTK_ROOT == "system-ctk-root"
    assert found_via.FOUND_VIA_SUPPORTED_INSTALL_DIR == "supported_install_dir"
    assert found_via.FOUND_VIA_WAS_ALREADY_LOADED_FROM_ELSEWHERE == "was-already-loaded-from-elsewhere"
