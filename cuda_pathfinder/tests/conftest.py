# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os

import pytest

_LOGGER_NAME = "cuda_pathfinder.test_info"


@pytest.fixture(scope="session")
def _info_summary_handler(request):
    strictness = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "default")
    log_path = request.config.rootpath / f"test-info-summary-{strictness}.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(test_node)s: %(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    yield handler

    logger.removeHandler(handler)
    handler.close()


@pytest.fixture
def info_log(request, _info_summary_handler):
    return logging.LoggerAdapter(
        logging.getLogger(_LOGGER_NAME),
        extra={"test_node": request.node.name},
    )
