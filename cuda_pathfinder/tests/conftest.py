# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os

import pytest

_LOGGER_NAME = "cuda_pathfinder.test_info"


def _info_summary_log_filename():
    strictness = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "default")
    return f"test-info-summary-{strictness}.log"


@pytest.fixture(scope="session")
def _info_summary_handler(request):
    log_path = request.config.rootpath / _info_summary_log_filename()
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(test_node)s: %(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
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
