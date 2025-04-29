import os

import pytest

skipif_testing_with_compute_sanitizer = pytest.mark.skipif(
    os.environ.get("CUDA_PYTHON_TESTING_WITH_COMPUTE_SANITIZER", "0") == "1",
    reason="The compute-sanitizer is running, and this test causes an API error.",
)
