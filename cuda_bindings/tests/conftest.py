import os

import pytest

skipif_compute_sanitizer_is_running = pytest.mark.skipif(
    os.environ.get("CUDA_PYTHON_SANITIZER_RUNNING", "0") == "1",
    reason="The compute-sanitizer is running, and this test causes an API error.",
)
