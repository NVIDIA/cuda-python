import os

import pytest

skipif_testing_with_compute_sanitizer = pytest.mark.skipif(
    os.environ.get("CUDA_PYTHON_TESTING_WITH_COMPUTE_SANITIZER", "0") == "1",
    reason="The compute-sanitizer is running, and this test causes an API error.",
)


def pytest_configure(config):
    config.custom_info = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.custom_info:
        terminalreporter.write_sep("=", "INFO summary")
        for msg in config.custom_info:
            terminalreporter.line(f"INFO {msg}")


@pytest.fixture
def info_summary_append(request):
    def _append(message):
        request.config.custom_info.append(f"{request.node.name}: {message}")

    return _append
