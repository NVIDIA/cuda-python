import pytest
from cuda import cuda
from cuda.core.experimental._device import Device
from cuda.core.experimental._context import Context
from cuda.core.experimental._utils import handle_return

@pytest.fixture(scope="module")
def init_cuda():
    device = Device()
    device.set_current()

@pytest.fixture(scope="function")
def reestablish_valid_context():
    yield
    device = Device()
    device.set_current()