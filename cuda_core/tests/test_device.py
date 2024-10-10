from cuda.py._device import Device

def test_device_initialization():
    device = Device()
    assert device is not None