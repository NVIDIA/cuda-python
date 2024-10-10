from cuda.core._memory import Buffer, MemoryResource

class DummyMemoryResource(MemoryResource):
    def __init__(self):
        pass

    def allocate(self, size, stream=None) -> Buffer:
        pass

    def deallocate(self, ptr, size, stream=None):
        pass

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        return 0

def test_buffer_initialization():
    dummy_mr = DummyMemoryResource()
    buffer = Buffer(ptr=1234, size=1024, mr=dummy_mr)
    assert buffer._ptr == 1234
    assert buffer._size == 1024
    assert buffer._mr == dummy_mr