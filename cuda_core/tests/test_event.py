from cuda import cuda
from cuda.core.experimental._event import  EventOptions, Event
from cuda.core.experimental._utils import handle_return

def test_is_timing_disabled():
    options = EventOptions(enable_timing=False)
    event = Event._init(options)
    assert event.is_timing_disabled == True

def test_is_sync_busy_waited():
    options = EventOptions(busy_waited_sync=True)
    event = Event._init(options)
    assert event.is_sync_busy_waited == True

def test_is_ipc_supported():
    options = EventOptions(support_ipc=True)
    try:
        event = Event._init(options)
    except NotImplementedError:
        assert True
    else:
        assert False

def test_sync():
    options = EventOptions()
    event = Event._init(options)
    event.sync()
    assert event.is_done == True

def test_is_done():
    options = EventOptions()
    event = Event._init(options)
    assert event.is_done == True

def test_handle():
    options = EventOptions()
    event = Event._init(options)
    assert isinstance(event.handle, int)
