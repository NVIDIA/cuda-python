from cuda.core._event import Event

def test_event_initialization():
    try:
        event = Event()
    except NotImplementedError as e:
        assert True
    else:
        assert False, "Expected NotImplementedError was not raised"