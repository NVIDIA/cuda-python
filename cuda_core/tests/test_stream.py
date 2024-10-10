from cuda.py._stream import Stream

def test_stream_initialization():
    try:
        stream = Stream()
    except NotImplementedError as e:
        assert True
    else:
        assert False, "Expected NotImplementedError was not raised"