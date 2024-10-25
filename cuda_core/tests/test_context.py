from cuda.core.experimental._context import Context

def test_context_initialization():
    try:
        context = Context()
    except NotImplementedError as e:
        assert True
    else:
        assert False, "Expected NotImplementedError was not raised"