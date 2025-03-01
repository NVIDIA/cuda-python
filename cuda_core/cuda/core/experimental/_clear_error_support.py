def assert_type(obj, expected_type):
    """Ensure obj is of expected_type, else raise AssertionError with a clear message."""
    if not isinstance(obj, expected_type):
        raise AssertionError(f"Expected type {expected_type.__name__}, but got {type(obj).__name__}")
