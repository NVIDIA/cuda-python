# Vendored from the Deprecated package (https://pypi.org/project/Deprecated/),
# version 1.3.1, (c) Laurent LAPORTE, MIT License.
# Modified to remove the dependency on the `wrapt` package.

import collections
import functools
import inspect
import warnings


class DeprecatedParams:
    """
    Decorator for functions where one or more parameters are deprecated.
    """

    def __init__(self, param, reason="", category=DeprecationWarning):
        self.messages = {}
        self.category = category
        self.populate_messages(param, reason=reason)

    def populate_messages(self, param, reason=""):
        if isinstance(param, dict):
            self.messages.update(param)
        elif isinstance(param, str):
            fmt = "'{param}' parameter is deprecated"
            reason = reason or fmt.format(param=param)
            self.messages[param] = reason
        else:
            raise TypeError(param)

    def check_params(self, signature, *args, **kwargs):
        binding = signature.bind(*args, **kwargs)
        bound = collections.OrderedDict(binding.arguments, **binding.kwargs)
        return [param for param in bound if param in self.messages]

    def warn_messages(self, messages):
        for message in messages:
            warnings.warn(message, category=self.category, stacklevel=3)

    def __call__(self, f):
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            invalid_params = self.check_params(signature, *args, **kwargs)
            self.warn_messages([self.messages[param] for param in invalid_params])
            return f(*args, **kwargs)

        return wrapper


deprecated_params = DeprecatedParams
