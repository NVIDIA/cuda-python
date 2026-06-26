# Vendored from the Deprecated package (https://pypi.org/project/Deprecated/),
# version 1.3.1, (c) Laurent LAPORTE, MIT License.
# Modified to remove the dependency on the `wrapt` package.

import functools
import inspect
import warnings

# stacklevel=2 points past the wrapper to the actual call site
_routine_stacklevel = 2
_class_stacklevel = 2

string_types = (bytes, str)


class ClassicAdapter:
    """
    Classic adapter -- *for advanced usage only*

    This adapter is used to get the deprecation message according to the wrapped
    object type: class, function, standard method, static method, or class method.

    This is the base class of the :class:`~deprecated.sphinx.SphinxAdapter` class
    which is used to update the wrapped object docstring.
    """

    def __init__(self, reason="", version="", action=None, category=DeprecationWarning, extra_stacklevel=0):
        self.reason = reason or ""
        self.version = version or ""
        self.action = action
        self.category = category
        self.extra_stacklevel = extra_stacklevel

    def get_deprecated_msg(self, wrapped, instance):
        if instance is None:
            if inspect.isclass(wrapped):
                fmt = "Call to deprecated class {name}."
            else:
                fmt = "Call to deprecated function (or staticmethod) {name}."
        else:
            if inspect.isclass(instance):
                fmt = "Call to deprecated class method {name}."
            else:
                fmt = "Call to deprecated method {name}."
        if self.reason:
            fmt += " ({reason})"
        if self.version:
            fmt += " -- Deprecated since version {version}."
        return fmt.format(name=wrapped.__name__, reason=self.reason or "", version=self.version or "")

    def __call__(self, wrapped):
        if inspect.isclass(wrapped):
            old_new1 = wrapped.__new__

            def wrapped_cls(cls, *args, **kwargs):
                msg = self.get_deprecated_msg(wrapped, None)
                stacklevel = _class_stacklevel + self.extra_stacklevel
                if self.action:
                    with warnings.catch_warnings():
                        warnings.simplefilter(self.action, self.category)
                        warnings.warn(msg, category=self.category, stacklevel=stacklevel)
                else:
                    warnings.warn(msg, category=self.category, stacklevel=stacklevel)
                if old_new1 is object.__new__:
                    return old_new1(cls)
                return old_new1(cls, *args, **kwargs)

            wrapped.__new__ = staticmethod(wrapped_cls)
            return wrapped

        elif inspect.isroutine(wrapped):
            adapter = self

            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                msg = adapter.get_deprecated_msg(wrapped, None)
                stacklevel = _routine_stacklevel + adapter.extra_stacklevel
                if adapter.action:
                    with warnings.catch_warnings():
                        warnings.simplefilter(adapter.action, adapter.category)
                        warnings.warn(msg, category=adapter.category, stacklevel=stacklevel)
                else:
                    warnings.warn(msg, category=adapter.category, stacklevel=stacklevel)
                return wrapped(*args, **kwargs)

            return wrapper

        else:
            raise TypeError(repr(type(wrapped)))


def deprecated(*args, **kwargs):
    """
    Decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """
    if args and isinstance(args[0], string_types):
        kwargs["reason"] = args[0]
        args = args[1:]

    if args and not callable(args[0]):
        raise TypeError(repr(type(args[0])))

    if args:
        adapter_cls = kwargs.pop("adapter_cls", ClassicAdapter)
        adapter = adapter_cls(**kwargs)
        wrapped = args[0]
        return adapter(wrapped)

    return functools.partial(deprecated, **kwargs)
