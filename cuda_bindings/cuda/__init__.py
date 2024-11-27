def __getattr__(name):
    if name == "__version__":
        import warnings

        warnings.warn(
            "accessing cuda.__version__ is deprecated, " "please switch to use cuda.bindings.__version__ instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import bindings

        return bindings.__version__

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
