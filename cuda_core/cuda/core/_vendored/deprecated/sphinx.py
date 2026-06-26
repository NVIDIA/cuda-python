# Vendored from the Deprecated package (https://pypi.org/project/Deprecated/),
# version 1.3.1, (c) Laurent LAPORTE, MIT License.
# Modified to remove the dependency on the `wrapt` package.

import re
import textwrap

from cuda.core._vendored.deprecated.classic import ClassicAdapter
from cuda.core._vendored.deprecated.classic import deprecated as _classic_deprecated


class SphinxAdapter(ClassicAdapter):
    """
    Sphinx adapter -- *for advanced usage only*

    This adapter overrides :class:`~deprecated.classic.ClassicAdapter` to add
    Sphinx directives ("versionadded", "versionchanged", "deprecated") to the
    end of the decorated function or class docstring.
    """

    def __init__(
        self,
        directive,
        reason="",
        version="",
        action=None,
        category=DeprecationWarning,
        extra_stacklevel=0,
        line_length=70,
    ):
        if not version:
            raise ValueError("'version' argument is required in Sphinx directives")
        self.directive = directive
        self.line_length = line_length
        super().__init__(
            reason=reason, version=version, action=action, category=category, extra_stacklevel=extra_stacklevel
        )

    def __call__(self, wrapped):
        fmt = ".. {directive}:: {version}" if self.version else ".. {directive}::"
        div_lines = [fmt.format(directive=self.directive, version=self.version)]
        width = self.line_length - 3 if self.line_length > 3 else 2**16
        reason = textwrap.dedent(self.reason).strip()
        for paragraph in reason.splitlines():
            if paragraph:
                div_lines.extend(
                    textwrap.fill(
                        paragraph,
                        width=width,
                        initial_indent="   ",
                        subsequent_indent="   ",
                    ).splitlines()
                )
            else:
                div_lines.append("")

        docstring = wrapped.__doc__ or ""
        lines = docstring.splitlines(True) or [""]
        docstring = textwrap.dedent("".join(lines[1:])) if len(lines) > 1 else ""
        docstring = lines[0] + docstring
        if docstring:
            docstring = re.sub(r"\n+$", "", docstring, flags=re.DOTALL) + "\n\n"
        else:
            docstring = "\n"

        docstring += "".join(f"{line}\n" for line in div_lines)

        wrapped.__doc__ = docstring
        if self.directive in {"versionadded", "versionchanged"}:
            return wrapped
        return super().__call__(wrapped)

    def get_deprecated_msg(self, wrapped, instance):
        msg = super().get_deprecated_msg(wrapped, instance)
        msg = re.sub(r"(?: : [a-zA-Z]+ )? : [a-zA-Z]+ : (`[^`]*`)", r"\1", msg, flags=re.X)
        return msg


def versionadded(reason="", version="", line_length=70):
    """
    Decorator that inserts a "versionadded" Sphinx directive into the docstring.
    """
    return SphinxAdapter(
        "versionadded",
        reason=reason,
        version=version,
        line_length=line_length,
    )


def versionchanged(reason="", version="", line_length=70):
    """
    Decorator that inserts a "versionchanged" Sphinx directive into the docstring.
    """
    return SphinxAdapter(
        "versionchanged",
        reason=reason,
        version=version,
        line_length=line_length,
    )


def deprecated(reason="", version="", line_length=70, **kwargs):
    """
    Decorator that inserts a "deprecated" Sphinx directive into the docstring
    and emits a :exc:`DeprecationWarning` when the decorated object is called.
    """
    directive = kwargs.pop("directive", "deprecated")
    adapter_cls = kwargs.pop("adapter_cls", SphinxAdapter)
    kwargs["reason"] = reason
    kwargs["version"] = version
    kwargs["line_length"] = line_length
    return _classic_deprecated(directive=directive, adapter_cls=adapter_cls, **kwargs)
