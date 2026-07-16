# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def format_or_list(values):
    """Format an iterable of values as a human-readable ``A, B or C`` string.

    Each value is passed through :func:`repr`. A single value is returned
    as-is; two values are joined with ``" or "``; three or more use a
    comma-separated list with ``" or "`` before the last item.
    """
    reprs = [repr(v) for v in values]
    if len(reprs) <= 1:
        return reprs[0] if reprs else ""
    *head, tail = reprs
    return ", ".join(head) + " or " + tail


def check_str_enum(value, enum_class, *, allow_none=False):
    """Raise ValueError if *value* is not a member of *enum_class*.

    Derives the list of acceptable values from the enum itself so callers
    do not need to maintain a parallel copy of the valid strings.

    If *allow_none* is True, ``None`` is also accepted and included in the
    error message when an invalid value is provided.
    """
    if allow_none and value is None:
        return
    if value not in {m.value for m in enum_class}:
        valid = sorted(m.value for m in enum_class)
        if allow_none:
            valid = [None, *valid]
        raise ValueError(f"{value!r} is not a valid {enum_class.__name__}. Must be {format_or_list(valid)}")
