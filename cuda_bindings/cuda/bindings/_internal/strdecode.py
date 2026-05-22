# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
"""Decode C strings returned by CUDA libraries with actionable failure context."""

# Cap sized for the #2118 mojibake without flooding exception text.
_PREVIEW_MAX_BYTES = 64


def _bounded_hex_preview(data: bytes, max_bytes: int = _PREVIEW_MAX_BYTES) -> str:
    # Bytes after the first NUL are not part of the returned C string. The
    # marker is explicit so truncation cannot be misread as an empty value.
    nul = data.find(b"\x00")
    nul_stopped = nul != -1
    visible_end = len(data) if not nul_stopped else nul
    snippet_end = min(visible_end, max_bytes)
    snippet = data[:snippet_end]
    body = snippet.hex(" ") if snippet else ""
    parts = []
    if snippet_end < visible_end:
        parts.append(f"+{visible_end - snippet_end} more")
    if nul_stopped:
        parts.append(f"stopped at NUL@{nul}")
    suffix = f" ...({'; '.join(parts)})" if parts else ""
    return f"<{visible_end} bytes; hex='{body}'{suffix}>"


def decode_c_str(data: bytes, api_name: str) -> str:
    """Decode ``data`` as UTF-8, or raise ``UnicodeDecodeError`` with ``api_name`` and a bounded hex preview in ``reason``.

    Internal API. ``api_name`` is trusted caller input and embedded verbatim.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        # Same exception type, not a subclass, so existing handlers still catch.
        preview = _bounded_hex_preview(data)
        reason = f"{e.reason} (returned by {api_name}; bytes={preview})"
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, reason) from e
