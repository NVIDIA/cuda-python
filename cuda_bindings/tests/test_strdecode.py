# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.bindings._internal.strdecode import _bounded_hex_preview, decode_c_str

WSL_MOJIBAKE_PREFIX = b"\xf8\x9a\x80\x80\xaf"


def test_valid_utf8_passthrough():
    assert decode_c_str(b"hello world", "fakeApi") == "hello world"


def test_invalid_bytes_raise_unicode_decode_error():
    with pytest.raises(UnicodeDecodeError):
        decode_c_str(WSL_MOJIBAKE_PREFIX, "nvmlSystemGetProcessName")


def test_failure_reason_includes_api_name():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(WSL_MOJIBAKE_PREFIX, "nvmlSystemGetProcessName")
    assert "nvmlSystemGetProcessName" in excinfo.value.reason


def test_failure_reason_includes_hex_preview():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(WSL_MOJIBAKE_PREFIX, "nvmlSystemGetProcessName")
    assert "f8 9a 80 80 af" in excinfo.value.reason


def test_failure_chains_original_error():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(b"\xf8", "fakeApi")
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_failure_preserves_codec_and_position():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(b"\xf8\x9a", "fakeApi")
    assert excinfo.value.encoding == "utf-8"
    assert excinfo.value.start == 0
    assert excinfo.value.end == 1


def test_preview_stops_at_first_nul():
    preview = _bounded_hex_preview(b"\xf8\xf8\x00trailing junk")
    assert "f8 f8" in preview
    assert "trailing" not in preview
    assert "<2 bytes;" in preview
    assert "stopped at NUL@2" in preview


def test_preview_caps_long_buffers():
    preview = _bounded_hex_preview(b"\xf8" * 200, max_bytes=8)
    assert "f8 f8 f8 f8 f8 f8 f8 f8" in preview
    assert "+192 more" in preview
    assert "stopped at NUL" not in preview


def test_preview_combines_truncation_and_nul_markers():
    preview = _bounded_hex_preview(b"\xf8" * 20 + b"\x00rest", max_bytes=8)
    assert "+12 more" in preview
    assert "stopped at NUL@20" in preview


def test_failure_preview_stops_at_embedded_nul_even_with_bad_bytes_before():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(b"\xf8\x9a\x00ignored_after_nul", "fakeApi")
    reason = excinfo.value.reason
    assert "f8 9a" in reason
    assert "ignored_after_nul" not in reason


def test_failure_message_stays_bounded_for_long_garbage():
    with pytest.raises(UnicodeDecodeError) as excinfo:
        decode_c_str(b"\xf8" * 1024, "fakeApi")
    reason = excinfo.value.reason
    assert "+960 more" in reason
    assert len(reason) < 500
