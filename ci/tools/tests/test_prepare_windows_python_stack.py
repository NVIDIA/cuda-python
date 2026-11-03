# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test executable patching without requiring Windows or MSVC."""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_windows_python_stack.py"
spec = importlib.util.spec_from_file_location("prepare_windows_python_stack", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _pe_image(magic=0x20B, reserve=2_000_000):
    # DOS stub, COFF header, optional header, and sentinel payload. Keep the
    # commit/heap values nonzero to detect writes beyond the reserve field.
    image = bytearray(512)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", image, 0x94, 224 if magic == 0x10B else 240)
    struct.pack_into("<H", image, 0x98, magic)
    struct.pack_into("<IIII" if magic == 0x10B else "<QQQQ", image, 0xE0, reserve, 4096, 1048576, 4096)
    image[0x188:] = b"\xa5" * (len(image) - 0x188)
    return image


@pytest.mark.parametrize("magic,width", [(0x10B, 4), (0x20B, 8)])
@pytest.mark.agent_authored(model="gpt-6")
def test_raises_reserve_without_changing_source_or_other_fields(tmp_path, magic, width):
    original = _pe_image(magic)
    source = tmp_path / "python.exe"
    destination = tmp_path / "python.exe.stack-reserve"
    source.write_bytes(original)

    assert tool.prepare_executable(source, destination) == (2_000_000, 8_388_608)

    patched = destination.read_bytes()
    assert source.read_bytes() == original
    assert len(patched) == len(original)
    assert int.from_bytes(patched[0xE0 : 0xE0 + width], "little") == 8_388_608
    assert patched[:0xE0] == original[:0xE0]
    assert patched[0xE0 + width :] == original[0xE0 + width :]


@pytest.mark.parametrize("magic,reserve", [(0x10B, 8_388_608), (0x20B, 16_777_216), (0x20B, 1 << 33)])
@pytest.mark.agent_authored(model="gpt-6")
def test_preserves_existing_equal_or_larger_reserve(tmp_path, magic, reserve):
    source = tmp_path / "python.exe"
    destination = tmp_path / "patched.exe"
    original = _pe_image(magic, reserve)
    source.write_bytes(original)

    assert tool.prepare_executable(source, destination) == (reserve, reserve)
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "offset,replacement,message",
    [
        (0, b"EL", "DOS header"),
        (0x3C, struct.pack("<I", 0), "PE header"),
        (0x3C, struct.pack("<I", 0xFFFF), "PE header"),
        (0x80, b"BAD!", "PE header"),
        (0x94, struct.pack("<H", 1), "Truncated PE optional header"),
        (0x94, struct.pack("<H", 80), "Truncated PE optional header"),
        (0x94, struct.pack("<H", 0xFFFF), "Truncated PE optional header"),
        (0x98, struct.pack("<H", 0x107), "Unsupported PE optional header magic"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_rejects_invalid_headers_without_creating_output(tmp_path, offset, replacement, message):
    original = _pe_image()
    original[offset : offset + len(replacement)] = replacement
    source = tmp_path / "python.exe"
    destination = tmp_path / "patched.exe"
    source.write_bytes(original)

    with pytest.raises(ValueError, match=message):
        tool.prepare_executable(source, destination)
    assert source.read_bytes() == original
    assert not destination.exists()


@pytest.mark.parametrize("length", [0, 2, 63, 0x90, 0xA0, 0xE4])
@pytest.mark.agent_authored(model="gpt-6")
def test_rejects_truncated_images(tmp_path, length):
    source = tmp_path / "python.exe"
    destination = tmp_path / "patched.exe"
    source.write_bytes(_pe_image()[:length])

    with pytest.raises(ValueError, match="header"):
        tool.prepare_executable(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_refuses_to_overwrite_an_existing_file(tmp_path, in_place):
    source = tmp_path / "python.exe"
    source.write_bytes(_pe_image())
    destination = source if in_place else tmp_path / "patched.exe"
    if not in_place:
        destination.write_bytes(b"existing output")
    original = source.read_bytes()
    existing_output = destination.read_bytes()

    with pytest.raises(FileExistsError):
        tool.prepare_executable(source, destination)
    assert source.read_bytes() == original
    assert destination.read_bytes() == existing_output


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_works_without_site_packages_and_allows_shell_replacement(tmp_path):
    source = tmp_path / "python with spaces.exe"
    destination = tmp_path / "python with spaces.exe.stack-reserve"
    source.write_bytes(_pe_image())
    result = subprocess.run(  # noqa: S603 - repository-owned script with test-owned files.
        [sys.executable, "-I", "-S", str(SCRIPT), str(source), str(destination)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert "stack reserve 2000000 -> 8388608 bytes" in result.stdout
    assert result.stderr == ""
    # The caller can replace the executable once the patching process exits.
    destination.replace(source)
    assert int.from_bytes(source.read_bytes()[0xE0:0xE8], "little") == 8_388_608
