#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


def extract_enum_block(header_file_lines):
    line_iter = iter(header_file_lines)
    for line in line_iter:
        if line == "typedef enum cudaError_enum {":
            closing_line = "} CUresult;"
            python_dict_name = "DRIVER_CU_RESULT_EXPLANATIONS"
            break
        if line == "enum __device_builtin__ cudaError":
            line = next(line_iter)
            assert line == "{", line
            closing_line = "};"
            python_dict_name = "RUNTIME_CUDA_ERROR_EXPLANATIONS"
            break
    else:
        raise RuntimeError("Opening line not found.")
    block = []
    for line in line_iter:
        if line == closing_line:
            break
        block.append(line)
    else:
        raise RuntimeError("Closing line not found.")
    return python_dict_name, block


def parse_enum_doc_and_value_pairs(enum_block):
    entries = []
    comment_lines = []
    inside_comment = False

    for line in enum_block:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("/**"):
            inside_comment = True
            comment = stripped[3:].lstrip()
            if comment:
                comment_lines = [comment]
        elif inside_comment:
            if stripped.endswith("*/"):
                comment = stripped[:-2].strip()
                if comment:
                    comment_lines.append(comment)
                inside_comment = False
            else:
                comment_lines.append(stripped.lstrip("*").strip())
        elif stripped:
            assert stripped.count(",") <= 1, line
            stripped = stripped.replace(",", "")
            flds = stripped.split(" = ")
            assert len(flds) == 2, line
            try:
                val = int(flds[1].strip())
            except Exception as e:
                raise RuntimeError(f"Unexpected {line=!r}") from e
            entries.append((int(val), comment_lines))
            comment_lines = []

    return entries


def emit_python_dict(python_dict_name, entries):
    print(f"{python_dict_name} = {{")
    for val, lines in entries:
        py_lines = []
        continuation_space = ""
        for line in lines:
            if line == r"\deprecated":
                continue
            mod_line = line.replace("\\ref ", "")
            assert "\\" not in mod_line, line
            mod_line = mod_line.replace('"', '\\"')
            py_lines.append(f'"{continuation_space}{mod_line}"')
            continuation_space = " "
        assert py_lines, lines
        if len(py_lines) == 1:
            print(f"    {val}: {py_lines[0]},")
        else:
            print(f"    {val}: (")
            for py_line in py_lines:
                print(f"        {py_line}")
            print("    ),")
    print("}")


def run(args):
    if len(args) != 1:
        print(
            "Usage: reformat_cuda_enums_as_py.py /path/to/cuda.h|driver_types.h",
            file=sys.stderr,
        )
        sys.exit(1)

    header_file_text = Path(sys.argv[1]).read_text().splitlines()
    python_dict_name, enum_block = extract_enum_block(header_file_text)
    entries = parse_enum_doc_and_value_pairs(enum_block)
    emit_python_dict(python_dict_name, entries)


if __name__ == "__main__":
    run(sys.argv[1:])
