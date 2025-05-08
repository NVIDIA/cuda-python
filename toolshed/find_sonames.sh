#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

find "$@" -type f -name '*.so*' -print0 | while IFS= read -r -d '' f; do
  type=$(test -L "$f" && echo SYMLINK || echo FILE)
  soname=$(readelf -d "$f" 2>/dev/null | awk '/SONAME/ {gsub(/[][]/, "", $5); print $5; exit}')
  echo "$f $type ${soname:-SONAME_NOT_SET}"
done
