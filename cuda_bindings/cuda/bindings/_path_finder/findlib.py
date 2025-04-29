# SPDX-License-Identifier: BSD-2-Clause
#
# Forked from:
# https://github.com/numba/numba/blob/f0d24824fcd6a454827e3c108882395d00befc04/numba/misc/findlib.py
#
# Original LICENSE:
# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import sys


def get_lib_dirs():
    """
    Anaconda specific
    """
    if sys.platform == "win32":
        # on windows, historically `DLLs` has been used for CUDA libraries,
        # since approximately CUDA 9.2, `Library\bin` has been used.
        dirnames = ["DLLs", os.path.join("Library", "bin")]
    else:
        dirnames = [
            "lib",
        ]
    libdirs = [os.path.join(sys.prefix, x) for x in dirnames]
    return libdirs


DLLNAMEMAP = {
    "linux": r"lib%(name)s\.so\.%(ver)s$",
    "linux2": r"lib%(name)s\.so\.%(ver)s$",
    "linux-static": r"lib%(name)s\.a$",
    "darwin": r"lib%(name)s\.%(ver)s\.dylib$",
    "win32": r"%(name)s%(ver)s\.dll$",
    "win32-static": r"%(name)s\.lib$",
    "bsd": r"lib%(name)s\.so\.%(ver)s$",
}

RE_VER = r"[0-9]*([_\.][0-9]+)*"


def find_lib(libname, libdir=None, platform=None, static=False):
    platform = platform or sys.platform
    platform = "bsd" if "bsd" in platform else platform
    if static:
        platform = f"{platform}-static"
    if platform not in DLLNAMEMAP:
        # Return empty list if platform name is undefined.
        # Not all platforms define their static library paths.
        return []
    pat = DLLNAMEMAP[platform] % {"name": libname, "ver": RE_VER}
    regex = re.compile(pat)
    return find_file(regex, libdir)


def find_file(pat, libdir=None):
    if libdir is None:
        libdirs = get_lib_dirs()
    elif isinstance(libdir, str):
        libdirs = [
            libdir,
        ]
    else:
        libdirs = list(libdir)
    files = []
    for ldir in libdirs:
        try:
            entries = os.listdir(ldir)
        except FileNotFoundError:
            continue
        candidates = [os.path.join(ldir, ent) for ent in entries if pat.match(ent)]
        files.extend([c for c in candidates if os.path.isfile(c)])
    return files
