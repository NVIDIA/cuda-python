# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cdef extern from "<dlfcn.h>" nogil:
    void *dlopen(const char *, int)
    char *dlerror()
    void *dlsym(void *, const char *)
    int dlclose(void *)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL
